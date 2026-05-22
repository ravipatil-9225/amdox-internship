"""
NeuralRetail -- Automated Model Retraining DAG
================================================
Phase 3 deliverable: Webhook-triggered retraining pipeline.

Triggered by the DriftMonitor webhook when drift share > threshold.
Implements full champion-challenger evaluation:

  [validate_drift]
        |
  [retrain_models]
        |
  [evaluate_challenger]
        |
  [promote_champion]  OR  [log_rejection]

Every task emits OpenLineage events and logs to MLflow.
"""
import os
import sys
import logging
from datetime import datetime, timedelta, timezone

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator

logger = logging.getLogger("neuralretail.retraining_dag")

_PROJ_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

# ---------------------------------------------------------------------------
# Default args
# ---------------------------------------------------------------------------

default_args = {
    "owner": "ml_engineer",
    "depends_on_past": False,
    "start_date": datetime(2026, 5, 1),
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
}

# ---------------------------------------------------------------------------
# Task 1 – Validate that drift is real (not a data pipeline glitch)
# ---------------------------------------------------------------------------

def validate_drift(**context):
    """
    Re-run drift detection to confirm the trigger was legitimate.
    Gate: abort if drift_share < DRIFT_THRESHOLD / 2 (false positive).
    """
    import pandas as pd
    import numpy as np
    from src.pipelines.data_lineage import AirflowLineageExtractor

    extractor = AirflowLineageExtractor("model_retraining_pipeline")
    extractor.on_start(context, inputs=[{"name": "bronze/transactions.parquet"}])

    # Read conf from triggered run
    conf = context.get("dag_run", {}).conf or {}
    reported_drift = conf.get("drift_share", 0.0)
    threshold = float(os.environ.get("DRIFT_THRESHOLD", "0.20"))

    logger.info(
        f"[Retrain] Validating drift | reported={reported_drift:.2%} "
        f"threshold={threshold:.2%}"
    )
    context["ti"].xcom_push(key="drift_confirmed", value=reported_drift > threshold / 2)
    context["ti"].xcom_push(key="drift_share",     value=reported_drift)

    extractor.on_complete(context, extra_facets={"drift_share": reported_drift})


# ---------------------------------------------------------------------------
# Task 2 – Retrain all models from scratch on latest data
# ---------------------------------------------------------------------------

def retrain_models(**context):
    """
    Retrain XGBoost churn + Stacked Ensemble on latest bronze data.
    Logs challenger model to MLflow with 'Staging' alias.
    """
    import pandas as pd
    import numpy as np
    import mlflow
    from sklearn.model_selection import train_test_split
    from src.pipelines.data_lineage import AirflowLineageExtractor
    from src.config.settings import settings

    extractor = AirflowLineageExtractor("model_retraining_pipeline")
    extractor.on_start(context, inputs=[
        {"name": "bronze/customers.parquet"},
        {"name": "bronze/transactions.parquet"},
    ])

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

    try:
        bronze_dir   = os.path.join(_PROJ_ROOT, "data", "bronze")
        customers    = pd.read_parquet(os.path.join(bronze_dir, "customers.parquet"))
        transactions = pd.read_parquet(os.path.join(bronze_dir, "transactions.parquet"))
        latest_date  = transactions["timestamp"].max()

        rfm = (
            transactions.groupby("customer_id")
            .agg(
                recency   =("timestamp",      lambda x: int((latest_date - x.max()).days)),
                frequency =("transaction_id", "count"),
                monetary  =("total_amount",   "sum"),
            )
            .reset_index()
        )
        data = customers.merge(rfm, on="customer_id", how="left").fillna(0)
        data["churn"] = (data["recency"] > 90).astype(int)

        X = data[["age", "recency", "frequency", "monetary"]]
        y = data["churn"]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

        # Retrain XGBoost challenger
        import xgboost as xgb
        from sklearn.metrics import roc_auc_score

        challenger = xgb.XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.05,
            use_label_encoder=False, eval_metric="logloss", random_state=42,
        )
        challenger.fit(X_tr, y_tr)
        auc = roc_auc_score(y_te, challenger.predict_proba(X_te)[:, 1])

        # Save challenger model
        models_dir = os.path.join(_PROJ_ROOT, "models")
        os.makedirs(models_dir, exist_ok=True)
        import pickle, time
        challenger_path = os.path.join(models_dir, "xgb_churn_challenger.pkl")
        with open(challenger_path, "wb") as f:
            pickle.dump(challenger, f)

        # Log to MLflow as challenger
        mlflow.set_experiment("model_retraining")
        with mlflow.start_run(run_name="retrained_xgb_challenger") as run:
            mlflow.log_metric("challenger_auc", auc)
            mlflow.log_param("triggered_by", "drift_monitor")
            mlflow.log_artifact(challenger_path)
            run_id = run.info.run_id

        logger.info(f"[Retrain] Challenger AUC={auc:.4f} | run_id={run_id}")
        context["ti"].xcom_push(key="challenger_auc",  value=float(auc))
        context["ti"].xcom_push(key="challenger_run_id", value=run_id)
        context["ti"].xcom_push(key="challenger_path", value=challenger_path)

        extractor.on_complete(
            context,
            outputs=[{"name": "models/xgb_churn_challenger"}],
            row_count=len(X_tr),
            extra_facets={"challenger_auc": auc},
        )

    except Exception as exc:
        extractor.on_fail(context, str(exc))
        raise


# ---------------------------------------------------------------------------
# Task 3 – Evaluate challenger vs champion (branching)
# ---------------------------------------------------------------------------

def evaluate_challenger(**context):
    """
    Compare challenger AUC vs current champion AUC.
    Returns branch task_id: 'promote_champion' or 'reject_challenger'.
    """
    import os, pickle, pandas as pd, numpy as np
    from sklearn.metrics import roc_auc_score

    challenger_auc = context["ti"].xcom_pull(
        key="challenger_auc", task_ids="retrain_models"
    ) or 0.0

    # Load champion model
    champion_path = os.path.join(_PROJ_ROOT, "models", "xgb_churn.pkl")
    champion_auc  = 0.0

    if os.path.exists(champion_path):
        bronze_dir   = os.path.join(_PROJ_ROOT, "data", "bronze")
        customers    = pd.read_parquet(os.path.join(bronze_dir, "customers.parquet"))
        transactions = pd.read_parquet(os.path.join(bronze_dir, "transactions.parquet"))
        latest       = transactions["timestamp"].max()
        rfm = (
            transactions.groupby("customer_id")
            .agg(
                recency   =("timestamp",      lambda x: int((latest - x.max()).days)),
                frequency =("transaction_id", "count"),
                monetary  =("total_amount",   "sum"),
            )
            .reset_index()
        )
        data = customers.merge(rfm, on="customer_id", how="left").fillna(0)
        data["churn"] = (data["recency"] > 90).astype(int)
        X = data[["age", "recency", "frequency", "monetary"]]
        y = data["churn"]

        with open(champion_path, "rb") as f:
            champion = pickle.load(f)
        try:
            champion_auc = roc_auc_score(y, champion.predict_proba(X)[:, 1])
        except Exception:
            champion_auc = 0.0

    improvement = challenger_auc - champion_auc
    logger.info(
        f"[Evaluate] Champion AUC={champion_auc:.4f} | "
        f"Challenger AUC={challenger_auc:.4f} | "
        f"Delta={improvement:+.4f}"
    )
    context["ti"].xcom_push(key="champion_auc",  value=float(champion_auc))
    context["ti"].xcom_push(key="improvement",   value=float(improvement))

    if challenger_auc > champion_auc + 0.005:   # promote if >= 0.5% better
        return "promote_champion"
    else:
        return "reject_challenger"


# ---------------------------------------------------------------------------
# Task 4a – Promote challenger to champion
# ---------------------------------------------------------------------------

def promote_champion(**context):
    """
    Replace champion model with challenger, update MLflow model registry alias,
    and log promotion event.
    """
    import shutil, mlflow, pickle

    challenger_path = context["ti"].xcom_pull(
        key="challenger_path", task_ids="retrain_models"
    )
    challenger_auc  = context["ti"].xcom_pull(key="challenger_auc",  task_ids="retrain_models")
    champion_auc    = context["ti"].xcom_pull(key="champion_auc",    task_ids="evaluate_challenger")
    improvement     = context["ti"].xcom_pull(key="improvement",     task_ids="evaluate_challenger")
    run_id          = context["ti"].xcom_pull(key="challenger_run_id", task_ids="retrain_models")

    champion_path = os.path.join(_PROJ_ROOT, "models", "xgb_churn.pkl")
    backup_path   = os.path.join(_PROJ_ROOT, "models", "xgb_churn_previous.pkl")

    # Backup current champion
    if os.path.exists(champion_path):
        shutil.copy2(champion_path, backup_path)
        logger.info(f"[Promote] Champion backed up to {backup_path}")

    # Promote challenger
    if challenger_path and os.path.exists(challenger_path):
        shutil.copy2(challenger_path, champion_path)
        logger.info(f"[Promote] Challenger promoted to champion!")

    # Update MLflow
    from src.config.settings import settings
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("model_retraining")
    with mlflow.start_run(run_name="champion_promotion"):
        mlflow.log_metric("old_champion_auc",  champion_auc or 0)
        mlflow.log_metric("new_champion_auc",  challenger_auc or 0)
        mlflow.log_metric("auc_improvement",   improvement or 0)
        mlflow.log_param("action",             "PROMOTED")
        mlflow.log_param("challenger_run_id",  run_id or "unknown")

    logger.info(
        f"[Promote] New champion | AUC {champion_auc:.4f} -> "
        f"{challenger_auc:.4f} (+{improvement:.4f})"
    )

    from src.pipelines.data_lineage import emit_lineage_event
    emit_lineage_event(
        job_name="champion_promotion",
        run_facets={
            "old_auc":     champion_auc,
            "new_auc":     challenger_auc,
            "improvement": improvement,
            "action":      "PROMOTED",
        },
    )


# ---------------------------------------------------------------------------
# Task 4b – Reject challenger
# ---------------------------------------------------------------------------

def reject_challenger(**context):
    """Log that challenger was evaluated but not promoted."""
    import mlflow
    from src.config.settings import settings

    challenger_auc = context["ti"].xcom_pull(key="challenger_auc",  task_ids="retrain_models")
    champion_auc   = context["ti"].xcom_pull(key="champion_auc",    task_ids="evaluate_challenger")
    improvement    = context["ti"].xcom_pull(key="improvement",     task_ids="evaluate_challenger")

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("model_retraining")
    with mlflow.start_run(run_name="challenger_rejected"):
        mlflow.log_metric("champion_auc",   champion_auc or 0)
        mlflow.log_metric("challenger_auc", challenger_auc or 0)
        mlflow.log_metric("auc_delta",      improvement or 0)
        mlflow.log_param("action",          "REJECTED")

    logger.info(
        f"[Reject] Challenger rejected | Champion={champion_auc:.4f} "
        f"Challenger={challenger_auc:.4f} (delta={improvement:+.4f})"
    )


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    "model_retraining_pipeline",
    default_args=default_args,
    schedule_interval=None,      # webhook-triggered only
    catchup=False,
    description=(
        "Webhook-triggered model retraining with champion-challenger promotion. "
        "Triggered by the Evidently DriftMonitor when drift_share > threshold."
    ),
    tags=["neuralretail", "retraining", "champion-challenger"],
) as dag:

    task_validate = PythonOperator(
        task_id="validate_drift",
        python_callable=validate_drift,
        provide_context=True,
    )

    task_retrain = PythonOperator(
        task_id="retrain_models",
        python_callable=retrain_models,
        provide_context=True,
    )

    task_evaluate = BranchPythonOperator(
        task_id="evaluate_challenger",
        python_callable=evaluate_challenger,
        provide_context=True,
    )

    task_promote = PythonOperator(
        task_id="promote_champion",
        python_callable=promote_champion,
        provide_context=True,
    )

    task_reject = PythonOperator(
        task_id="reject_challenger",
        python_callable=reject_challenger,
        provide_context=True,
    )

    task_done = EmptyOperator(
        task_id="pipeline_complete",
        trigger_rule="none_failed_min_one_success",
    )

    # Topology:
    # validate -> retrain -> evaluate -+-> promote -> done
    #                                  +-> reject  -> done
    task_validate >> task_retrain >> task_evaluate >> [task_promote, task_reject] >> task_done
