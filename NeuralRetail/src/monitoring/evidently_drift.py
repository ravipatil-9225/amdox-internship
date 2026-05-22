"""
NeuralRetail -- Evidently Drift Monitor + Airflow Webhook
==========================================================
Compatible with Evidently 0.4.x (used in Poetry environment).
"""
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("neuralretail.drift")

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------

DRIFT_THRESHOLD      = float(os.environ.get("DRIFT_THRESHOLD", "0.20"))
AIRFLOW_BASE_URL     = os.environ.get("AIRFLOW_BASE_URL",         "http://localhost:8080")
AIRFLOW_DAG_ID       = os.environ.get("AIRFLOW_RETRAINING_DAG",   "model_retraining_pipeline")
AIRFLOW_USER         = os.environ.get("AIRFLOW_USER",             "admin")
AIRFLOW_PASSWORD     = os.environ.get("AIRFLOW_PASSWORD",         "admin")
PUSHGATEWAY_URL      = os.environ.get("PROMETHEUS_PUSHGATEWAY",   "")
MLFLOW_TRACKING_URI  = os.environ.get("MLFLOW_TRACKING_URI",      "file:./mlruns")

REPORTS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "reports", "drift"
)
os.makedirs(REPORTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Airflow webhook
# ---------------------------------------------------------------------------

def _trigger_airflow_dag(dag_id: str, conf: dict) -> dict:
    try:
        import requests
    except ImportError:
        return {"success": False, "run_id": None, "error": "requests not installed"}

    url = f"{AIRFLOW_BASE_URL}/api/v1/dags/{dag_id}/dagRuns"
    payload = {"conf": conf, "note": "Triggered by NeuralRetail drift monitor"}
    try:
        resp = requests.post(
            url, json=payload,
            auth=(AIRFLOW_USER, AIRFLOW_PASSWORD),
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        if resp.status_code in (200, 201):
            run_id = resp.json().get("dag_run_id", "unknown")
            logger.info(f"[Webhook] Airflow DAG '{dag_id}' triggered! run_id={run_id}")
            return {"success": True, "run_id": run_id, "error": None}
        else:
            msg = f"HTTP {resp.status_code}: {resp.text[:200]}"
            logger.warning(f"[Webhook] {msg}")
            return {"success": False, "run_id": None, "error": msg}
    except Exception as exc:
        logger.warning(f"[Webhook] Airflow trigger failed: {exc}")
        return {"success": False, "run_id": None, "error": str(exc)}


# ---------------------------------------------------------------------------
# Prometheus push
# ---------------------------------------------------------------------------

def _push_metrics(drift_share: float, drifted_cols: int, total_cols: int):
    if not PUSHGATEWAY_URL:
        return
    try:
        from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
        reg = CollectorRegistry()
        Gauge("neuralretail_drift_share",    "Drift share", registry=reg).set(drift_share)
        Gauge("neuralretail_drifted_cols",   "Drifted columns", registry=reg).set(drifted_cols)
        push_to_gateway(PUSHGATEWAY_URL, job="drift_monitor", registry=reg)
    except Exception as exc:
        logger.warning(f"[Drift] Pushgateway push failed (non-fatal): {exc}")


# ---------------------------------------------------------------------------
# Core DriftMonitor class
# ---------------------------------------------------------------------------

class DriftMonitor:
    def __init__(
        self,
        drift_threshold: float = DRIFT_THRESHOLD,
        airflow_dag_id:  str   = AIRFLOW_DAG_ID,
    ):
        self.drift_threshold = drift_threshold
        self.airflow_dag_id  = airflow_dag_id

    def check_drift(
        self,
        reference_data: pd.DataFrame,
        current_data:   pd.DataFrame,
        run_id:         Optional[str] = None,
    ) -> dict:
        
        # Import Evdiently 0.4.x specific modules
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset

        logger.info(
            f"[Drift] Running Evidently drift analysis "
            f"(ref={len(reference_data)} rows, cur={len(current_data)} rows)..."
        )
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference_data, current_data=current_data)
        
        # Parse 0.4.x output
        report_dict = report.as_dict()
        metrics = report_dict.get("metrics", [])
        
        drift_share = 0.0
        drifted_cols = 0
        total_cols = 0
        column_drift = {}

        # The first metric is usually the dataset drift summary
        for m in metrics:
            if m.get("metric") == "DatasetDriftMetric":
                res = m.get("result", {})
                drift_share = res.get("share_of_drifted_columns", 0.0)
                drifted_cols = res.get("number_of_drifted_columns", 0)
                total_cols = res.get("number_of_columns", 0)
            
            if m.get("metric") == "DataDriftTable":
                res = m.get("result", {})
                drift_by_columns = res.get("drift_by_columns", {})
                for col, info in drift_by_columns.items():
                    column_drift[col] = {
                        "detected": info.get("drift_detected", False),
                        "score": info.get("drift_score", 0.0),
                        "stattest": info.get("stattest_name", ""),
                    }

        logger.info(
            f"[Drift] {drifted_cols}/{total_cols} columns drifted | "
            f"share={drift_share:.2%} | threshold={self.drift_threshold:.2%}"
        )

        html_path = os.path.join(REPORTS_DIR, f"drift_report_{ts}.html")
        json_path = os.path.join(REPORTS_DIR, f"drift_summary_{ts}.json")

        report.save_html(html_path)

        json_summary = {
            "timestamp":            ts,
            "drift_share":          drift_share,
            "drifted_columns":      drifted_cols,
            "total_columns":        total_cols,
            "column_drift":         column_drift,
            "threshold":            self.drift_threshold,
            "retraining_required":  drift_share > self.drift_threshold,
        }
        with open(json_path, "w") as f:
            json.dump(json_summary, f, indent=2)

        self._log_to_mlflow(drift_share, drifted_cols, total_cols)
        _push_metrics(drift_share, drifted_cols, total_cols)

        webhook_result = {"success": False, "run_id": None, "error": "not triggered"}
        retraining_triggered = False

        if drift_share > self.drift_threshold:
            logger.warning(
                f"[Drift] THRESHOLD EXCEEDED "
                f"({drift_share:.2%} > {self.drift_threshold:.2%}) "
                "-- triggering Airflow retraining webhook!"
            )
            webhook_result = _trigger_airflow_dag(
                dag_id=self.airflow_dag_id,
                conf={
                    "drift_share":     drift_share,
                    "drifted_columns": drifted_cols,
                    "triggered_by":    "drift_monitor",
                    "timestamp":       ts,
                    "report_json":     json_path,
                },
            )
            retraining_triggered = True

        return {
            "drift_share":          drift_share,
            "drifted_columns":      drifted_cols,
            "total_columns":        total_cols,
            "column_drift":         column_drift,
            "report_html_path":     html_path,
            "report_json_path":     json_path,
            "retraining_triggered": retraining_triggered,
            "webhook_result":       webhook_result,
            "threshold":            self.drift_threshold,
            "status":               "DRIFT_DETECTED" if retraining_triggered else "OK",
        }

    def _log_to_mlflow(self, drift_share: float, drifted_cols: int, total_cols: int):
        try:
            import mlflow
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment("drift_monitoring")
            with mlflow.start_run(run_name="drift_check"):
                mlflow.log_metric("drift_share",     drift_share)
                mlflow.log_metric("drifted_columns", drifted_cols)
                mlflow.log_metric("total_columns",   total_cols)
                mlflow.log_param("threshold",        self.drift_threshold)
        except Exception:
            pass


def compare_champion_challenger(
    reference_data:       pd.DataFrame,
    current_data:         pd.DataFrame,
    champion_model_name:  str = "xgb_churn",
    challenger_model_name: str = "stacked_churn",
) -> dict:
    import pickle
    from sklearn.metrics import roc_auc_score

    feature_cols = [c for c in current_data.columns if c not in ("churn", "customer_id")]
    X = current_data[feature_cols].select_dtypes(include=[np.number])
    y = current_data.get("churn")

    results: dict = {}
    proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    for model_name in [champion_model_name, challenger_model_name]:
        path = os.path.join(proj_root, "models", f"{model_name}.pkl")
        if not os.path.exists(path):
            continue
        try:
            with open(path, "rb") as f:
                model = pickle.load(f)
            proba = model.predict_proba(X)[:, 1]
            auc   = roc_auc_score(y, proba) if y is not None else 0.0
            results[model_name] = {"auc": round(float(auc), 6)}
        except Exception as exc:
            results[model_name] = {"auc": 0.0, "error": str(exc)}

    promote = False
    recommendation = "INSUFFICIENT_MODELS"
    if len(results) == 2:
        champ_auc = results[champion_model_name].get("auc", 0.0)
        chall_auc = results[challenger_model_name].get("auc", 0.0)
        promote   = chall_auc > champ_auc + 0.01
        recommendation = f"PROMOTE {challenger_model_name}" if promote else f"KEEP {champion_model_name}"
        logger.info(f"[Champion] {champion_model_name} AUC={champ_auc:.4f} | {challenger_model_name} AUC={chall_auc:.4f} -> {recommendation}")
        try:
            import mlflow
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment("champion_challenger")
            with mlflow.start_run(run_name="champion_challenger_eval"):
                mlflow.log_metric("champion_auc",    champ_auc)
                mlflow.log_metric("challenger_auc",  chall_auc)
                mlflow.log_param("recommendation",   recommendation)
        except Exception:
            pass

    return {
        "models":              results,
        "promote_challenger":  promote,
        "recommendation":      recommendation,
    }
