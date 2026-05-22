"""
NeuralRetail -- OpenLineage Data Lineage Tracker
=================================================
Phase 2 deliverable: Real OpenLineage integration.

Transport priority:
  1. HTTP to Marquez / OpenLineage-compatible server (OPENLINEAGE_URL env)
  2. Local JSON file fallback (reports/lineage/)

Includes:
  - Full OpenLineage RunEvent v1 spec
  - Dataset facets (schema, data quality stats)
  - Job facets (source code location, processing engine)
  - Airflow extractor hook mixin for DAG integration
  - Schema hash for change detection
"""
import hashlib
import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("neuralretail.lineage")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------

OL_URL       = os.environ.get("OPENLINEAGE_URL", "")           # e.g. http://localhost:5000
OL_NAMESPACE = os.environ.get("OPENLINEAGE_NAMESPACE", "neuralretail")
OL_API_KEY   = os.environ.get("OPENLINEAGE_API_KEY", "")

LINEAGE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "reports", "lineage"
)

_HAS_REQUESTS = False
try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# OpenLineage event builder
# ---------------------------------------------------------------------------

def _build_dataset(name: str, namespace: str, schema_fields: list[dict] | None = None,
                   row_count: int | None = None) -> dict:
    """
    Build a fully-spec'd OpenLineage Dataset node with optional
    SchemaDatasetFacet and DataQualityMetricsInputDatasetFacet.
    """
    facets: dict = {}

    if schema_fields:
        facets["schema"] = {
            "_producer": "neuralretail-lineage/2.0",
            "_schemaURL": "https://openlineage.io/spec/facets/1-1-1/SchemaDatasetFacet.json",
            "fields": schema_fields,
        }

    if row_count is not None:
        facets["dataQualityMetrics"] = {
            "_producer": "neuralretail-lineage/2.0",
            "_schemaURL": "https://openlineage.io/spec/facets/1-0-2/DataQualityMetricsInputDatasetFacet.json",
            "rowCount": row_count,
        }

    return {
        "namespace": namespace,
        "name": name,
        "facets": facets,
    }


def _build_job_facets(processing_engine: str = "python",
                      source_location: str | None = None) -> dict:
    """Build job-level OpenLineage facets."""
    facets: dict = {
        "processing_engine": {
            "_producer": "neuralretail-lineage/2.0",
            "_schemaURL": "https://openlineage.io/spec/facets/1-2-2/ProcessingEngineRunFacet.json",
            "version": "3.11",
            "name": processing_engine,
        }
    }
    if source_location:
        facets["sourceCodeLocation"] = {
            "_producer": "neuralretail-lineage/2.0",
            "_schemaURL": "https://openlineage.io/spec/facets/1-0-1/SourceCodeLocationJobFacet.json",
            "type": "git",
            "url": source_location,
        }
    return facets


def _schema_hash(fields: list[dict] | None) -> str:
    """Return a short SHA-256 hash of the schema for change detection."""
    if not fields:
        return ""
    raw = json.dumps(fields, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()[:12]


# ---------------------------------------------------------------------------
# HTTP emitter
# ---------------------------------------------------------------------------

def _emit_http(event: dict) -> bool:
    """
    POST a single OpenLineage RunEvent to the configured server.
    Returns True on success.
    """
    if not (_HAS_REQUESTS and OL_URL):
        return False

    url = f"{OL_URL.rstrip('/')}/api/v1/lineage"
    headers = {"Content-Type": "application/json"}
    if OL_API_KEY:
        headers["Authorization"] = f"Bearer {OL_API_KEY}"

    try:
        resp = _requests.post(url, json=event, headers=headers, timeout=5)
        if resp.status_code in (200, 201, 202):
            logger.debug(f"[Lineage] HTTP OK -> {url}")
            return True
        else:
            logger.warning(
                f"[Lineage] HTTP {resp.status_code} from {url}: {resp.text[:200]}"
            )
            return False
    except Exception as exc:
        logger.warning(f"[Lineage] HTTP emit failed ({exc}); writing to local file.")
        return False


# ---------------------------------------------------------------------------
# File emitter (fallback)
# ---------------------------------------------------------------------------

def _emit_file(event: dict, job_name: str):
    """Write lineage event to a local JSON file."""
    os.makedirs(LINEAGE_DIR, exist_ok=True)
    run_id_short = event["run"]["runId"][:8]
    filename = f"{job_name}_{run_id_short}.json"
    filepath = os.path.join(LINEAGE_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(event, f, indent=2)
    logger.info(f"[Lineage] Event written -> {filepath}")


# ---------------------------------------------------------------------------
# Primary emitter
# ---------------------------------------------------------------------------

def emit_lineage_event(
    job_name: str,
    job_namespace: str = OL_NAMESPACE,
    inputs: list[dict] | None = None,
    outputs: list[dict] | None = None,
    run_facets: dict | None = None,
    input_schema: list[dict] | None = None,
    output_schema: list[dict] | None = None,
    input_row_count: int | None = None,
    output_row_count: int | None = None,
    processing_engine: str = "python",
    event_type: str = "COMPLETE",
):
    """
    Emit a full OpenLineage RunEvent.

    Parameters
    ----------
    job_name : str
        Logical pipeline step name (e.g. 'kafka_ingestion').
    job_namespace : str
        OpenLineage namespace (default: OPENLINEAGE_NAMESPACE env var).
    inputs / outputs : list[dict]
        Each must have a ``name`` key; optionally ``namespace``.
    run_facets : dict
        Arbitrary metadata attached to the run (row counts, versions, etc.).
    input_schema / output_schema : list[dict]
        Schema fields [{name, type}] for the SchemaDatasetFacet.
    input_row_count / output_row_count : int
        Row counts for the DataQualityMetrics facet.
    processing_engine : str
        e.g. 'python', 'spark', 'airflow'.
    event_type : str
        One of START | RUNNING | COMPLETE | FAIL | ABORT.
    """
    run_id = str(uuid.uuid4())

    # Build input / output dataset nodes
    input_nodes = []
    for d in (inputs or []):
        ns = d.get("namespace", job_namespace)
        input_nodes.append(
            _build_dataset(d["name"], ns, input_schema, input_row_count)
        )

    output_nodes = []
    for d in (outputs or []):
        ns = d.get("namespace", job_namespace)
        output_nodes.append(
            _build_dataset(d["name"], ns, output_schema, output_row_count)
        )

    # Add schema hash to run facets for change detection
    schema_h = _schema_hash(output_schema or input_schema)
    extra_facets = dict(run_facets or {})
    if schema_h:
        extra_facets["schema_hash"] = schema_h

    event = {
        "_producer": "neuralretail-lineage/2.0",
        "_schemaURL": "https://openlineage.io/spec/2-0-2/OpenLineage.json",
        "eventType": event_type,
        "eventTime": datetime.now(timezone.utc).isoformat(),
        "run": {
            "runId": run_id,
            "facets": extra_facets,
        },
        "job": {
            "namespace": job_namespace,
            "name": job_name,
            "facets": _build_job_facets(processing_engine),
        },
        "inputs":  input_nodes,
        "outputs": output_nodes,
    }

    # Try HTTP first; fall back to file
    if not _emit_http(event):
        _emit_file(event, job_name)

    return event


# ---------------------------------------------------------------------------
# Airflow extractor mixin
# ---------------------------------------------------------------------------

class AirflowLineageExtractor:
    """
    Mixin / utility for Airflow PythonOperators to emit lineage events
    before and after task execution.

    Usage in a DAG:
        extractor = AirflowLineageExtractor("my_pipeline")

        def my_task(**context):
            extractor.on_start(context, inputs=[...])
            # ... do work ...
            extractor.on_complete(context, outputs=[...], row_count=n)

        PythonOperator(task_id='my_task', python_callable=my_task)
    """

    def __init__(self, pipeline_name: str, namespace: str = OL_NAMESPACE):
        self.pipeline_name = pipeline_name
        self.namespace = namespace
        self._run_id: Optional[str] = None

    def on_start(self, context: dict,
                 inputs: list[dict] | None = None,
                 extra_facets: dict | None = None):
        """Call at the beginning of an Airflow task."""
        task_id = context.get("task_instance", {}).task_id if hasattr(
            context.get("task_instance", {}), "task_id") else "unknown"
        dag_run_id = context.get("run_id", str(uuid.uuid4()))
        self._run_id = dag_run_id

        emit_lineage_event(
            job_name=f"{self.pipeline_name}.{task_id}",
            job_namespace=self.namespace,
            inputs=inputs,
            run_facets={
                "dag_run_id": dag_run_id,
                "airflow_task": task_id,
                **(extra_facets or {}),
            },
            processing_engine="airflow",
            event_type="START",
        )

    def on_complete(self, context: dict,
                    outputs: list[dict] | None = None,
                    row_count: int | None = None,
                    extra_facets: dict | None = None):
        """Call at the successful end of an Airflow task."""
        task_id = context.get("task_instance", {}).task_id if hasattr(
            context.get("task_instance", {}), "task_id") else "unknown"
        dag_run_id = context.get("run_id", self._run_id or str(uuid.uuid4()))

        emit_lineage_event(
            job_name=f"{self.pipeline_name}.{task_id}",
            job_namespace=self.namespace,
            outputs=outputs,
            output_row_count=row_count,
            run_facets={
                "dag_run_id": dag_run_id,
                "airflow_task": task_id,
                **(extra_facets or {}),
            },
            processing_engine="airflow",
            event_type="COMPLETE",
        )

    def on_fail(self, context: dict, error: str = ""):
        """Call when an Airflow task fails."""
        task_id = context.get("task_instance", {}).task_id if hasattr(
            context.get("task_instance", {}), "task_id") else "unknown"

        emit_lineage_event(
            job_name=f"{self.pipeline_name}.{task_id}",
            job_namespace=self.namespace,
            run_facets={"error": error},
            processing_engine="airflow",
            event_type="FAIL",
        )


# ---------------------------------------------------------------------------
# Convenience wrappers (backwards-compatible with existing callers)
# ---------------------------------------------------------------------------

def log_ingestion_lineage(source_files: list[str], output_tables: list[str],
                          row_count: int | None = None):
    """Log lineage for the ingestion (raw -> bronze) step."""
    emit_lineage_event(
        job_name="data_ingestion",
        inputs=[{"name": f} for f in source_files],
        outputs=[{"name": t} for t in output_tables],
        output_row_count=row_count,
        run_facets={"step": "ingestion", "layer": "bronze"},
    )


def log_feature_engineering_lineage(
    bronze_tables: list[str],
    silver_tables: list[str],
    gold_tables: list[str],
    row_counts: dict | None = None,
):
    """Log lineage for bronze -> silver/gold feature engineering."""
    emit_lineage_event(
        job_name="feature_engineering",
        inputs=[{"name": t} for t in bronze_tables],
        outputs=[{"name": t} for t in silver_tables + gold_tables],
        run_facets={
            "step": "feature_engineering",
            "layers": ["silver", "gold"],
            "row_counts": row_counts or {},
        },
    )


def log_training_lineage(
    feature_tables: list[str],
    model_name: str,
    metrics: dict | None = None,
):
    """Log lineage for model training (silver -> model artifact)."""
    emit_lineage_event(
        job_name=f"train_{model_name}",
        inputs=[{"name": t} for t in feature_tables],
        outputs=[{"name": f"models/{model_name}"}],
        run_facets={
            "step": "training",
            "model": model_name,
            "metrics": metrics or {},
        },
    )


def log_feast_materialise_lineage(
    bronze_tables: list[str],
    online_store: str,
    row_count: int,
):
    """Log lineage for a Feast materialisation job."""
    emit_lineage_event(
        job_name="feast_materialise",
        inputs=[{"name": t} for t in bronze_tables],
        outputs=[{"name": "feast_online_store", "namespace": online_store}],
        output_row_count=row_count,
        run_facets={"online_store": online_store, "step": "materialise"},
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    transport = f"HTTP ({OL_URL})" if OL_URL else "local file (reports/lineage/)"
    print(f"\n[OpenLineage] Transport: {transport}")
    print("Running lineage self-test...")

    # Ingestion
    log_ingestion_lineage(
        source_files=["raw/pos_transactions.csv", "raw/customers.csv"],
        output_tables=["bronze/transactions.parquet", "bronze/customers.parquet"],
        row_count=50000,
    )

    # Feature engineering
    log_feature_engineering_lineage(
        bronze_tables=["bronze/customers.parquet", "bronze/transactions.parquet"],
        silver_tables=["silver/customer_features.parquet", "silver/daily_sku_features.parquet"],
        gold_tables=["gold/product_catalog.parquet"],
        row_counts={"customers": 5000, "transactions": 50000},
    )

    # Training
    log_training_lineage(
        feature_tables=["silver/customer_features.parquet"],
        model_name="xgb_churn",
        metrics={"auc": 0.95, "f1": 0.91},
    )

    # Feast materialise
    log_feast_materialise_lineage(
        bronze_tables=["bronze/customers.parquet", "bronze/transactions.parquet"],
        online_store="sqlite",
        row_count=5000,
    )

    # Airflow extractor simulation
    print("\n[OpenLineage] Simulating Airflow extractor...")
    extractor = AirflowLineageExtractor("demand_forecast_pipeline")
    fake_ctx = {"run_id": "manual__2026-05-20T00:00:00"}
    extractor.on_start(fake_ctx, inputs=[{"name": "bronze/transactions.parquet"}])
    extractor.on_complete(fake_ctx, outputs=[{"name": "silver/daily_sku_features.parquet"}], row_count=365)

    print("\nLineage self-test complete! [OK]")
    print(f"Check: reports/lineage/")
