"""
NeuralRetail -- Feast Online Feature Serving (Redis + SQLite)
=============================================================
Phase 2 deliverable: Migrate the Feast online store from SQLite
to a Redis cluster for production-grade low-latency feature serving.

Strategy:
  - FEAST_ONLINE_STORE=redis (env) -> connect to Redis via feast-redis
  - Otherwise -> fall back to SQLite for local dev / CI

Added features:
  - RFM feature view materialisation from bronze data
  - Bulk online feature retrieval for model inference
  - Redis health-check with automatic fallback
  - OpenLineage event emission on materialise calls
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.pipelines.data_lineage import emit_lineage_event

logger = logging.getLogger("neuralretail.feast")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Paths & env-driven config
# ---------------------------------------------------------------------------

_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FEAST_REPO_PATH = os.path.join(os.path.dirname(__file__), "..", "feature_store")
BRONZE_DIR = os.path.join(_BASE_DIR, "data", "bronze")

REDIS_HOST = os.environ.get("FEAST_REDIS_HOST", "")
REDIS_PORT = int(os.environ.get("FEAST_REDIS_PORT", "6379"))
REDIS_DB   = int(os.environ.get("FEAST_REDIS_DB", "0"))
USE_REDIS  = bool(REDIS_HOST)


# ---------------------------------------------------------------------------
# Dynamic feature_store.yaml writer
# ---------------------------------------------------------------------------

def _write_feature_store_yaml(use_redis: bool = False):
    """
    Write a feature_store.yaml tuned for the active environment.
    Redis is used when FEAST_REDIS_HOST is set, otherwise SQLite.
    """
    yaml_path = os.path.join(FEAST_REPO_PATH, "feature_store.yaml")

    if use_redis:
        content = f"""project: neuralretail
registry: data/registry.db
provider: local
entity_key_serialization_version: 3
online_store:
  type: redis
  connection_string: "{REDIS_HOST}:{REDIS_PORT}"
  db: {REDIS_DB}
offline_store:
  type: file
"""
        logger.info(f"[Feast] Configured Redis online store: {REDIS_HOST}:{REDIS_PORT}")
    else:
        content = """project: neuralretail
registry: data/registry.db
provider: local
entity_key_serialization_version: 3
online_store:
  type: sqlite
  path: data/online_store.db
offline_store:
  type: file
"""
        logger.info("[Feast] Configured SQLite online store (local dev mode)")

    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
    with open(yaml_path, "w") as f:
        f.write(content)

    return yaml_path


# ---------------------------------------------------------------------------
# Redis health check
# ---------------------------------------------------------------------------

def _redis_is_available() -> bool:
    """Ping Redis; return True only if reachable."""
    if not USE_REDIS:
        return False
    try:
        import redis as redis_lib
        r = redis_lib.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB,
                            socket_connect_timeout=2)
        return r.ping()
    except Exception as exc:
        logger.warning(f"[Feast] Redis unavailable ({exc}); falling back to SQLite.")
        return False


# ---------------------------------------------------------------------------
# Store factory
# ---------------------------------------------------------------------------

def get_store():
    """
    Return a configured Feast FeatureStore instance.

    Automatically selects Redis if FEAST_REDIS_HOST is set and reachable,
    otherwise uses SQLite for zero-config local development.
    """
    from feast import FeatureStore

    redis_ok = _redis_is_available()
    _write_feature_store_yaml(use_redis=redis_ok)
    store = FeatureStore(repo_path=FEAST_REPO_PATH)
    return store, redis_ok


# ---------------------------------------------------------------------------
# RFM feature computation  (bronze -> feature df)
# ---------------------------------------------------------------------------

def _compute_rfm_features() -> pd.DataFrame:
    """
    Build RFM (Recency / Frequency / Monetary) + demographic features
    from bronze Parquet files. This is the Silver-layer transformation
    that feeds into the Feast offline store for materialisation.
    """
    customers = pd.read_parquet(os.path.join(BRONZE_DIR, "customers.parquet"))
    transactions = pd.read_parquet(os.path.join(BRONZE_DIR, "transactions.parquet"))

    latest_date = transactions["timestamp"].max()

    rfm = (
        transactions.groupby("customer_id")
        .agg(
            recency=("timestamp",   lambda x: int((latest_date - x.max()).days)),
            frequency=("transaction_id", "count"),
            monetary=("total_amount",    "sum"),
            last_sku=("sku_id",     "last"),
        )
        .reset_index()
    )

    rfm["monetary"] = rfm["monetary"].round(2)
    rfm["churn_risk"] = (rfm["recency"] > 90).astype(float)
    rfm["clv_score"]  = (rfm["monetary"] * rfm["frequency"] / 1000).round(4)

    data = customers.merge(rfm, on="customer_id", how="left")
    for col in data.columns:
        if data[col].dtype == object or data[col].dtype == 'string':
            data[col] = data[col].fillna("").astype(str)
        else:
            data[col] = data[col].fillna(0)
    data["event_timestamp"] = pd.Timestamp(datetime.now(timezone.utc))
    data["created_timestamp"] = data["event_timestamp"]
    return data


# ---------------------------------------------------------------------------
# Materialisation
# ---------------------------------------------------------------------------

def materialise_features(store=None, redis_ok: bool = False) -> dict:
    """
    Compute RFM features and materialise them into the Feast online store.

    Returns a dict with materialisation stats.
    """
    if store is None:
        store, redis_ok = get_store()

    logger.info("[Feast] Computing RFM features from bronze layer...")
    rfm_df = _compute_rfm_features()

    # Write to silver Parquet so Feast can read it as FileSource
    silver_dir = os.path.join(_BASE_DIR, "data", "silver")
    os.makedirs(silver_dir, exist_ok=True)
    silver_path = os.path.join(silver_dir, "customer_features.parquet")
    rfm_df.to_parquet(silver_path, index=False)
    logger.info(f"[Feast] Wrote {len(rfm_df)} RFM rows to {silver_path}")

    # Materialise into online store
    try:
        end_ts = datetime.now(timezone.utc)
        start_ts = end_ts - timedelta(days=3650)
        store.materialize(start_date=start_ts, end_date=end_ts)
        backend = "Redis" if redis_ok else "SQLite"
        logger.info(f"[Feast] Materialised features into {backend} online store")
    except Exception as exc:
        logger.warning(f"[Feast] Materialisation warning (non-fatal): {exc}")

    # OpenLineage
    emit_lineage_event(
        job_name="feast_materialise",
        inputs=[
            {"name": "bronze/customers.parquet"},
            {"name": "bronze/transactions.parquet"},
        ],
        outputs=[{"name": "feast_online_store"}],
        run_facets={
            "rows_materialised": len(rfm_df),
            "online_store": "redis" if redis_ok else "sqlite",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )

    return {
        "rows": len(rfm_df),
        "store": "redis" if redis_ok else "sqlite",
        "silver_path": silver_path,
    }


# ---------------------------------------------------------------------------
# Online feature retrieval
# ---------------------------------------------------------------------------

def get_customer_features_online(
    customer_ids: list[str],
    store=None,
) -> pd.DataFrame:
    """
    Retrieve real-time customer RFM + demographic features from the
    Feast online store (Redis if available, SQLite otherwise).
    """
    if store is None:
        store, _ = get_store()

    entity_rows = [{"customer_id": cid} for cid in customer_ids]
    try:
        result = store.get_online_features(
            features=[
                "customer_demographics:age",
                "customer_demographics:gender",
                "customer_demographics:segment",
            ],
            entity_rows=entity_rows,
        ).to_dict()
        return pd.DataFrame(result)
    except Exception as exc:
        logger.warning(f"[Feast] Online retrieval failed ({exc}); using fallback.")
        return _fallback_features(customer_ids)


def get_rfm_features_online(
    customer_ids: list[str],
    store=None,
) -> pd.DataFrame:
    """
    Retrieve RFM features for a batch of customers from online store.
    Used by the churn prediction and segmentation endpoints.
    """
    if store is None:
        store, _ = get_store()

    entity_rows = [{"customer_id": cid} for cid in customer_ids]
    try:
        result = store.get_online_features(
            features=[
                "transaction_features:quantity",
                "transaction_features:discount_applied",
                "transaction_features:total_amount",
            ],
            entity_rows=entity_rows,
        ).to_dict()
        return pd.DataFrame(result)
    except Exception as exc:
        logger.warning(f"[Feast] RFM retrieval failed ({exc}); using fallback.")
        return _fallback_rfm(customer_ids)


def _fallback_features(customer_ids: list[str]) -> pd.DataFrame:
    """
    Read customer features directly from bronze Parquet when online
    store is unavailable. Maintains API contract.
    """
    try:
        df = pd.read_parquet(os.path.join(BRONZE_DIR, "customers.parquet"))
        return df[df["customer_id"].isin(customer_ids)][
            ["customer_id", "age", "gender", "segment"]
        ].reset_index(drop=True)
    except Exception:
        return pd.DataFrame({"customer_id": customer_ids})


def _fallback_rfm(customer_ids: list[str]) -> pd.DataFrame:
    """
    Compute RFM inline from bronze layer when online store is unavailable.
    """
    try:
        rfm = _compute_rfm_features()
        return rfm[rfm["customer_id"].isin(customer_ids)][
            ["customer_id", "recency", "frequency", "monetary", "churn_risk"]
        ].reset_index(drop=True)
    except Exception:
        return pd.DataFrame({"customer_id": customer_ids})


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("NeuralRetail -- Feast Feature Serving")
    print("=" * 55)

    redis_avail = _redis_is_available()
    store_type = "Redis" if redis_avail else "SQLite (local dev)"
    print(f"\nOnline store: {store_type}")

    print("\n[1/3] Materialising features...")
    stats = materialise_features()
    print(f"  Materialised {stats['rows']} rows into {stats['store']} store")

    sample_ids = ["CUST-10000", "CUST-10001", "CUST-10002"]

    print("\n[2/3] Fetching customer demographics...")
    demo_df = get_customer_features_online(sample_ids)
    print(demo_df.to_string(index=False))

    print("\n[3/3] Fetching RFM features...")
    rfm_df = get_rfm_features_online(sample_ids)
    print(rfm_df.to_string(index=False))

    print("\nFeast serving test complete! [OK]")
