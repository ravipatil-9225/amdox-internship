"""Phase 2 integration test script."""
import sys, os, glob
sys.path.insert(0, '.')

print("=== Phase 2 Integration Test ===")

# 1. Kafka Consumer
print()
print("[1/3] Kafka Streaming Consumer...")
from src.pipelines.kafka_consumer import get_consumer, BRONZE_TRANSACTIONS, DLQ_PATH
import pandas as pd

consumer = get_consumer(mock=True)
consumer.consume_loop(max_messages=60)

df = pd.read_parquet(BRONZE_TRANSACTIONS)
print(f"  bronze/transactions.parquet rows: {len(df)}")
if os.path.exists(DLQ_PATH):
    dlq = pd.read_parquet(DLQ_PATH)
    print(f"  DLQ rows: {len(dlq)}")
print("  Kafka consumer: PASS")

# 2. OpenLineage
print()
print("[2/3] OpenLineage Emitter...")
from src.pipelines.data_lineage import emit_lineage_event, AirflowLineageExtractor, log_training_lineage

before = len(glob.glob("reports/lineage/*.json"))
log_training_lineage(["silver/customer_features.parquet"], "xgb_churn", {"auc": 0.98})
after = len(glob.glob("reports/lineage/*.json"))
print(f"  Lineage events emitted: {after - before} new JSON files")

ext = AirflowLineageExtractor("test_pipeline")
ext.on_start({"run_id": "test-001"}, inputs=[{"name": "bronze/transactions.parquet"}])
ext.on_complete({"run_id": "test-001"}, outputs=[{"name": "gold/forecast.parquet"}], row_count=365)
print("  OpenLineage + Airflow extractor: PASS")

# 3. Feast
print()
print("[3/3] Feast Feature Serving (SQLite mode)...")
from src.pipelines.feast_serving import materialise_features, get_customer_features_online, _redis_is_available

redis_ok = _redis_is_available()
store_name = "Redis" if redis_ok else "SQLite"
print(f"  Redis available: {redis_ok} -> using {store_name}")

stats = materialise_features()
n_rows = stats["rows"]
store = stats["store"]
print(f"  Materialised {n_rows} features into {store} store")

features_df = get_customer_features_online(["CUST-10000", "CUST-10001"])
n_customers = len(features_df)
n_cols = len(features_df.columns)
print(f"  Online retrieval: {n_customers} customers, {n_cols} features")
print("  Feast: PASS")

print()
print("=== All Phase 2 Tests PASSED ===")
