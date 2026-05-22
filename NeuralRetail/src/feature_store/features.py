"""
NeuralRetail -- Feast Feature Definitions
==========================================
Phase 2: Extended feature views including RFM computed features
backed by the silver-layer Parquet files.
"""
from datetime import timedelta
import os

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64, Int32, String, Float32

# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------

customer = Entity(
    name="customer",
    join_keys=["customer_id"],
    description="Unique retail customer",
)

# ---------------------------------------------------------------------------
# Data Sources
# ---------------------------------------------------------------------------

_BASE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data")
)

customer_source = FileSource(
    path=os.path.join(_BASE, "bronze", "customers.parquet"),
    timestamp_field="signup_date",
)

transaction_source = FileSource(
    path=os.path.join(_BASE, "bronze", "transactions.parquet"),
    timestamp_field="timestamp",
)

# Silver layer -- RFM features (computed by feature_engineering pipeline)
rfm_source = FileSource(
    path=os.path.join(_BASE, "silver", "customer_features.parquet"),
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# ---------------------------------------------------------------------------
# Feature Views
# ---------------------------------------------------------------------------

customer_demographics = FeatureView(
    name="customer_demographics",
    entities=[customer],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="age",     dtype=Int32),
        Field(name="gender",  dtype=String),
        Field(name="segment", dtype=String),
    ],
    source=customer_source,
    description="Customer demographic attributes from CRM",
)

transaction_features = FeatureView(
    name="transaction_features",
    entities=[customer],
    ttl=timedelta(days=365),
    schema=[
        Field(name="sku_id",            dtype=String),
        Field(name="quantity",          dtype=Int32),
        Field(name="discount_applied",  dtype=Float64),
        Field(name="total_amount",      dtype=Float64),
    ],
    source=transaction_source,
    description="Raw POS transaction features",
)

rfm_features = FeatureView(
    name="rfm_features",
    entities=[customer],
    ttl=timedelta(days=30),
    schema=[
        Field(name="recency",    dtype=Int32),
        Field(name="frequency",  dtype=Int32),
        Field(name="monetary",   dtype=Float64),
        Field(name="churn_risk", dtype=Float32),
        Field(name="clv_score",  dtype=Float32),
    ],
    source=rfm_source,
    description="Computed RFM behavioural features (silver layer)",
)
