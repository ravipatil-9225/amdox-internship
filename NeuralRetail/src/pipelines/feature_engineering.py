"""
NeuralRetail – Advanced Feature Engineering Pipeline
=====================================================
Generates production-grade ML features from bronze-layer data:
  • RFM (Recency / Frequency / Monetary)
  • Rolling averages & Lag features
  • Seasonal / holiday / promotional indicators
  • Recency-decay weights
  • Category-switching flags

Outputs a silver-layer Parquet for downstream model training.
"""
import os
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
BRONZE = os.path.join(DATA_DIR, "bronze")
SILVER = os.path.join(DATA_DIR, "silver")
GOLD = os.path.join(DATA_DIR, "gold")


def load_bronze():
    """Load all bronze-layer tables."""
    customers = pd.read_parquet(os.path.join(BRONZE, "customers.parquet"))
    transactions = pd.read_parquet(os.path.join(BRONZE, "transactions.parquet"))
    products = pd.read_parquet(os.path.join(BRONZE, "products.parquet"))
    inventory = pd.read_parquet(os.path.join(BRONZE, "inventory.parquet"))
    print(f"  Loaded bronze: {len(customers)} customers, "
          f"{len(transactions)} transactions, "
          f"{len(products)} products, "
          f"{len(inventory)} inventory rows")
    return customers, transactions, products, inventory


# ─────────────────────────────────────────────────────────────────────────────
# RFM Features
# ─────────────────────────────────────────────────────────────────────────────
def build_rfm(transactions: pd.DataFrame, reference_date=None) -> pd.DataFrame:
    """Classic RFM + recency-decay weight."""
    if reference_date is None:
        reference_date = transactions["timestamp"].max()

    rfm = transactions.groupby("customer_id").agg(
        recency=("timestamp", lambda x: (reference_date - x.max()).days),
        frequency=("transaction_id", "nunique"),
        monetary=("total_amount", "sum"),
        avg_basket=("total_amount", "mean"),
        max_basket=("total_amount", "max"),
        total_qty=("quantity", "sum"),
    ).reset_index()

    # Recency-decay: exponential weight (higher = more recently active)
    rfm["recency_decay"] = np.exp(-0.01 * rfm["recency"])
    return rfm


# ─────────────────────────────────────────────────────────────────────────────
# Time-Series Features (per-SKU daily aggregation)
# ─────────────────────────────────────────────────────────────────────────────
def build_timeseries_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Rolling averages, lag features, and seasonal indicators per SKU."""
    df = transactions.copy()
    df["date"] = df["timestamp"].dt.date
    daily = df.groupby(["sku_id", "date"]).agg(
        daily_qty=("quantity", "sum"),
        daily_revenue=("total_amount", "sum"),
        daily_txn_count=("transaction_id", "nunique"),
    ).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values(["sku_id", "date"])

    # Rolling averages
    for window in [7, 14, 28]:
        daily[f"qty_roll_{window}d"] = (
            daily.groupby("sku_id")["daily_qty"]
            .transform(lambda s: s.rolling(window, min_periods=1).mean())
        )
        daily[f"rev_roll_{window}d"] = (
            daily.groupby("sku_id")["daily_revenue"]
            .transform(lambda s: s.rolling(window, min_periods=1).mean())
        )

    # Lag features
    for lag in [1, 7, 14]:
        daily[f"qty_lag_{lag}d"] = daily.groupby("sku_id")["daily_qty"].shift(lag)

    # Seasonal indicators
    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["month"] = daily["date"].dt.month
    daily["is_weekend"] = daily["day_of_week"].isin([5, 6]).astype(int)
    daily["is_month_end"] = daily["date"].dt.is_month_end.astype(int)
    daily["quarter"] = daily["date"].dt.quarter

    # Simple holiday proxy (first week of Jan, last week of Dec)
    daily["is_holiday_proxy"] = (
        ((daily["month"] == 1) & (daily["date"].dt.day <= 7)) |
        ((daily["month"] == 12) & (daily["date"].dt.day >= 25))
    ).astype(int)

    daily = daily.fillna(0)
    return daily


# ─────────────────────────────────────────────────────────────────────────────
# Category-Switching Features
# ─────────────────────────────────────────────────────────────────────────────
def build_category_features(transactions: pd.DataFrame,
                            products: pd.DataFrame) -> pd.DataFrame:
    """Count distinct categories purchased and detect switching."""
    merged = transactions.merge(products[["sku_id", "category"]], on="sku_id", how="left")

    cat_feats = merged.groupby("customer_id").agg(
        distinct_categories=("category", "nunique"),
        favourite_category=("category", lambda x: x.mode().iloc[0] if len(x) > 0 else "Unknown"),
        category_entropy=("category", lambda x: _entropy(x)),
    ).reset_index()

    return cat_feats


def _entropy(series: pd.Series) -> float:
    """Shannon entropy of value counts — measures category diversity."""
    probs = series.value_counts(normalize=True)
    return float(-(probs * np.log2(probs + 1e-9)).sum())


# ─────────────────────────────────────────────────────────────────────────────
# Promotional / Discount Features
# ─────────────────────────────────────────────────────────────────────────────
def build_promo_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Discount usage stats per customer."""
    promo = transactions.groupby("customer_id").agg(
        avg_discount=("discount_applied", "mean"),
        max_discount=("discount_applied", "max"),
        discount_txn_ratio=("discount_applied", lambda x: (x > 0).mean()),
    ).reset_index()
    return promo


# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------
def run_feature_engineering():
    """Full pipeline: bronze -> silver -> gold."""
    print("\n===================================================")
    print("  NeuralRetail Feature Engineering Pipeline")
    print("===================================================\n")

    os.makedirs(SILVER, exist_ok=True)
    os.makedirs(GOLD, exist_ok=True)

    customers, transactions, products, inventory = load_bronze()

    # -- Silver: Customer feature table ------------------------------------
    print("\n[1/4] Building RFM features ...")
    rfm = build_rfm(transactions)
    print(f"      -> {rfm.shape[0]} rows, {rfm.shape[1]} columns")

    print("[2/4] Building category-switching features ...")
    cat_feats = build_category_features(transactions, products)
    print(f"      -> {cat_feats.shape[0]} rows, {cat_feats.shape[1]} columns")

    print("[3/4] Building promotional features ...")
    promo_feats = build_promo_features(transactions)
    print(f"      -> {promo_feats.shape[0]} rows, {promo_feats.shape[1]} columns")

    # Merge into a single customer-level feature table
    customer_features = (
        customers
        .merge(rfm, on="customer_id", how="left")
        .merge(cat_feats, on="customer_id", how="left")
        .merge(promo_feats, on="customer_id", how="left")
        .fillna(0)
    )

    # Churn label: no purchase in last 90 days
    customer_features["churn"] = (customer_features["recency"] > 90).astype(int)

    silver_cust_path = os.path.join(SILVER, "customer_features.parquet")
    customer_features.to_parquet(silver_cust_path, index=False)
    print(f"  + Saved silver/customer_features.parquet ({customer_features.shape})")

    # -- Silver: Daily SKU time-series features ----------------------------
    print("\n[4/4] Building time-series features (rolling, lag, seasonal) ...")
    ts_features = build_timeseries_features(transactions)
    silver_ts_path = os.path.join(SILVER, "daily_sku_features.parquet")
    ts_features.to_parquet(silver_ts_path, index=False)
    print(f"  + Saved silver/daily_sku_features.parquet ({ts_features.shape})")

    # -- Gold: Enriched product catalog ------------------------------------
    gold_product = products.merge(inventory, on="sku_id", how="left")
    gold_product["margin_pct"] = (
        (gold_product["base_price"] - gold_product["cost_price"]) / gold_product["base_price"] * 100
    ).round(2)
    gold_product["days_of_stock"] = (
        gold_product["current_stock"] /
        (transactions.groupby("sku_id")["quantity"].sum() / 
         (transactions["timestamp"].max() - transactions["timestamp"].min()).days)
    ).round(1)

    gold_product_path = os.path.join(GOLD, "product_catalog.parquet")
    gold_product.to_parquet(gold_product_path, index=False)
    print(f"  + Saved gold/product_catalog.parquet ({gold_product.shape})")

    print("\n===================================================")
    print("  Feature Engineering Pipeline Complete!")
    print(f"  Customer features : {customer_features.shape[1]} cols")
    print(f"  TS features       : {ts_features.shape[1]} cols")
    print(f"  Product catalog   : {gold_product.shape[1]} cols")
    print("===================================================\n")

    return customer_features, ts_features, gold_product


if __name__ == "__main__":
    run_feature_engineering()
