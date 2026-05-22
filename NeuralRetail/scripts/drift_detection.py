"""
NeuralRetail -- Drift Detection Script
=======================================
Phase 3: Uses the upgraded DriftMonitor that:
  - Emits an HTTP webhook to Airflow when drift > threshold
  - Logs metrics to MLflow
  - Persists both HTML report and JSON summary
"""
import os
import sys
import numpy as np
import pandas as pd
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.monitoring.evidently_drift import DriftMonitor, compare_champion_challenger


def generate_drift_report(drift_threshold: float = 0.20, compare_models: bool = True):
    print("Loading training (reference) data...")
    customers    = pd.read_parquet("data/bronze/customers.parquet")
    transactions = pd.read_parquet("data/bronze/transactions.parquet")

    latest_date = transactions["timestamp"].max()
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

    features = ["age", "recency", "frequency", "monetary", "churn"]
    reference_data = data[features].sample(frac=0.5, random_state=42)

    # Simulate production data with realistic drift
    print("Simulating production data with realistic drift...")
    np.random.seed(99)
    current_data = data[features].copy()
    current_data["age"]      = current_data["age"]      + np.random.normal(2, 1, len(current_data))
    current_data["recency"]  = current_data["recency"]  * np.random.uniform(0.85, 1.15, len(current_data))
    current_data["monetary"] = current_data["monetary"] * np.random.uniform(0.9, 1.2, len(current_data))

    # Run drift analysis with webhook
    monitor = DriftMonitor(drift_threshold=drift_threshold)
    result  = monitor.check_drift(reference_data, current_data)

    # Print summary
    print("\n--- Drift Detection Summary ---")
    print(f"  Columns analyzed:   {result['total_columns']}")
    print(f"  Drifted columns:    {result['drifted_columns']}")
    print(f"  Drift share:        {result['drift_share']:.2%}")
    print(f"  Status:             {result['status']}")
    print(f"  Retraining signal:  {result['retraining_triggered']}")
    if result.get("retraining_triggered"):
        wh = result["webhook_result"]
        print(f"  Airflow webhook:    {'SUCCESS' if wh['success'] else 'ATTEMPTED'}")
        if wh.get("error"):
            print(f"  Webhook note:       {wh['error']}")
    print()

    # Per-column drift
    print("--- Column Drift Breakdown ---")
    for col, info in result["column_drift"].items():
        flag = "[DRIFT]" if info["detected"] else "[ OK  ]"
        print(f"  {flag} {col:15s} | score={info['score']:.4f} | test={info['stattest']}")

    print(f"\n  HTML report: {result['report_html_path']}")
    print(f"  JSON report: {result['report_json_path']}")

    # Optional champion-challenger evaluation
    if compare_models:
        print("\n--- Champion-Challenger Comparison ---")
        cc_result = compare_champion_challenger(reference_data, current_data)
        print(f"  Recommendation: {cc_result['recommendation']}")
        for model, metrics in cc_result.get("models", {}).items():
            print(f"  {model}: AUC={metrics.get('auc', 'N/A')}")

    print("\nDrift detection complete! [OK]")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuralRetail Drift Detection")
    parser.add_argument("--threshold", type=float, default=0.20,
                        help="Drift share threshold for retraining trigger")
    parser.add_argument("--no-compare", action="store_true", default=False,
                        help="Skip champion-challenger model comparison")
    args = parser.parse_args()

    generate_drift_report(
        drift_threshold=args.threshold,
        compare_models=not args.no_compare,
    )
