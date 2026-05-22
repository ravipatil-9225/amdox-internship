"""
Great Expectations Data Quality Gate
Validates the Bronze layer datasets against defined expectations
to ensure data integrity before ML pipelines consume them.
Compatible with Great Expectations 0.18.x
"""
import pandas as pd
import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import great_expectations as gx
from great_expectations.core.expectation_suite import ExpectationSuite
from great_expectations.core import ExpectationConfiguration


def validate_dataset(context, df, suite_name, expectations):
    """Validate a DataFrame against a set of expectations."""
    # Create expectation suite
    suite = context.add_or_update_expectation_suite(expectation_suite_name=suite_name)
    for exp in expectations:
        suite.add_expectation(ExpectationConfiguration(
            expectation_type=exp["type"],
            kwargs=exp["kwargs"]
        ))
    context.add_or_update_expectation_suite(expectation_suite=suite)

    # Create validator
    validator = context.sources.pandas_default.read_dataframe(df)
    result = validator.validate(expectation_suite=suite)

    passed = result.success
    stats = result.statistics
    total = stats.get('evaluated_expectations', 0)
    success = stats.get('successful_expectations', 0)

    return {'passed': passed, 'total': total, 'success': success}


def validate_bronze_data():
    print("=" * 60)
    print("  NeuralRetail - Data Quality Gate (Great Expectations)")
    print("=" * 60)

    os.makedirs("reports/data_quality", exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    all_results = {}

    context = gx.get_context()

    # ── 1. Validate Customers ──
    print("\n[1/4] Validating customers.parquet...")
    customers = pd.read_parquet("data/bronze/customers.parquet")
    result = validate_dataset(context, customers, "customers_suite", [
        {"type": "expect_column_to_exist", "kwargs": {"column": "customer_id"}},
        {"type": "expect_column_to_exist", "kwargs": {"column": "age"}},
        {"type": "expect_column_to_exist", "kwargs": {"column": "region"}},
        {"type": "expect_column_values_to_be_between", "kwargs": {"column": "age", "min_value": 18, "max_value": 100}},
        {"type": "expect_column_values_to_not_be_null", "kwargs": {"column": "customer_id"}},
        {"type": "expect_column_values_to_be_unique", "kwargs": {"column": "customer_id"}},
    ])
    print(f"  Result: {'PASSED' if result['passed'] else 'FAILED'} ({result['success']}/{result['total']} expectations)")
    all_results['customers'] = result

    # ── 2. Validate Products ──
    print("\n[2/4] Validating products.parquet...")
    products = pd.read_parquet("data/bronze/products.parquet")
    result = validate_dataset(context, products, "products_suite", [
        {"type": "expect_column_to_exist", "kwargs": {"column": "sku_id"}},
        {"type": "expect_column_to_exist", "kwargs": {"column": "category"}},
        {"type": "expect_column_to_exist", "kwargs": {"column": "cost_price"}},
        {"type": "expect_column_to_exist", "kwargs": {"column": "base_price"}},
        {"type": "expect_column_values_to_not_be_null", "kwargs": {"column": "sku_id"}},
        {"type": "expect_column_values_to_be_unique", "kwargs": {"column": "sku_id"}},
    ])
    print(f"  Result: {'PASSED' if result['passed'] else 'FAILED'} ({result['success']}/{result['total']} expectations)")
    all_results['products'] = result

    # ── 3. Validate Transactions ──
    print("\n[3/4] Validating transactions.parquet...")
    transactions = pd.read_parquet("data/bronze/transactions.parquet")
    result = validate_dataset(context, transactions, "transactions_suite", [
        {"type": "expect_column_to_exist", "kwargs": {"column": "transaction_id"}},
        {"type": "expect_column_to_exist", "kwargs": {"column": "customer_id"}},
        {"type": "expect_column_to_exist", "kwargs": {"column": "sku_id"}},
        {"type": "expect_column_to_exist", "kwargs": {"column": "quantity"}},
        {"type": "expect_column_to_exist", "kwargs": {"column": "total_amount"}},
        {"type": "expect_column_values_to_not_be_null", "kwargs": {"column": "transaction_id"}},
        {"type": "expect_column_values_to_be_unique", "kwargs": {"column": "transaction_id"}},
        {"type": "expect_column_values_to_be_between", "kwargs": {"column": "quantity", "min_value": 1, "max_value": 100}},
    ])
    print(f"  Result: {'PASSED' if result['passed'] else 'FAILED'} ({result['success']}/{result['total']} expectations)")
    all_results['transactions'] = result

    # ── 4. Validate Inventory ──
    print("\n[4/4] Validating inventory.parquet...")
    inventory = pd.read_parquet("data/bronze/inventory.parquet")
    result = validate_dataset(context, inventory, "inventory_suite", [
        {"type": "expect_column_to_exist", "kwargs": {"column": "sku_id"}},
        {"type": "expect_column_to_exist", "kwargs": {"column": "current_stock"}},
        {"type": "expect_column_values_to_not_be_null", "kwargs": {"column": "sku_id"}},
        {"type": "expect_column_values_to_be_between", "kwargs": {"column": "current_stock", "min_value": 0, "max_value": 10000}},
    ])
    print(f"  Result: {'PASSED' if result['passed'] else 'FAILED'} ({result['success']}/{result['total']} expectations)")
    all_results['inventory'] = result

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  DATA QUALITY GATE SUMMARY")
    print("=" * 60)
    total_exp = sum(r['total'] for r in all_results.values())
    total_pass = sum(r['success'] for r in all_results.values())
    all_passed = all(r['passed'] for r in all_results.values())
    dq_score = (total_pass / total_exp * 100) if total_exp > 0 else 0

    for name, r in all_results.items():
        status = "PASS" if r['passed'] else "FAIL"
        print(f"  {name:20s} [{status}]  {r['success']}/{r['total']} expectations")

    print(f"\n  Overall DQ Score: {dq_score:.1f}%  (Target: >= 98%)")
    print(f"  Gate Status: {'PASSED' if all_passed else 'FAILED'}")

    # Save summary
    summary_path = f"reports/data_quality/dq_report_{timestamp}.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'dq_score': dq_score,
            'gate_passed': all_passed,
            'details': all_results
        }, f, indent=2)
    print(f"\n  Report saved to: {summary_path}")

    return all_passed


if __name__ == "__main__":
    validate_bronze_data()
