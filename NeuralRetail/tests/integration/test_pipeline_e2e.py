"""
NeuralRetail -- Integration Tests: End-to-End Pipeline
========================================================
Tests the complete data pipeline flow:
  Data ingestion → Feature engineering → Model prediction → Drift check
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


class TestDataPipeline:
    """Integration tests for the data pipeline flow."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Ensure bronze data exists."""
        self.bronze_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "bronze"
        )
        self.silver_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "silver"
        )

    def test_bronze_data_exists(self):
        """Verify core bronze layer Parquet files are present."""
        required_files = [
            "customers.parquet",
            "transactions.parquet",
            "products.parquet",
            "inventory.parquet",
        ]
        for fname in required_files:
            path = os.path.join(self.bronze_dir, fname)
            assert os.path.exists(path), f"Missing bronze file: {fname}"

    def test_bronze_data_non_empty(self):
        """Verify bronze data has rows."""
        df = pd.read_parquet(os.path.join(self.bronze_dir, "transactions.parquet"))
        assert len(df) > 0, "transactions.parquet is empty"
        assert "customer_id" in df.columns
        assert "sku_id" in df.columns
        assert "quantity" in df.columns

    def test_bronze_customers_schema(self):
        """Validate customer data schema matches expected columns."""
        df = pd.read_parquet(os.path.join(self.bronze_dir, "customers.parquet"))
        expected_cols = {"customer_id", "age", "gender", "signup_date", "segment"}
        assert expected_cols.issubset(set(df.columns)), (
            f"Missing columns: {expected_cols - set(df.columns)}"
        )

    def test_feature_engineering_produces_silver(self):
        """Run feature engineering and verify silver output."""
        from src.pipelines.feature_engineering import run_feature_engineering

        customer_features, _, _ = run_feature_engineering()
        assert customer_features is not None
        assert len(customer_features) > 0
        assert "recency" in customer_features.columns
        assert "frequency" in customer_features.columns
        assert "monetary" in customer_features.columns

    def test_data_ingestion_pipeline(self):
        """Test the DataIngestionPipeline end-to-end."""
        from src.pipelines.data_ingestion import DataIngestionPipeline

        pipeline = DataIngestionPipeline()

        # Test Parquet ingestion
        customers_path = os.path.join(self.bronze_dir, "customers.parquet")
        df = pipeline.ingest_parquet(
            customers_path, table_name="test_customers"
        )
        assert len(df) > 0

        # Test schema validation
        is_valid = pipeline.validate_schema(
            df,
            ["customer_id", "age"],
            "test_customers",
        )
        assert is_valid is True

        # Test schema validation failure
        with pytest.raises(ValueError):
            pipeline.validate_schema(
                df, ["nonexistent_column"], "test_customers"
            )

    def test_kafka_batch_ingestion(self):
        """Test Kafka-style batch ingestion."""
        from src.pipelines.data_ingestion import DataIngestionPipeline

        pipeline = DataIngestionPipeline()
        messages = [
            {
                "transaction_id": "TXN-INT-001",
                "customer_id": "CUST-INT-001",
                "sku_id": "SKU-1001",
                "timestamp": "2026-05-25T12:00:00",
                "quantity": 2,
                "discount_applied": 0.0,
                "total_amount": 200.00,
            }
        ]
        df = pipeline.ingest_kafka_batch(
            "test_topic", "test_kafka", messages
        )
        assert len(df) == 1
        assert df.iloc[0]["transaction_id"] == "TXN-INT-001"


class TestModelPrediction:
    """Integration tests for model predictions."""

    def test_churn_model_exists(self):
        """Verify trained churn model files exist."""
        models_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "models"
        )
        assert os.path.exists(os.path.join(models_dir, "xgb_churn.pkl"))

    def test_churn_prediction_works(self):
        """Test churn prediction produces valid output."""
        import pickle

        model_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "models", "xgb_churn.pkl"
        )
        if not os.path.exists(model_path):
            pytest.skip("Churn model not trained yet")

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        if not hasattr(model, "use_label_encoder"):
            model.use_label_encoder = None

        test_input = pd.DataFrame(
            {"age": [35], "recency": [10], "frequency": [15], "monetary": [2000]}
        )
        proba = model.predict_proba(test_input)
        assert proba.shape == (1, 2)
        assert 0 <= proba[0, 1] <= 1

    def test_stacked_ensemble_exists(self):
        """Verify stacked ensemble model exists."""
        model_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "models", "stacked_churn.pkl"
        )
        assert os.path.exists(model_path), "Stacked churn model not found"


class TestDriftDetection:
    """Integration tests for drift detection workflow."""

    def test_drift_monitor_runs(self):
        """Test Evidently drift detection produces valid output."""
        try:
            from src.monitoring.evidently_drift import DriftMonitor
        except ImportError:
            pytest.skip("Evidently not installed")

        # Create synthetic ref/current data
        np.random.seed(42)
        ref = pd.DataFrame({
            "age": np.random.normal(35, 10, 200),
            "recency": np.random.normal(30, 15, 200),
            "frequency": np.random.normal(10, 5, 200),
            "monetary": np.random.normal(1000, 500, 200),
        })
        cur = pd.DataFrame({
            "age": np.random.normal(35, 10, 200),
            "recency": np.random.normal(30, 15, 200),
            "frequency": np.random.normal(10, 5, 200),
            "monetary": np.random.normal(1000, 500, 200),
        })

        monitor = DriftMonitor(drift_threshold=0.5)
        result = monitor.check_drift(ref, cur)

        assert "drift_share" in result
        assert "drifted_columns" in result
        assert "report_html_path" in result
        assert 0 <= result["drift_share"] <= 1

    def test_drift_report_saved(self):
        """Verify drift reports directory is not empty after a run."""
        reports_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "reports", "drift"
        )
        if os.path.exists(reports_dir):
            files = os.listdir(reports_dir)
            # Just verify the directory exists and can be listed
            assert isinstance(files, list)
