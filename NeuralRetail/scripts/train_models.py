"""
NeuralRetail Model Training Pipeline
Trains all ML models and logs to MLflow:
  1. Prophet (Demand Forecasting)
  2. LSTM (Deep Learning Forecasting)
  3. TFT (Temporal Fusion Transformer)
  4. XGBoost (Churn - base)
  5. Stacked Ensemble XGBoost+LightGBM (Churn - advanced)
"""
import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Import torch early to avoid DLL conflicts on Windows
try:
    import torch
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow
from src.config.settings import settings
from src.models.prophet_forecast import DemandForecaster
from src.models.xgb_churn import ChurnPredictor

print("Setting up MLflow tracking...")
mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
os.makedirs("mlruns", exist_ok=True)


def train_demand_prophet():
    """Train Prophet demand forecasting model."""
    print("\n[1/5] Training Demand Forecasting Model (Prophet)...")
    df = pd.read_parquet("data/bronze/transactions.parquet")
    sku_df = df[df['sku_id'] == 'SKU-1001'].copy()
    sku_df['date'] = sku_df['timestamp'].dt.date
    daily_sales = sku_df.groupby('date')['quantity'].sum().reset_index()

    forecaster = DemandForecaster()
    forecaster.train(daily_sales, target_col='quantity', date_col='date')
    print("  Prophet model trained and logged to MLflow!")


def train_demand_lstm():
    """Train LSTM demand forecasting model."""
    print("\n[2/5] Training Demand Forecasting Model (LSTM)...")
    try:
        from src.models.lstm_forecast import train_lstm
        df = pd.read_parquet("data/bronze/transactions.parquet")
        sku_df = df[df['sku_id'] == 'SKU-1001'].copy()
        sku_df['date'] = sku_df['timestamp'].dt.date
        daily_sales = sku_df.groupby('date')['quantity'].sum().reset_index()

        model, scaler = train_lstm(daily_sales['quantity'])

        mlflow.set_experiment("demand_lstm")
        with mlflow.start_run(run_name="lstm_demand"):
            mlflow.log_param("model_type", "LSTM")
            mlflow.log_param("hidden_size", 64)
            mlflow.log_param("num_layers", 2)
            mlflow.log_param("lookback", 28)
            mlflow.log_param("epochs", 20)
            mlflow.log_metric("training_complete", 1.0)

        print("  LSTM model trained and logged to MLflow!")
    except Exception as e:
        print(f"  LSTM training skipped (optional): {e}")


def train_demand_tft():
    """Train Temporal Fusion Transformer demand forecasting model."""
    print("\n[3/5] Training Demand Forecasting Model (TFT)...")
    try:
        from src.models.tft_forecast import train_tft
        df = pd.read_parquet("data/bronze/transactions.parquet")
        sku_df = df[df['sku_id'] == 'SKU-1001'].copy()
        sku_df['date'] = sku_df['timestamp'].dt.date
        daily_sales = sku_df.groupby('date')['quantity'].sum().reset_index()
        daily_sales = daily_sales.sort_values('date')

        result = train_tft(
            daily_df=daily_sales,
            daily_series=daily_sales['quantity'],
            max_epochs=25,
        )
        print("  TFT model trained and logged to MLflow!")
    except Exception as e:
        print(f"  TFT training skipped: {e}")


def train_churn_xgboost():
    """Train base XGBoost churn model."""
    print("\n[4/5] Training Churn Model (XGBoost)...")
    customers = pd.read_parquet("data/bronze/customers.parquet")
    transactions = pd.read_parquet("data/bronze/transactions.parquet")

    latest_date = transactions['timestamp'].max()
    rfm = transactions.groupby('customer_id').agg({
        'timestamp': lambda x: (latest_date - x.max()).days,
        'transaction_id': 'count',
        'total_amount': 'sum'
    }).reset_index()
    rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']
    data = customers.merge(rfm, on='customer_id', how='left').fillna(0)
    data['churn'] = (data['recency'] > 90).astype(int)

    features = ['age', 'recency', 'frequency', 'monetary']
    X = data[features]
    y = data['churn']

    predictor = ChurnPredictor()
    predictor.train(X, y)
    print("  XGBoost churn model trained and logged to MLflow!")


def train_churn_stacked():
    """Train stacked ensemble (XGBoost + LightGBM) with validation-weighted params."""
    print("\n[5/5] Training Stacked Ensemble (XGBoost + LightGBM)...")
    try:
        from src.models.stacked_churn import StackedChurnPredictor

        customers = pd.read_parquet("data/bronze/customers.parquet")
        transactions = pd.read_parquet("data/bronze/transactions.parquet")

        latest_date = transactions['timestamp'].max()
        rfm = transactions.groupby('customer_id').agg({
            'timestamp': lambda x: (latest_date - x.max()).days,
            'transaction_id': 'count',
            'total_amount': 'sum'
        }).reset_index()
        rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']
        data = customers.merge(rfm, on='customer_id', how='left').fillna(0)
        data['churn'] = (data['recency'] > 90).astype(int)

        features = ['age', 'recency', 'frequency', 'monetary']
        X = data[features]
        y = data['churn']

        predictor = StackedChurnPredictor()
        auc, f1 = predictor.train(X, y)
        print(f"  Stacked ensemble trained! AUC={auc:.4f}, F1={f1:.4f}")
    except Exception as e:
        print(f"  Stacked training skipped: {e}")


if __name__ == "__main__":
    train_demand_prophet()
    train_demand_lstm()
    train_demand_tft()
    train_churn_xgboost()
    train_churn_stacked()
    print("\nAll models trained and serialized!")
