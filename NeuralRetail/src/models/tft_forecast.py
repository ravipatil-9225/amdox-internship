"""
NeuralRetail -- Temporal Fusion Transformer (TFT) Demand Forecaster
====================================================================
Phase 1 deliverable: Advanced multi-horizon demand forecasting using
PyTorch Forecasting's TFT to capture complex temporal dependencies
and exogenous variables (day-of-week, month, promotions, price).

Falls back to a lightweight PyTorch-native Transformer if
pytorch-forecasting is unavailable.
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import mlflow
import joblib

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# --- Try PyTorch Forecasting TFT first, fallback to native -------

_USE_PTF = False
try:
    from pytorch_forecasting import (
        TimeSeriesDataSet,
        TemporalFusionTransformer,
        QuantileLoss,
    )
    from pytorch_forecasting.data import GroupNormalizer
    _USE_PTF = True
except ImportError:
    pass


# =====================================================================
# 1. PyTorch Forecasting TFT (production path)
# =====================================================================

def _prepare_ptf_dataset(daily_df: pd.DataFrame, max_encoder_length=30,
                         max_prediction_length=7):
    """
    Build a TimeSeriesDataSet from daily SKU sales with exogenous vars.
    """
    df = daily_df.copy()
    df = df.sort_values('date').reset_index(drop=True)
    df['time_idx'] = np.arange(len(df))
    df['group'] = '0'  # single series

    # Exogenous features
    df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek.astype(str)
    df['month'] = pd.to_datetime(df['date']).dt.month.astype(str)

    cutoff = len(df) - max_prediction_length

    training = TimeSeriesDataSet(
        df[:cutoff],
        time_idx='time_idx',
        target='quantity',
        group_ids=['group'],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=['group'],
        time_varying_known_categoricals=['day_of_week', 'month'],
        time_varying_unknown_reals=['quantity'],
        target_normalizer=GroupNormalizer(groups=['group']),
    )

    validation = TimeSeriesDataSet.from_dataset(
        training, df, min_prediction_idx=cutoff
    )
    return training, validation


def train_tft_ptf(daily_df: pd.DataFrame, max_epochs=25):
    """Train a TFT via pytorch-forecasting and log to MLflow."""
    print("  Using PyTorch Forecasting TFT implementation")
    training_ds, val_ds = _prepare_ptf_dataset(daily_df)

    train_dl = training_ds.to_dataloader(train=True, batch_size=32, num_workers=0)
    val_dl = val_ds.to_dataloader(train=False, batch_size=32, num_workers=0)

    tft = TemporalFusionTransformer.from_dataset(
        training_ds,
        hidden_size=32,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=16,
        loss=QuantileLoss(),
        optimizer="adam",
        learning_rate=1e-3,
        reduce_on_plateau_patience=3,
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        enable_progress_bar=True,
        enable_checkpointing=False,
        gradient_clip_val=0.5,
        logger=False,
    )
    trainer.fit(tft, train_dl, val_dl)

    # Evaluate
    preds = tft.predict(val_dl, return_y=True)
    mae = float(torch.mean(torch.abs(preds.output - preds.y[0])))

    # Log to MLflow
    mlflow.set_experiment("demand_tft")
    with mlflow.start_run(run_name="tft_demand"):
        mlflow.log_param("model_type", "TemporalFusionTransformer")
        mlflow.log_param("hidden_size", 32)
        mlflow.log_param("attention_heads", 2)
        mlflow.log_param("max_epochs", max_epochs)
        mlflow.log_param("backend", "pytorch_forecasting")
        mlflow.log_metric("val_mae", mae)
        mlflow.log_metric("training_complete", 1.0)

    print(f"  TFT validation MAE: {mae:.4f}")
    return tft, mae


# =====================================================================
# 2. Native PyTorch Transformer Fallback
# =====================================================================

class TransformerForecaster(pl.LightningModule):
    """
    Lightweight Transformer encoder for time-series forecasting.
    Uses positional encoding + multi-head self-attention to capture
    multi-horizon temporal dependencies.
    """

    def __init__(self, input_size=1, d_model=64, nhead=4,
                 num_layers=2, dim_ff=128, dropout=0.1,
                 output_size=1, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.d_model = d_model

        # Project input features into d_model dimensions
        self.input_proj = nn.Linear(input_size, d_model)

        # Learnable positional encoding
        self.pos_enc = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc_out = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_size),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        seq_len = x.size(1)
        x = self.input_proj(x)              # -> (batch, seq, d_model)
        x = x + self.pos_enc[:, :seq_len, :]
        x = self.transformer(x)             # -> (batch, seq, d_model)
        x = self.fc_out(x[:, -1, :])        # last step -> (batch, output)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.MSELoss()(self(x), y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.MSELoss()(self(x), y)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10)
        return [opt], [sched]


def create_sequences(data, lookback=30):
    """Sliding-window sequences for Transformer input."""
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i])
    return np.array(X), np.array(y)


def train_tft_native(daily_sales_series: pd.Series, lookback=30, max_epochs=30):
    """
    Train a native PyTorch Transformer forecaster as fallback.
    """
    print("  Using native PyTorch Transformer implementation (fallback)")
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(daily_sales_series.values.reshape(-1, 1))

    X, y = create_sequences(scaled, lookback)
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y)

    split = int(len(X) * 0.8)
    train_ds = TensorDataset(X[:split], y[:split])
    val_ds = TensorDataset(X[split:], y[split:])
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=32)

    model = TransformerForecaster(
        input_size=1, d_model=64, nhead=4,
        num_layers=2, dim_ff=128, dropout=0.1, lr=1e-3,
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        enable_progress_bar=True,
        enable_checkpointing=False,
        logger=False,
    )
    trainer.fit(model, train_dl, val_dl)

    # Evaluate
    model.eval()
    with torch.no_grad():
        preds = model(X[split:]).numpy()
    mse = float(np.mean((y[split:].numpy() - preds) ** 2))
    mae = float(np.mean(np.abs(y[split:].numpy() - preds)))

    # Log to MLflow
    mlflow.set_experiment("demand_tft")
    with mlflow.start_run(run_name="tft_demand_native"):
        mlflow.log_param("model_type", "TransformerForecaster")
        mlflow.log_param("d_model", 64)
        mlflow.log_param("nhead", 4)
        mlflow.log_param("num_layers", 2)
        mlflow.log_param("lookback", lookback)
        mlflow.log_param("max_epochs", max_epochs)
        mlflow.log_param("backend", "native_pytorch")
        mlflow.log_metric("val_mse", mse)
        mlflow.log_metric("val_mae", mae)
        mlflow.log_metric("training_complete", 1.0)

    # Save artifacts
    os.makedirs("models/artifacts", exist_ok=True)
    torch.save(model.state_dict(), "models/artifacts/tft_native.pt")
    joblib.dump(scaler, "models/artifacts/tft_scaler.pkl")

    print(f"  Transformer val MSE: {mse:.6f}, MAE: {mae:.6f}")
    return model, scaler, mse


# =====================================================================
# Public entry point
# =====================================================================

def train_tft(daily_df: pd.DataFrame = None, daily_series: pd.Series = None,
              max_epochs=25):
    """
    Train the best available TFT implementation.
    Pass daily_df (with columns date, quantity) for PTF path,
    or daily_series for native fallback.
    """
    if _USE_PTF and daily_df is not None:
        return train_tft_ptf(daily_df, max_epochs=max_epochs)
    elif daily_series is not None:
        return train_tft_native(daily_series, max_epochs=max_epochs)
    elif daily_df is not None:
        return train_tft_native(daily_df['quantity'], max_epochs=max_epochs)
    else:
        raise ValueError("Provide either daily_df or daily_series")


# =====================================================================
# Self-test
# =====================================================================

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from src.config.settings import settings
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

    print("\n[TFT] Loading data...")
    df = pd.read_parquet("data/bronze/transactions.parquet")
    sku = df[df['sku_id'] == 'SKU-1001'].copy()
    sku['date'] = sku['timestamp'].dt.date
    daily = sku.groupby('date')['quantity'].sum().reset_index()
    daily = daily.sort_values('date')

    print(f"[TFT] {len(daily)} daily records, backend={'PTF' if _USE_PTF else 'Native'}")
    result = train_tft(daily_df=daily, daily_series=daily['quantity'])
    print("[TFT] Training complete! [OK]")
