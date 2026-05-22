"""
LSTM Demand Forecasting Model (PyTorch Lightning)
Window-based input with multi-step output for SKU-level demand prediction.
"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import mlflow


class LSTMForecaster(pl.LightningModule):
    """LSTM model for time-series demand forecasting."""

    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
        return [optimizer], [scheduler]


def create_sequences(data, lookback=28):
    """Create sliding window sequences for LSTM input."""
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i])
    return np.array(X), np.array(y)


def train_lstm(daily_sales: pd.DataFrame, lookback=28):
    """Train LSTM model on daily sales data."""
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(daily_sales.values.reshape(-1, 1))

    X, y = create_sequences(scaled, lookback)
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y)

    # Split
    split = int(len(X) * 0.8)
    train_dataset = torch.utils.data.TensorDataset(X[:split], y[:split])
    val_dataset = torch.utils.data.TensorDataset(X[split:], y[split:])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

    # Train
    model = LSTMForecaster(input_size=1, hidden_size=64, num_layers=2)
    trainer = pl.Trainer(max_epochs=20, enable_progress_bar=True, enable_checkpointing=False, logger=False)
    trainer.fit(model, train_loader, val_loader)

    return model, scaler
