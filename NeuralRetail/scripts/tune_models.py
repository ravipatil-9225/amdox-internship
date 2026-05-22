"""
NeuralRetail -- Optuna Hyperparameter Optimization Engine
==========================================================
Automated HPO sweeping across all three model families:
  1. XGBoost  (churn classification)
  2. LightGBM (churn classification)
  3. PyTorch Lightning LSTM (demand forecasting)

Each study logs the best trial to MLflow for full traceability.
"""
import optuna
import pandas as pd
import numpy as np
import os, sys, warnings, argparse

# Import torch early to avoid DLL conflicts on Windows
try:
    import torch
except Exception:
    pass

import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, mean_squared_error
import mlflow

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config.settings import settings

mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
os.makedirs("mlruns", exist_ok=True)

# -- Data Loaders ------------------------------------------------

def load_churn_data():
    customers = pd.read_parquet("data/bronze/customers.parquet")
    transactions = pd.read_parquet("data/bronze/transactions.parquet")
    latest_date = transactions['timestamp'].max()
    rfm = transactions.groupby('customer_id').agg(
        {'timestamp': lambda x: (latest_date - x.max()).days,
         'transaction_id': 'count', 'total_amount': 'sum'}
    ).reset_index()
    rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']
    data = customers.merge(rfm, on='customer_id', how='left').fillna(0)
    data['churn'] = (data['recency'] > 90).astype(int)
    X = data[['age', 'recency', 'frequency', 'monetary']]
    y = data['churn']
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def load_demand_data(lookback=28):
    import torch
    df = pd.read_parquet("data/bronze/transactions.parquet")
    sku = df[df['sku_id'] == 'SKU-1001'].copy()
    sku['date'] = sku['timestamp'].dt.date
    daily = sku.groupby('date')['quantity'].sum().reset_index().sort_values('date')
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(daily['quantity'].values.reshape(-1, 1))
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i]); y.append(scaled[i])
    X, y = torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y))
    s = int(len(X) * 0.8)
    return X[:s], y[:s], X[s:], y[s:], scaler


# -- 1. XGBoost --------------------------------------------------

def xgb_objective(trial):
    X_tr, X_te, y_tr, y_te = load_churn_data()
    p = {
        'objective': 'binary:logistic', 'eval_metric': 'auc',
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }
    m = xgb.XGBClassifier(**p, random_state=42, use_label_encoder=False)
    m.fit(X_tr, y_tr)
    return roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])


def tune_xgboost(n_trials=15):
    print("\n[Phase 1] Optuna HPO - XGBoost Churn Classifier")
    study = optuna.create_study(direction='maximize', study_name="xgboost_churn_tuning")
    study.optimize(xgb_objective, n_trials=n_trials)
    t = study.best_trial
    print(f"  Best AUC: {t.value:.4f}")
    for k, v in t.params.items(): print(f"    {k}: {v}")
    mlflow.set_experiment("optuna_xgboost_churn")
    with mlflow.start_run(run_name="optuna_best_xgboost"):
        mlflow.log_params(t.params); mlflow.log_metric("best_auc", t.value)
    print("  [OK] Logged to MLflow.")
    return study


# -- 2. LightGBM ------------------------------------------------

def lgb_objective(trial):
    X_tr, X_te, y_tr, y_te = load_churn_data()
    bt = trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss'])
    p = {
        'objective': 'binary', 'metric': 'auc', 'boosting_type': bt,
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'verbose': -1,
    }
    if bt != 'goss':
        p['subsample'] = trial.suggest_float('subsample', 0.5, 1.0)
    m = lgb.LGBMClassifier(**p, random_state=42)
    m.fit(X_tr, y_tr)
    return roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])


def tune_lightgbm(n_trials=15):
    print("\n[Phase 1] Optuna HPO - LightGBM Churn Classifier")
    study = optuna.create_study(direction='maximize', study_name="lightgbm_churn_tuning")
    study.optimize(lgb_objective, n_trials=n_trials)
    t = study.best_trial
    print(f"  Best AUC: {t.value:.4f}")
    for k, v in t.params.items(): print(f"    {k}: {v}")
    mlflow.set_experiment("optuna_lightgbm_churn")
    with mlflow.start_run(run_name="optuna_best_lightgbm"):
        mlflow.log_params(t.params); mlflow.log_metric("best_auc", t.value)
    print("  [OK] Logged to MLflow.")
    return study


# -- 3. PyTorch Lightning LSTM ----------------------------

def _build_tunable_lstm_class():
    """Lazy-build LSTM class to avoid importing torch at module level."""
    import torch
    import torch.nn as nn
    import pytorch_lightning as pl

    class _TunableLSTM(pl.LightningModule):
        def __init__(self, hidden, layers, drop, lr):
            super().__init__()
            self.save_hyperparameters()
            self.lstm = nn.LSTM(1, hidden, layers, batch_first=True,
                                dropout=drop if layers > 1 else 0.0)
            self.fc = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(),
                                    nn.Dropout(drop), nn.Linear(hidden // 2, 1))
            self.lr = lr

        def forward(self, x):
            o, _ = self.lstm(x); return self.fc(o[:, -1, :])

        def training_step(self, batch, _):
            x, y = batch; return nn.MSELoss()(self(x), y)

        def validation_step(self, batch, _):
            x, y = batch; self.log('val_loss', nn.MSELoss()(self(x), y))

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=self.lr)

    return _TunableLSTM


def lstm_objective(trial):
    import torch
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader, TensorDataset

    lb = trial.suggest_int('lookback', 14, 42)
    hs = trial.suggest_categorical('hidden_size', [32, 64, 128])
    nl = trial.suggest_int('num_layers', 1, 3)
    dr = trial.suggest_float('dropout', 0.05, 0.4)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    bs = trial.suggest_categorical('batch_size', [16, 32, 64])
    Xt, yt, Xv, yv, _ = load_demand_data(lookback=lb)
    if len(Xt) < bs or len(Xv) < 4: return float('inf')
    tl = DataLoader(TensorDataset(Xt, yt), batch_size=bs, shuffle=True)
    vl = DataLoader(TensorDataset(Xv, yv), batch_size=bs)
    LSTMCls = _build_tunable_lstm_class()
    model = LSTMCls(hs, nl, dr, lr)
    pl.Trainer(max_epochs=15, enable_progress_bar=False,
               enable_checkpointing=False, logger=False,
               enable_model_summary=False).fit(model, tl, vl)
    model.eval()
    with torch.no_grad(): preds = model(Xv).numpy()
    return mean_squared_error(yv.numpy(), preds)


def tune_lstm(n_trials=10):
    print("\n[Phase 1] Optuna HPO - PyTorch Lightning LSTM Forecaster")
    study = optuna.create_study(direction='minimize', study_name="lstm_demand_tuning")
    study.optimize(lstm_objective, n_trials=n_trials)
    t = study.best_trial
    print(f"  Best MSE: {t.value:.6f}")
    for k, v in t.params.items(): print(f"    {k}: {v}")
    mlflow.set_experiment("optuna_lstm_demand")
    with mlflow.start_run(run_name="optuna_best_lstm"):
        mlflow.log_params(t.params); mlflow.log_metric("best_mse", t.value)
    print("  [OK] Logged to MLflow.")
    return study


# -- Main --------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuralRetail Optuna Tuning")
    parser.add_argument("--model", default="all",
                        choices=["xgboost", "lightgbm", "lstm", "all"])
    parser.add_argument("--trials", type=int, default=15)
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("  NeuralRetail - Optuna Hyperparameter Optimization")
    print("=" * 55)

    studies = {}
    if args.model in ("xgboost", "all"):
        studies['xgboost'] = tune_xgboost(args.trials)
    if args.model in ("lightgbm", "all"):
        studies['lightgbm'] = tune_lightgbm(args.trials)
    if args.model in ("lstm", "all"):
        studies['lstm'] = tune_lstm(min(args.trials, 10))

    print("\n" + "=" * 55)
    print("TUNING SUMMARY")
    print("=" * 55)
    for n, s in studies.items():
        d = "max" if s.direction.name == "MAXIMIZE" else "min"
        print(f"  {n:12s} | best={s.best_value:.6f} ({d}) | trials={len(s.trials)}")
    print("\nAll best parameters logged to MLflow. [OK]")
