"""
NeuralRetail -- Stacked Ensemble Churn Predictor
==================================================
Phase 1 deliverable: XGBoost + LightGBM base learners with
**validation-performance-weighted** meta-learner.

Ensemble weights are automatically derived from each base model's
out-of-fold AUC-ROC on the validation set, rather than being
hardcoded or learned via a simple LR.
"""
import lightgbm as lgb
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
import mlflow
import joblib
import os


class StackedChurnPredictor:
    """
    Stacked ensemble: XGBoost + LightGBM base learners
    with two meta-learning strategies:
      1. Validation-AUC-weighted average (fully parameterized)
      2. Logistic Regression meta-learner on OOF predictions
    """

    def __init__(self):
        self.xgb_model = None
        self.lgb_model = None
        self.meta_model = None
        self.xgb_weight = 0.5       # dynamically updated
        self.lgb_weight = 0.5       # dynamically updated
        self.use_weighted_avg = True  # primary strategy

    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train the stacked ensemble using out-of-fold predictions."""
        mlflow.set_experiment("churn_stacked_ensemble")

        with mlflow.start_run(run_name="stacked_xgb_lgb_weighted"):
            kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            oof_xgb = np.zeros(len(X))
            oof_lgb = np.zeros(len(X))
            fold_auc_xgb = []
            fold_auc_lgb = []

            # XGBoost base learner config
            xgb_params = dict(
                n_estimators=150, learning_rate=0.05, max_depth=5,
                use_label_encoder=False, eval_metric="logloss",
                scale_pos_weight=len(y[y == 0]) / max(len(y[y == 1]), 1),
                random_state=42,
            )

            # LightGBM base learner config
            lgb_params = dict(
                n_estimators=150, learning_rate=0.05, max_depth=5,
                boosting_type='dart', random_state=42, verbose=-1,
            )

            # Generate out-of-fold predictions per fold
            for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # XGBoost fold
                xgb_fold = xgb.XGBClassifier(**xgb_params)
                xgb_fold.fit(X_train, y_train)
                oof_xgb[val_idx] = xgb_fold.predict_proba(X_val)[:, 1]
                fold_auc_xgb.append(roc_auc_score(y_val, oof_xgb[val_idx]))

                # LightGBM fold
                lgb_fold = lgb.LGBMClassifier(**lgb_params)
                lgb_fold.fit(X_train, y_train)
                oof_lgb[val_idx] = lgb_fold.predict_proba(X_val)[:, 1]
                fold_auc_lgb.append(roc_auc_score(y_val, oof_lgb[val_idx]))

            # ── Parameterize weights from validation AUC ──
            mean_auc_xgb = np.mean(fold_auc_xgb)
            mean_auc_lgb = np.mean(fold_auc_lgb)
            total_auc = mean_auc_xgb + mean_auc_lgb
            self.xgb_weight = mean_auc_xgb / total_auc
            self.lgb_weight = mean_auc_lgb / total_auc

            print(f"  Validation AUC — XGB: {mean_auc_xgb:.4f}, LGB: {mean_auc_lgb:.4f}")
            print(f"  Auto-parameterized weights — XGB: {self.xgb_weight:.4f}, "
                  f"LGB: {self.lgb_weight:.4f}")

            # Train final base models on full data
            self.xgb_model = xgb.XGBClassifier(**xgb_params)
            self.xgb_model.fit(X, y)
            self.lgb_model = lgb.LGBMClassifier(**lgb_params)
            self.lgb_model.fit(X, y)

            # ── Strategy 1: Validation-weighted average ──
            weighted_preds = (self.xgb_weight * oof_xgb +
                              self.lgb_weight * oof_lgb)
            auc_weighted = roc_auc_score(y, weighted_preds)
            f1_weighted = f1_score(y, (weighted_preds > 0.35).astype(int))

            # ── Strategy 2: LR meta-learner (backup) ──
            meta_features = np.column_stack([oof_xgb, oof_lgb])
            self.meta_model = LogisticRegression(random_state=42)
            self.meta_model.fit(meta_features, y)
            lr_preds = self.meta_model.predict_proba(meta_features)[:, 1]
            auc_lr = roc_auc_score(y, lr_preds)
            f1_lr = f1_score(y, (lr_preds > 0.35).astype(int))

            # Pick best strategy
            if auc_weighted >= auc_lr:
                self.use_weighted_avg = True
                final_auc, final_f1 = auc_weighted, f1_weighted
                strategy = "validation_weighted_average"
            else:
                self.use_weighted_avg = False
                final_auc, final_f1 = auc_lr, f1_lr
                strategy = "logistic_regression_meta"

            print(f"  Selected strategy: {strategy}")
            print(f"  Weighted-avg AUC={auc_weighted:.4f} vs LR-meta AUC={auc_lr:.4f}")

            # MLflow logging
            mlflow.log_param("base_models", "XGBoost + LightGBM DART")
            mlflow.log_param("meta_strategy", strategy)
            mlflow.log_param("n_folds", 5)
            mlflow.log_param("threshold", 0.35)
            mlflow.log_param("xgb_weight", round(self.xgb_weight, 4))
            mlflow.log_param("lgb_weight", round(self.lgb_weight, 4))
            mlflow.log_metric("xgb_mean_oof_auc", mean_auc_xgb)
            mlflow.log_metric("lgb_mean_oof_auc", mean_auc_lgb)
            mlflow.log_metric("stacked_auc_weighted", auc_weighted)
            mlflow.log_metric("stacked_auc_lr_meta", auc_lr)
            mlflow.log_metric("stacked_auc", final_auc)
            mlflow.log_metric("stacked_f1", final_f1)

            # Save models
            os.makedirs("models/artifacts", exist_ok=True)
            joblib.dump(self.xgb_model, "models/artifacts/xgb_base.pkl")
            joblib.dump(self.lgb_model, "models/artifacts/lgb_base.pkl")
            joblib.dump(self.meta_model, "models/artifacts/meta_lr.pkl")
            joblib.dump(
                {'xgb_weight': self.xgb_weight, 'lgb_weight': self.lgb_weight,
                 'strategy': strategy},
                "models/artifacts/ensemble_weights.pkl",
            )
            mlflow.log_artifacts("models/artifacts")

            print(f"  Stacked AUC-ROC: {final_auc:.4f}")
            print(f"  Stacked F1:      {final_f1:.4f}")

            return final_auc, final_f1

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict churn probability using the best ensemble strategy."""
        xgb_proba = self.xgb_model.predict_proba(X)[:, 1]
        lgb_proba = self.lgb_model.predict_proba(X)[:, 1]

        if self.use_weighted_avg:
            return self.xgb_weight * xgb_proba + self.lgb_weight * lgb_proba
        else:
            meta_features = np.column_stack([xgb_proba, lgb_proba])
            return self.meta_model.predict_proba(meta_features)[:, 1]

    @classmethod
    def load(cls, artifacts_dir="models/artifacts"):
        """Load a pre-trained ensemble from disk."""
        obj = cls()
        obj.xgb_model = joblib.load(os.path.join(artifacts_dir, "xgb_base.pkl"))
        obj.lgb_model = joblib.load(os.path.join(artifacts_dir, "lgb_base.pkl"))
        obj.meta_model = joblib.load(os.path.join(artifacts_dir, "meta_lr.pkl"))
        weights = joblib.load(os.path.join(artifacts_dir, "ensemble_weights.pkl"))
        obj.xgb_weight = weights['xgb_weight']
        obj.lgb_weight = weights['lgb_weight']
        obj.use_weighted_avg = weights['strategy'] == 'validation_weighted_average'
        return obj
