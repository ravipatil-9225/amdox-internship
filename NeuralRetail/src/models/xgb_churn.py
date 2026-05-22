import xgboost as xgb
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score

class ChurnPredictor:
    def __init__(self):
        self.model = None

    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Train XGBoost Churn Model.
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        mlflow.set_experiment("churn_prediction")
        with mlflow.start_run():
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                use_label_encoder=False,
                eval_metric="logloss"
            )
            self.model.fit(X_train, y_train)
            
            preds = self.model.predict(X_test)
            preds_proba = self.model.predict_proba(X_test)[:, 1]
            
            auc = roc_auc_score(y_test, preds_proba)
            f1 = f1_score(y_test, preds)
            
            mlflow.log_metric("auc", auc)
            mlflow.log_metric("f1_score", f1)
            mlflow.xgboost.log_model(self.model, "xgb_model")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if not self.model:
            raise ValueError("Model not trained yet.")
        return self.model.predict_proba(X)[:, 1]
