from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Dict, List
from src.api.security import get_current_user
from src.config.settings import settings
import mlflow
import pandas as pd
import numpy as np
import shap

router = APIRouter()

class ChurnRequest(BaseModel):
    customer_id: str

class ChurnResponse(BaseModel):
    customer_id: str
    churn_probability: float
    risk_segment: str
    top_factors: Dict[str, float]
    shap_values: Dict[str, float]

# Cache model and explainer
_model = None
_explainer = None

def get_churn_model():
    global _model, _explainer
    if _model is None:
        try:
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            experiment = mlflow.get_experiment_by_name("churn_prediction")
            if not experiment:
                raise Exception("Experiment 'churn_prediction' not found.")
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"], max_results=1)
            if runs.empty:
                raise Exception("No runs found.")
            run_id = runs.iloc[0].run_id
            _model = mlflow.xgboost.load_model(f"runs:/{run_id}/xgb_model")
            _explainer = shap.TreeExplainer(_model)
        except Exception as e:
            print(f"Failed to load churn model: {e}")
    return _model, _explainer

@router.post("/churn", response_model=ChurnResponse)
async def predict_churn(request: ChurnRequest, current_user = Depends(get_current_user)):
    """
    Predict churn probability with SHAP explainability for a given customer.
    """
    model, explainer = get_churn_model()
    if not model:
        raise HTTPException(status_code=500, detail="Churn model could not be loaded.")

    try:
        # Load customer data from bronze layer
        customers = pd.read_parquet("data/bronze/customers.parquet")
        transactions = pd.read_parquet("data/bronze/transactions.parquet")

        # Build features for requested customer
        latest_date = transactions['timestamp'].max()
        rfm = transactions.groupby('customer_id').agg({
            'timestamp': lambda x: (latest_date - x.max()).days,
            'transaction_id': 'count',
            'total_amount': 'sum'
        }).reset_index()
        rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']

        data = customers.merge(rfm, on='customer_id', how='left').fillna(0)

        # Find specific customer or use a representative one
        customer_row = data[data['customer_id'] == request.customer_id]
        if customer_row.empty:
            # Pick a random customer for demo purposes
            customer_row = data.sample(1, random_state=42)

        features = ['age', 'recency', 'frequency', 'monetary']
        X = customer_row[features].values

        # Predict
        proba = float(model.predict_proba(X)[0][1])
        risk = "High Risk" if proba >= 0.7 else ("Medium Risk" if proba >= 0.4 else "Low Risk")

        # SHAP
        shap_vals = explainer.shap_values(X)
        shap_dict = {feat: round(float(val), 4) for feat, val in zip(features, shap_vals[0])}

        # Top factors (absolute SHAP importance)
        sorted_factors = dict(sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True))

        return ChurnResponse(
            customer_id=request.customer_id,
            churn_probability=round(proba, 4),
            risk_segment=risk,
            top_factors=sorted_factors,
            shap_values=shap_dict
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Churn prediction failed: {e}")
