from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from src.api.security import get_current_user
from src.config.settings import settings
import pandas as pd
from datetime import datetime, timedelta

router = APIRouter()

class DemandRequest(BaseModel):
    sku_id: str
    horizon_days: int
    store_id: str

class DemandResponse(BaseModel):
    sku_id: str
    forecast_date: str
    predicted_demand: float
    confidence_lower: float
    confidence_upper: float

# Global variable to cache the model in memory
_model = None

def get_model():
    global _model
    if _model is None:
        try:
            import mlflow
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            experiment = mlflow.get_experiment_by_name("demand_forecasting")
            if not experiment:
                raise Exception("Experiment 'demand_forecasting' not found.")
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"], max_results=1)
            if runs.empty:
                raise Exception("No runs found for demand_forecasting.")
            run_id = runs.iloc[0].run_id
            model_uri = f"runs:/{run_id}/prophet_model"
            _model = mlflow.prophet.load_model(model_uri)
        except Exception as e:
            print(f"Failed to load model: {e}")
    return _model

@router.post("/demand", response_model=DemandResponse)
async def predict_demand(request: DemandRequest, current_user = Depends(get_current_user)):
    """
    Predict demand for a given SKU and store over a specified horizon using the trained MLflow model.
    """
    model = get_model()
    if not model:
        raise HTTPException(status_code=500, detail="ML model could not be loaded.")
        
    try:
        # Create future dataframe for Prophet
        future = model.make_future_dataframe(periods=request.horizon_days)
        forecast = model.predict(future)
        
        # Get the prediction for the end of the horizon
        final_prediction = forecast.iloc[-1]
        
        return DemandResponse(
            sku_id=request.sku_id,
            forecast_date=final_prediction['ds'].strftime('%Y-%m-%d'),
            predicted_demand=round(float(final_prediction['yhat']), 2),
            confidence_lower=round(float(final_prediction['yhat_lower']), 2),
            confidence_upper=round(float(final_prediction['yhat_upper']), 2)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
