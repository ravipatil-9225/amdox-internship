import pandas as pd
from prophet import Prophet
import mlflow
import optuna

class DemandForecaster:
    def __init__(self):
        self.model = None

    def train(self, df: pd.DataFrame, target_col: str, date_col: str):
        """
        Train Prophet model.
        """
        df_prophet = df[[date_col, target_col]].rename(columns={date_col: 'ds', target_col: 'y'})
        
        mlflow.set_experiment("demand_forecasting")
        with mlflow.start_run():
            self.model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
            self.model.fit(df_prophet)
            mlflow.prophet.log_model(self.model, "prophet_model")

    def predict(self, horizon: int) -> pd.DataFrame:
        """
        Generate forecasts.
        """
        if not self.model:
            raise ValueError("Model not trained yet.")
        future = self.model.make_future_dataframe(periods=horizon)
        forecast = self.model.predict(future)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(horizon)
