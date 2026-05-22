# NeuralRetail Architecture

## Overview

NeuralRetail employs a robust, scalable architecture consisting of the following key layers:

1.  **Data Ingestion Layer**: Handles CSV, Parquet, and API data. Data quality is verified via Great Expectations.
2.  **Feature Engineering Layer**: Feast feature store manages online (Redis) and offline (S3/Delta Lake) features.
3.  **Model Training Layer**: MLflow tracking, hyperparameter tuning with Optuna, and PyTorch/Prophet/LightGBM models.
4.  **API Serving Layer**: FastAPI backend with Pydantic validation, JWT Auth, and Redis caching.
5.  **Dashboard Layer**: Streamlit multi-page UI.
6.  **Monitoring Layer**: Evidently AI for drift detection, Prometheus/Grafana for system metrics.

## Component Diagram
(Placeholder for component diagram)

## MLOps Lifecycle
*   **Continuous Integration**: GitHub Actions for linting, testing, and security scanning (Trivy, Bandit).
*   **Continuous Deployment**: ArgoCD syncing to Kubernetes clusters.
*   **Model Monitoring**: Automated retraining triggered on high PSI (>0.2) or MAPE degradation (>15%).
