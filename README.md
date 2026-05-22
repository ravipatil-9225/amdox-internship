# NeuralRetail - AI Sales Intelligence Platform

Welcome to the **NeuralRetail** project! This is an advanced AI Sales Intelligence Platform developed as part of an Amdocs Internship.

## 🚀 Overview

NeuralRetail is a comprehensive end-to-end Machine Learning and AI platform designed for the retail domain. It provides actionable intelligence through predictive modeling, data monitoring, and interactive dashboards.

### Key Capabilities:
- **Demand Forecasting**: Time-series forecasting using state-of-the-art models including LSTM, Prophet, and Temporal Fusion Transformers (TFT).
- **Customer Intelligence (Churn)**: Predicting customer churn using advanced ensemble methods (XGBoost, LightGBM, Stacked models).
- **Price Elasticity**: Advanced econometric modeling for price optimization.
- **Inventory Health**: Insights into stock management and supply chain efficiency.
- **MLOps & Monitoring**: Continuous monitoring of data quality, feature drift, and model performance.

## 🛠️ Technology Stack

- **Backend / API**: FastAPI, Uvicorn, Pydantic
- **Frontend / Dashboard**: Streamlit, Plotly
- **Machine Learning**: PyTorch, XGBoost, LightGBM, Prophet, DoWhy, EconML
- **Data Engineering & Feature Store**: Feast, PySpark, Delta-Spark
- **MLOps & Monitoring**: MLflow, Evidently, Great Expectations, Prometheus
- **Database**: SQLAlchemy, Alembic, Redis
- **Testing & QA**: Pytest, K6 (Load Testing)
- **Infrastructure**: Terraform (AWS)
- **Package Management**: Poetry

## 📂 Project Structure

```text
.
├── NeuralRetail/
│   ├── models/           # Pre-trained models and artifacts
│   ├── notebooks/        # Jupyter notebooks for EDA and experimentation
│   ├── reports/          # Data quality, drift, and lineage reports
│   ├── scripts/          # Automation scripts (training, tuning, drift detection)
│   ├── src/
│   │   ├── api/          # FastAPI application and routers
│   │   ├── config/       # Application settings
│   │   ├── dashboard/    # Streamlit interactive dashboard UI
│   │   ├── db/           # Database models and schema
│   │   ├── feature_store/# Feast feature store definitions
│   │   ├── models/       # ML model architectures and training logic
│   │   ├── monitoring/   # Data and model drift monitoring (Evidently)
│   │   └── pipelines/    # Data ingestion and processing pipelines
│   ├── terraform/        # Infrastructure as code (AWS)
│   ├── tests/            # Unit and integration tests (Pytest, K6)
│   └── pyproject.toml    # Poetry dependencies configuration
└── README.md             # This file
```

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or 3.11
- [Poetry](https://python-poetry.org/) (for dependency management)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ravipatil-9225/amdox-internship.git
   cd amdox-internship/NeuralRetail
   ```
2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

### Running the Application
**Start the FastAPI Backend:**
```bash
poetry run uvicorn src.api.main:app --reload
```

**Start the Streamlit Dashboard:**
```bash
poetry run streamlit run src/dashboard/app.py
```

## 📊 Monitoring & Reports
The platform automatically generates reports for Data Quality and Data Drift (powered by Evidently). You can find these HTML and JSON reports under the `NeuralRetail/reports/` directory.

## 🤝 Contribution & Testing
Run tests using pytest:
```bash
poetry run pytest tests/
```
Load testing scripts are available in the `tests/` directory and can be executed using K6.
