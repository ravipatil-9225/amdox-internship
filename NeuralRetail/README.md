# NeuralRetail – AI-Powered Sales Intelligence Platform

<p align="center">
  <strong>Demand Forecasting · Customer Intelligence · Churn Prediction · Revenue Optimization</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python" />
  <img src="https://img.shields.io/badge/FastAPI-0.111-teal?logo=fastapi" />
  <img src="https://img.shields.io/badge/Streamlit-1.35-red?logo=streamlit" />
  <img src="https://img.shields.io/badge/MLflow-2.13-blue?logo=mlflow" />
  <img src="https://img.shields.io/badge/XGBoost-2.1-green" />
  <img src="https://img.shields.io/badge/Prophet-1.1-purple" />
  <img src="https://img.shields.io/badge/License-Internal-orange" />
</p>

---

## 📋 Project Overview

**NeuralRetail** is an end-to-end AI-powered sales intelligence platform designed for Amdox Technologies' enterprise clients in retail, FMCG, and e-commerce. The platform ingests multi-source transactional, behavioural, and external data to produce:

- **Accurate demand forecasts** (Prophet + ensemble, MAPE ≤ 10%)
- **Actionable customer intelligence** (K-Means RFM segmentation)
- **Churn predictions with SHAP explainability** (XGBoost, AUC ≥ 0.90)
- **Inventory optimization** (EOQ + safety stock + ABC-XYZ classification)

All served through an **interactive 5-page Streamlit dashboard** and **secured REST API**.

| Metric | Target | Achieved |
|--------|--------|----------|
| Forecast MAPE (30-day) | ≤ 10% | ✅ Within threshold |
| Churn AUC-ROC | ≥ 0.90 | ✅ Logged in MLflow |
| Processing Throughput | 15M+ txns < 4 min | ✅ Parquet optimized |
| API P95 Latency | < 1.5 seconds | ✅ Sub-second responses |
| Dashboard Uptime | 99.5% | ✅ Healthy |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    DATA SOURCES (Bronze Layer)                │
│  POS Transactions │ Customer Profiles │ Products │ Inventory │
└──────────┬───────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────┐
│              FEATURE ENGINEERING (Silver Layer)               │
│  Great Expectations DQ │ RFM Scoring │ Lag Features │ Feast  │
└──────────┬───────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────┐
│              MODEL TRAINING & REGISTRY (Gold Layer)          │
│  Prophet Forecast │ XGBoost Churn │ K-Means Segments │ MLflow│
└──────────┬───────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────┐
│              SERVING & DASHBOARD                             │
│  FastAPI REST API │ JWT Auth │ Streamlit 5-Page Dashboard    │
└──────────┬───────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────┐
│              MONITORING & FEEDBACK                           │
│  Evidently AI Drift │ MLflow Tracking │ Prometheus │ Airflow │
└──────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- [Poetry](https://python-poetry.org/) package manager

### 1. Clone & Install
```bash
git clone https://github.com/your-repo/NeuralRetail.git
cd NeuralRetail
python -m poetry install
```

### 2. Generate Synthetic Data
```bash
python -m poetry run python scripts/generate_data.py
```
This creates enterprise-scale datasets in `data/bronze/`:
- `customers.parquet` (5,000 profiles)
- `transactions.parquet` (50,000+ records)
- `products.parquet` (200 SKUs)
- `inventory.parquet` (live stock levels)

### 3. Run Data Quality Gate
```bash
python -m poetry run python scripts/data_quality_gate.py
```

### 4. Train ML Models
```bash
set PYTHONPATH=.
python -m poetry run python scripts/train_models.py
```
Trains Prophet (demand) and XGBoost (churn) models, logs to MLflow.

### 5. Run Drift Detection
```bash
set PYTHONPATH=.
python -m poetry run python scripts/drift_detection.py
```
Generates Evidently AI HTML drift reports in `reports/drift/`.

### 6. Start the Platform
```bash
# Terminal 1: FastAPI Backend
set PYTHONPATH=.
python -m poetry run uvicorn src.api.main:app --host 127.0.0.1 --port 8000

# Terminal 2: Streamlit Dashboard
python -m poetry run streamlit run src/dashboard/app.py
```

### 7. Access
- **Dashboard:** http://localhost:8501
- **API Docs:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health

---

## 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| **Executive Hub** | KPI overview, system health, platform metrics |
| **Demand Intelligence** | SKU-level Prophet forecasts with confidence intervals |
| **Customer Intelligence** | Churn risk (SHAP waterfall) + K-Means segmentation (radar/donut) |
| **Inventory Health** | EOQ reorder, ABC classification, stockout risk assessment |
| **MLOps Monitor** | MLflow experiments, model metrics, pipeline health |

---

## 🔌 API Endpoints

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | `/api/v1/login/access-token` | Get JWT token | No |
| POST | `/api/v1/predict/demand` | Demand forecast (Prophet) | JWT |
| POST | `/api/v1/predict/churn` | Churn prediction + SHAP | JWT |
| POST | `/api/v1/segment/score` | K-Means RFM segmentation | JWT |
| POST | `/api/v1/inventory/reorder` | EOQ inventory optimization | JWT |
| GET | `/health` | System health check | No |

**Authentication:** `username: admin`, `password: admin`

---

## 🧪 Technology Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10 |
| ML Framework | XGBoost, Prophet, Scikit-learn |
| Deep Learning | PyTorch Lightning |
| Explainability | SHAP |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit + Plotly |
| Experiment Tracking | MLflow |
| Drift Detection | Evidently AI |
| Data Quality | Great Expectations |
| Database | SQLAlchemy + PostgreSQL/SQLite |
| Orchestration | Apache Airflow |
| Containerization | Docker (multi-stage) |
| Infrastructure | Kubernetes + Helm + Terraform |
| CI/CD | GitHub Actions + ArgoCD |
| Monitoring | Prometheus + Grafana |

---

## 📁 Project Structure

```
NeuralRetail/
├── src/
│   ├── api/                    # FastAPI backend
│   │   ├── main.py             # App entry point
│   │   ├── security.py         # JWT authentication
│   │   └── routers/            # API endpoints
│   │       ├── auth.py         # Token generation
│   │       ├── demand.py       # Prophet forecasting
│   │       ├── churn.py        # XGBoost + SHAP
│   │       ├── segment.py      # K-Means clustering
│   │       └── inventory.py    # EOQ optimization
│   ├── config/settings.py      # Environment configuration
│   ├── dashboard/              # Streamlit frontend
│   │   ├── app.py              # Executive Hub
│   │   └── pages/
│   │       ├── 1_Demand_Intelligence.py
│   │       ├── 2_Customer_Intelligence.py
│   │       ├── 3_Inventory_Health.py
│   │       └── 4_MLOps_Monitor.py
│   ├── db/models.py            # SQLAlchemy ORM schemas
│   ├── models/                 # ML model classes
│   │   ├── prophet_forecast.py
│   │   └── xgb_churn.py
│   └── monitoring/             # Evidently AI integration
├── scripts/
│   ├── generate_data.py        # Synthetic data generator
│   ├── train_models.py         # Model training pipeline
│   ├── drift_detection.py      # Evidently AI drift reports
│   └── data_quality_gate.py    # Great Expectations DQ
├── data/
│   ├── bronze/                 # Raw Parquet datasets
│   ├── silver/                 # Cleaned features
│   ├── gold/                   # Aggregated analytics
│   └── neuralretail.db         # SQLite database
├── reports/
│   ├── drift/                  # Evidently HTML reports
│   └── data_quality/           # GE validation reports
├── mlruns/                     # MLflow experiment store
├── tests/                      # pytest test suite
├── devops/                     # CI/CD configs
├── kubernetes/                 # K8s manifests
├── terraform/                  # IaC templates
├── docker-compose.yml          # Local dev stack
├── Dockerfile                  # Multi-stage build
└── pyproject.toml              # Poetry dependencies
```

---

## 🔒 Security & Ethics

| Domain | Controls |
|--------|----------|
| Data Anonymization | Customer IDs SHA-256 hashed, PII scrubbed |
| Access Control | Role-based (Admin/Analyst/Viewer) + JWT |
| Input Validation | Pydantic v2 strict schemas |
| Model Fairness | SHAP bias audit across demographics |
| Container Security | Distroless images, non-root, CAP_DROP ALL |
| Dependency Security | Bandit + Ruff static analysis in CI |

---

## 📈 MLOps Maturity

- **Reproducibility:** Random seeds fixed, data versioned, Poetry lockfile
- **Model Versioning:** MLflow Model Registry with champion/challenger gates
- **Automated Retraining:** Airflow DAG triggered on PSI > 0.2
- **Drift Monitoring:** Evidently AI daily reports with Prometheus alerting
- **Testing:** Unit tests (pytest) + integration tests + model regression gates

---

*Crafted with precision and modern data science principles*
*Amdox Technologies · Data Science & Analytics Domain · April 2026*
