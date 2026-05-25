# NeuralRetail Architecture

## Overview

NeuralRetail employs a **Lakehouse + Feature Store + MLOps** architecture with five distinct layers.
The platform is designed for horizontal scalability, full observability, and automated model lifecycle management.

## MLOps Pipeline Architecture

```mermaid
flowchart TB
    subgraph Sources["Data Sources"]
        POS["POS Transactions"]
        ECOM["E-Commerce Events"]
        ERP["ERP Inventory Feeds"]
        WEATHER["Weather API"]
        COMP["Competitor Pricing"]
    end

    subgraph Ingestion["Data Ingestion Layer"]
        SPARK["Apache Spark + Delta Lake"]
        GE["Great Expectations DQ Gates"]
        OL["OpenLineage + Marquez"]
        KAFKA["Kafka Consumer"]
    end

    subgraph Features["Feature Engineering Layer"]
        FEAST["Feast Feature Store"]
        REDIS["Redis Online Store"]
        S3["S3 Offline Store"]
        POLARS["Polars Transforms"]
    end

    subgraph Training["Model Training & Registry"]
        PROPHET["Prophet Forecasting"]
        LSTM["LSTM PyTorch Lightning"]
        XGB["XGBoost + LightGBM Stack"]
        OPTUNA["Optuna HPO"]
        SHAP["SHAP + LIME + Captum"]
        MLFLOW["MLflow Registry"]
    end

    subgraph Serving["Serving & Dashboard"]
        FASTAPI["FastAPI REST API"]
        JWT["JWT Authentication"]
        STREAMLIT["Streamlit 5-Page Dashboard"]
        EXPORT["PDF/Excel Export"]
    end

    subgraph Monitoring["Monitoring & Feedback"]
        EVIDENTLY["Evidently AI Drift"]
        PROM["Prometheus + Grafana"]
        LOKI["Loki Log Aggregation"]
        AIRFLOW["Airflow Auto-Retrain"]
    end

    Sources --> Ingestion
    Ingestion --> Features
    Features --> Training
    Training --> Serving
    Serving --> Monitoring
    Monitoring -->|"PSI > 0.2"| Training
```

## Data Flow Patterns

```mermaid
flowchart LR
    A["Bronze Layer<br/>Raw Parquets"] -->|"GE Validation"| B["Silver Layer<br/>Clean Features"]
    B -->|"Feast Materialise"| C["Feature Store<br/>Redis + S3"]
    C -->|"Training Pipeline"| D["Gold Layer<br/>Model Artifacts"]
    D -->|"MLflow Registry"| E["Champion Model<br/>FastAPI Serving"]
    E -->|"Daily Scoring"| F["Evidently AI<br/>Drift Detection"]
    F -->|"PSI > 0.2"| G["Airflow DAG<br/>Auto-Retrain"]
    G -->|"New Challenger"| D
```

## Kubernetes Deployment Architecture

```mermaid
flowchart TB
    subgraph EKS["AWS EKS Cluster"]
        subgraph NS["neuralretail namespace"]
            CHAMP["Champion Deployment<br/>FastAPI (replicas: 2)"]
            SHADOW["Shadow Deployment<br/>Challenger Model"]
            SL["Streamlit Service"]
        end
        subgraph ISTIO["Istio Service Mesh"]
            VS["VirtualService<br/>Traffic Splitting"]
        end
        subgraph MON["Monitoring"]
            PROM2["Prometheus"]
            GRAF["Grafana"]
            LOKI2["Loki"]
        end
    end
    
    INGRESS["HTTPS Ingress<br/>cert-manager TLS"] --> VS
    VS -->|"95% traffic"| CHAMP
    VS -->|"5% shadow"| SHADOW
    PROM2 -->|"Scrape /metrics"| CHAMP
```

## Component Responsibilities

| Component | Technology | Responsibility |
|-----------|-----------|----------------|
| Data Ingestion | Spark + Delta Lake | Multi-source ETL with schema enforcement |
| Data Quality | Great Expectations + Soda | Automated DQ gates on every pipeline run |
| Data Lineage | OpenLineage + Marquez | End-to-end dataset provenance tracking |
| Feature Store | Feast (Redis + S3) | Consistent train/serve feature management |
| ML Training | XGBoost + LightGBM + Prophet + LSTM | Ensemble models with Optuna HPO |
| Experiment Tracking | MLflow 2.13 | Versioned experiments, model registry |
| Explainability | SHAP + LIME + Captum | Feature attribution for all model types |
| Causal Inference | DoWhy + EconML | Price elasticity and promotion attribution |
| Drift Detection | Evidently AI | PSI-based drift with auto-retrain trigger |
| API Serving | FastAPI + Uvicorn | REST endpoints with Pydantic validation |
| Dashboard | Streamlit + Plotly | 5-page interactive analytics UI |
| Orchestration | Apache Airflow | DAG scheduling with retry and alerting |
| Containerization | Docker (multi-stage) | Non-root, slim base, HEALTHCHECK |
| Infrastructure | Kubernetes + Helm 3 | EKS deployment with HPA autoscaling |
| CI/CD | GitHub Actions + ArgoCD | Lint → test → build → deploy → SLO gate |
| Monitoring | Prometheus + Grafana + Loki | Metrics, dashboards, log aggregation |
| IaC | Terraform + Terragrunt | AWS multi-resource provisioning |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `sqlite:///data/neuralretail.db` |
| `REDIS_URL` | Redis connection for feature cache | `redis://localhost:6379/0` |
| `MLFLOW_TRACKING_URI` | MLflow server URI | `file:./mlruns` |
| `DRIFT_THRESHOLD` | PSI threshold for retraining trigger | `0.20` |
| `AIRFLOW_BASE_URL` | Airflow REST API base URL | `http://localhost:8080` |
| `PROMETHEUS_PUSHGATEWAY` | Pushgateway URL for metrics | _(empty)_ |
| `SECRET_KEY` | JWT signing secret | `neuralretail-secret-key` |
| `API_KEY` | API key for key-based auth | `neural_secret_key_prod_123!` |

## MLOps Lifecycle

- **Continuous Integration**: GitHub Actions for linting (Ruff + Black), testing (pytest), and security scanning (Bandit).
- **Continuous Deployment**: ArgoCD syncing Kubernetes manifests from Git. Staging auto-deploy on merge; production requires manual approval.
- **Model Monitoring**: Automated retraining triggered on high PSI (>0.2) or MAPE degradation (>15%).
- **Champion/Challenger**: Shadow deployment pattern via Istio VirtualService. Promote only if challenger improves AUC by ≥5%.
