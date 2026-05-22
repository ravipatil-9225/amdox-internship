"""
NeuralRetail -- Application Settings
=====================================
Phase 4: AWS Secrets Manager integration.

Secret resolution priority:
  1. AWS Secrets Manager (when AWS_SECRETS_NAME env var is set)
  2. Environment variables / .env file
  3. Hardcoded defaults (development only)

Usage:
  Production (EKS):  Set AWS_SECRETS_NAME=neuralretail-production/app-secrets
  Local dev:         Use .env file or export env vars directly
"""
import json
import logging
import os
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger("neuralretail.settings")


# ---------------------------------------------------------------------------
# AWS Secrets Manager loader
# ---------------------------------------------------------------------------

def _load_aws_secret(secret_name: str) -> dict:
    """
    Fetch a JSON secret from AWS Secrets Manager.
    Returns empty dict if boto3 is unavailable or secret not found.
    """
    try:
        import boto3
        from botocore.exceptions import ClientError

        region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        client = boto3.client("secretsmanager", region_name=region)
        resp   = client.get_secret_value(SecretId=secret_name)
        secret = resp.get("SecretString") or ""
        data   = json.loads(secret)
        logger.info(f"[Settings] Loaded secrets from AWS Secrets Manager: {secret_name}")
        return data
    except ImportError:
        logger.debug("[Settings] boto3 not installed — skipping Secrets Manager.")
        return {}
    except Exception as exc:
        logger.warning(f"[Settings] Could not load AWS secret '{secret_name}': {exc}")
        return {}


def _resolve_secrets() -> dict:
    """
    Load secrets from AWS Secrets Manager if configured,
    inject them into the environment so Pydantic can pick them up.
    """
    secret_name = os.environ.get("AWS_SECRETS_NAME", "")
    if not secret_name:
        return {}

    secrets = _load_aws_secret(secret_name)

    # Map secret keys → environment variable names
    mapping = {
        "url":             "DATABASE_URL",
        "database_url":    "DATABASE_URL",
        "redis_url":       "REDIS_URL",
        "url":             "REDIS_URL",        # redis secret uses 'url' key
        "secret_key":      "SECRET_KEY",
        "mlflow_s3_bucket": "MLFLOW_S3_BUCKET",
        "openlineage_url": "OPENLINEAGE_URL",
        "airflow_password": "AIRFLOW_PASSWORD",
    }
    injected = {}
    for secret_key, env_key in mapping.items():
        if secret_key in secrets and env_key not in os.environ:
            os.environ[env_key] = str(secrets[secret_key])
            injected[env_key] = "***"   # Don't log actual values

    # Load DB credentials separately
    db_secret_name = os.environ.get("AWS_DB_SECRET_NAME", "")
    if db_secret_name:
        db = _load_aws_secret(db_secret_name)
        if "url" in db and "DATABASE_URL" not in os.environ:
            os.environ["DATABASE_URL"] = db["url"]

    # Load Redis credentials separately
    redis_secret_name = os.environ.get("AWS_REDIS_SECRET_NAME", "")
    if redis_secret_name:
        redis = _load_aws_secret(redis_secret_name)
        if "url" in redis and "REDIS_URL" not in os.environ:
            os.environ["REDIS_URL"] = redis["url"]

    if injected:
        logger.info(f"[Settings] Injected from Secrets Manager: {list(injected.keys())}")
    return injected


# ---------------------------------------------------------------------------
# Settings class
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    # ── Core ──────────────────────────────────────────────────────────────
    PROJECT_NAME: str = "NeuralRetail API"
    VERSION:      str = "2.0.0"
    API_V1_STR:   str = "/api/v1"
    ENVIRONMENT:  str = "development"

    # ── Security ──────────────────────────────────────────────────────────
    SECRET_KEY:                   str = "super-secret-key-change-in-production"
    ALGORITHM:                    str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES:  int = 30

    # ── Database ──────────────────────────────────────────────────────────
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/neuralretail"

    # ── Redis / Feast ──────────────────────────────────────────────────────
    REDIS_URL:        str = "redis://localhost:6379/0"
    FEAST_REDIS_HOST: str = ""
    FEAST_REDIS_PORT: int = 6379

    # ── MLflow ────────────────────────────────────────────────────────────
    MLFLOW_TRACKING_URI: str = "file:./mlruns"
    MLFLOW_S3_BUCKET:    str = ""    # Set to S3 bucket name in production

    # ── AWS ───────────────────────────────────────────────────────────────
    AWS_SECRETS_NAME:    str = ""    # e.g. neuralretail-production/app-secrets
    AWS_DB_SECRET_NAME:  str = ""    # e.g. neuralretail-production/db-credentials
    AWS_REDIS_SECRET_NAME: str = ""  # e.g. neuralretail-production/redis-credentials
    AWS_DEFAULT_REGION:  str = "us-east-1"

    # ── Observability ─────────────────────────────────────────────────────
    OPENLINEAGE_URL:           str = ""
    OPENLINEAGE_NAMESPACE:     str = "neuralretail"
    PROMETHEUS_PUSHGATEWAY:    str = ""
    DRIFT_THRESHOLD:           float = 0.20
    AIRFLOW_BASE_URL:          str = "http://localhost:8080"
    AIRFLOW_RETRAINING_DAG:    str = "model_retraining_pipeline"

    # ── SLO thresholds (used in tests) ────────────────────────────────────
    SLO_P95_LATENCY_MS: float = 500.0   # milliseconds
    SLO_ERROR_RATE_MAX: float = 0.01    # 1%

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    @property
    def mlflow_uri(self) -> str:
        """Return S3-backed MLflow URI in production, local file otherwise."""
        if self.MLFLOW_S3_BUCKET:
            return f"s3://{self.MLFLOW_S3_BUCKET}/mlflow"
        return self.MLFLOW_TRACKING_URI

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT.lower() == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Cached settings factory.
    Loads AWS Secrets Manager on first call when AWS_SECRETS_NAME is set.
    """
    _resolve_secrets()
    return Settings()


settings = get_settings()
