"""
NeuralRetail -- FastAPI Application Entry Point
================================================
Phase 4: Added real Prometheus metrics middleware.

Metrics exposed at GET /metrics (scraped by Prometheus):
  neuralretail_http_requests_total          (counter, labels: method, endpoint, status)
  neuralretail_http_request_duration_seconds (histogram, labels: method, endpoint)
  neuralretail_model_inference_duration_seconds (histogram, labels: model_name)
  neuralretail_drift_share                  (gauge)
  neuralretail_active_requests              (gauge)

SLO: P95 latency < 500ms on /api/v1/predict/* endpoints.
     Prometheus alert rule fires when p95 > 0.5s over 5m window.
"""
import time
import os

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from src.config.settings import settings
from src.api.routers import demand, churn, segment, inventory, auth, revenue

# ---------------------------------------------------------------------------
# Prometheus instrumentation
# ---------------------------------------------------------------------------
try:
    from prometheus_client import (
        Counter, Histogram, Gauge,
        CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
        REGISTRY,
    )
    _HAS_PROMETHEUS = True
except ImportError:
    _HAS_PROMETHEUS = False

if _HAS_PROMETHEUS:
    # HTTP request counter
    HTTP_REQUESTS = Counter(
        "neuralretail_http_requests_total",
        "Total HTTP requests",
        ["method", "endpoint", "status_code"],
    )
    # P95 latency histogram (SLO target: p95 < 0.5s)
    HTTP_LATENCY = Histogram(
        "neuralretail_http_request_duration_seconds",
        "HTTP request duration in seconds",
        ["method", "endpoint"],
        buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
    )
    # Per-model inference latency
    INFERENCE_LATENCY = Histogram(
        "neuralretail_model_inference_duration_seconds",
        "Model inference latency in seconds",
        ["model_name"],
        buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
    )
    # Drift gauge (updated by drift monitor)
    DRIFT_SHARE = Gauge(
        "neuralretail_feature_drift_share",
        "Current feature drift share (0.0–1.0)",
    )
    # Active in-flight requests
    ACTIVE_REQUESTS = Gauge(
        "neuralretail_active_requests",
        "Number of requests currently being processed",
    )

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    description=(
        "NeuralRetail Enterprise AI Platform — production-grade REST API. "
        "SLO: P95 latency < 500ms on /predict/* endpoints."
    ),
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Prometheus middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    """
    Intercept every HTTP request to:
      1. Track active in-flight count
      2. Record request duration histogram (for P95 SLO)
      3. Increment request counter by method/endpoint/status
      4. Attach X-Process-Time header
    """
    # Normalise endpoint path (strip path params for cardinality control)
    path = request.url.path
    if "/predict/" in path or path.startswith("/api/v1/predict"):
        endpoint = "/api/v1/predict/*"
    elif "/segment" in path:
        endpoint = "/api/v1/segment/*"
    elif "/revenue" in path:
        endpoint = "/api/v1/revenue/*"
    elif "/inventory" in path:
        endpoint = "/api/v1/inventory/*"
    else:
        endpoint = path

    start = time.perf_counter()

    if _HAS_PROMETHEUS:
        ACTIVE_REQUESTS.inc()

    try:
        response = await call_next(request)
        status_code = str(response.status_code)
    except Exception as exc:
        status_code = "500"
        if _HAS_PROMETHEUS:
            ACTIVE_REQUESTS.dec()
        raise exc

    duration = time.perf_counter() - start

    if _HAS_PROMETHEUS:
        ACTIVE_REQUESTS.dec()
        HTTP_LATENCY.labels(
            method=request.method, endpoint=endpoint
        ).observe(duration)
        HTTP_REQUESTS.labels(
            method=request.method,
            endpoint=endpoint,
            status_code=status_code,
        ).inc()

    response.headers["X-Process-Time"] = f"{duration:.6f}"
    return response

# ---------------------------------------------------------------------------
# Prometheus scrape endpoint
# ---------------------------------------------------------------------------

@app.get("/metrics", include_in_schema=False)
async def metrics():
    """Prometheus scrape endpoint. Returns metrics in text/plain exposition format."""
    if not _HAS_PROMETHEUS:
        return Response(
            content="# prometheus_client not installed\n",
            media_type="text/plain",
        )
    return Response(
        content=generate_latest(REGISTRY),
        media_type=CONTENT_TYPE_LATEST,
    )

# ---------------------------------------------------------------------------
# Health & readiness
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
async def health_check():
    """Kubernetes liveness probe endpoint."""
    return {"status": "ok", "version": settings.VERSION}

@app.get("/ready", tags=["System"])
async def readiness_check():
    """
    Kubernetes readiness probe — verifies models are loaded.
    Returns 503 if any critical dependency is unavailable.
    """
    checks: dict = {"models": "ok", "version": settings.VERSION}
    import os
    models_dir = "models"
    required = ["xgb_churn.pkl"]
    missing = [m for m in required if not os.path.exists(os.path.join(models_dir, m))]
    if missing:
        checks["models"] = f"missing: {missing}"
        return Response(
            content=str(checks),
            status_code=503,
            media_type="application/json",
        )
    return checks

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(auth.router,      prefix=settings.API_V1_STR)
app.include_router(demand.router,    prefix=settings.API_V1_STR + "/predict",   tags=["Demand Forecasting"])
app.include_router(churn.router,     prefix=settings.API_V1_STR + "/predict",   tags=["Churn Prediction"])
app.include_router(segment.router,   prefix=settings.API_V1_STR + "/segment",   tags=["Customer Segmentation"])
app.include_router(inventory.router, prefix=settings.API_V1_STR + "/inventory", tags=["Inventory Optimization"])
app.include_router(revenue.router,   prefix=settings.API_V1_STR + "/revenue",   tags=["Revenue & Price Intelligence"])
