# Multi-stage build for NeuralRetail FastAPI
# Security: non-root user, slim base, minimal attack surface

# Stage 1: Build dependencies
FROM python:3.10 AS builder

WORKDIR /build
COPY requirements/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Production image
FROM python:3.10-slim AS production

# Security: create non-root user
RUN groupadd -r neuralretail && useradd -r -g neuralretail -d /app -s /bin/false neuralretail

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/

# Set ownership
RUN chown -R neuralretail:neuralretail /app

# Security: switch to non-root user
USER neuralretail

ENV PYTHONPATH=.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
