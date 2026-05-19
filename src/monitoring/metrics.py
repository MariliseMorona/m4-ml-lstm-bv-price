"""Métricas Prometheus para monitoramento da API em produção."""

from __future__ import annotations

import time
from contextlib import contextmanager

from prometheus_client import Counter, Gauge, Histogram

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

PREDICTIONS_TOTAL = Counter(
    "predictions_total",
    "Total predictions served",
    ["symbol"],
)

MODEL_LOADED = Gauge(
    "model_loaded",
    "Whether the LSTM model is loaded (1=yes, 0=no)",
)


@contextmanager
def track_request(method: str, endpoint: str):
    """Context manager que registra latência e status ao final da requisição.

    Args:
        method: Verbo HTTP (GET, POST, ...).
        endpoint: Caminho da rota (ex.: ``/predict``).

    Yields:
        Controle para o bloco da requisição; status 200 ou 500 é registrado.
    """
    start = time.perf_counter()
    status = "500"
    try:
        yield
        status = "200"
    except Exception:
        status = "500"
        raise
    finally:
        elapsed = time.perf_counter() - start
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(elapsed)
