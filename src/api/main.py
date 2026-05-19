"""API REST FastAPI para servir previsões do modelo LSTM em produção.

Endpoints principais:
    GET  /health          — status e métricas de teste
    POST /predict         — previsão a partir de lista de preços
    POST /predict/symbol  — busca Yahoo Finance + previsão
    GET  /metrics         — métricas Prometheus
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field, field_validator

from src.config import API_HOST, API_PORT, SYMBOL, load_metadata
from src.model.predict import StockPredictor
from src.monitoring.metrics import MODEL_LOADED, PREDICTIONS_TOTAL, track_request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_predictor: StockPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega o modelo na inicialização e libera recursos no shutdown."""
    global _predictor
    try:
        _predictor = StockPredictor(SYMBOL)
        MODEL_LOADED.set(1)
        logger.info("Model loaded for symbol %s", SYMBOL)
    except FileNotFoundError as exc:
        MODEL_LOADED.set(0)
        logger.warning("Model not loaded at startup: %s", exc)
    yield
    MODEL_LOADED.set(0)
    _predictor = None


app = FastAPI(
    title="LSTM Stock Price API",
    description="API REST para previsão de preços de fechamento com LSTM",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    prices: list[float] = Field(..., description="Histórico de preços de fechamento")
    steps: int = Field(1, ge=1, le=30, description="Número de dias à frente")
    symbol: str | None = Field(None, description="Ticker solicitado (rótulo na resposta)")

    @field_validator("prices")
    @classmethod
    def validate_prices(cls, v: list[float]) -> list[float]:
        if len(v) < 1:
            raise ValueError("prices cannot be empty")
        if any(p <= 0 for p in v):
            raise ValueError("all prices must be positive")
        return v


class PredictBySymbolRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=12)
    steps: int = Field(1, ge=1, le=30)
    period: str = Field("1y", description="Período Yahoo Finance: 6mo, 1y, 2y, 5y")


class PredictResponse(BaseModel):
    symbol: str
    model_symbol: str
    predicted_close: list[float]
    lookback_used: int
    latency_ms: float
    warning: str | None = None
    historical_close: list[float] | None = None


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Registra latência e contagem de requisições (exceto /metrics)."""
    if request.url.path == "/metrics":
        return await call_next(request)
    method = request.method
    endpoint = request.url.path
    with track_request(method, endpoint):
        return await call_next(request)


def _symbol_warning(requested: str) -> str | None:
    """Retorna aviso quando o ticker solicitado difere do modelo treinado."""
    if requested.upper() != SYMBOL.upper():
        return (
            f"Modelo treinado para {SYMBOL.upper()}. "
            f"Previsões para {requested.upper()} podem ser imprecisas."
        )
    return None


@app.get("/health")
def health():
    """Verifica status da API, modelo carregado e métricas de teste."""
    meta = load_metadata(SYMBOL)
    return {
        "status": "ok",
        "model_loaded": _predictor is not None,
        "symbol": SYMBOL,
        "model_symbol": SYMBOL,
        "lookback": meta.get("lookback"),
        "test_metrics": meta.get("test_metrics"),
    }


@app.get("/metrics")
def metrics():
    """Expõe métricas no formato Prometheus."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest):
    """Prevê fechamento(s) futuro(s) a partir de histórico de preços enviado."""
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model first.")

    start = time.perf_counter()
    try:
        predictions = _predictor.predict_next(body.prices, steps=body.steps)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    requested = (body.symbol or SYMBOL).upper()
    PREDICTIONS_TOTAL.labels(symbol=requested).inc()
    latency_ms = (time.perf_counter() - start) * 1000

    return PredictResponse(
        symbol=requested,
        model_symbol=SYMBOL.upper(),
        predicted_close=predictions,
        lookback_used=_predictor.lookback,
        latency_ms=round(latency_ms, 2),
        warning=_symbol_warning(requested),
    )


@app.post("/predict/symbol", response_model=PredictResponse)
def predict_by_symbol(body: PredictBySymbolRequest):
    """Baixa cotações no Yahoo Finance e executa o modelo para o ticker."""
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    sym = body.symbol.upper()
    df = yf.download(sym, period=body.period, progress=False, auto_adjust=True)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data for symbol {sym}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    prices = df["Close"].dropna().astype(float).tolist()

    if len(prices) < _predictor.lookback:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {_predictor.lookback} prices for {sym}, got {len(prices)}.",
        )

    start = time.perf_counter()
    try:
        predictions = _predictor.predict_next(prices, steps=body.steps)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    PREDICTIONS_TOTAL.labels(symbol=sym).inc()
    lookback = _predictor.lookback

    return PredictResponse(
        symbol=sym,
        model_symbol=SYMBOL.upper(),
        predicted_close=predictions,
        lookback_used=lookback,
        latency_ms=round((time.perf_counter() - start) * 1000, 2),
        warning=_symbol_warning(sym),
        historical_close=prices[-lookback:],
    )


def main() -> None:
    """Sobe o servidor Uvicorn."""
    import uvicorn

    uvicorn.run("src.api.main:app", host=API_HOST, port=API_PORT, reload=False)


if __name__ == "__main__":
    main()
