"""Dashboard Streamlit para consumir a API de previsão LSTM.

Funções principais:
    fetch_health          — status da API
    call_predict_symbol   — previsão automática via Yahoo Finance
    call_predict_prices   — previsão com histórico manual
    show_prediction_result — exibe resultado, avisos e gráfico
"""

from __future__ import annotations

import os

import plotly.graph_objects as go
import requests
import streamlit as st

def _resolve_api_url() -> str:
    """Normaliza API_URL (aceita host:port da rede privada Render)."""
    raw = os.getenv("API_URL", "http://localhost:8000").strip()
    if not raw.startswith(("http://", "https://")):
        return f"http://{raw}".rstrip("/")
    return raw.rstrip("/")


API_URL = _resolve_api_url()
DEFAULT_SYMBOL = os.getenv("SYMBOL", "DIS")


def fetch_health() -> dict | None:
    """Consulta GET /health e retorna JSON ou None se a API estiver offline."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def call_predict_symbol(symbol: str, steps: int = 1, period: str = "1y") -> dict | None:
    """Chama POST /predict/symbol — API busca cotações e retorna previsão."""
    try:
        r = requests.post(
            f"{API_URL}/predict/symbol",
            json={"symbol": symbol, "steps": steps, "period": period},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()
    except requests.RequestException as exc:
        if hasattr(exc, "response") and exc.response is not None:
            try:
                detail = exc.response.json().get("detail", str(exc))
            except Exception:
                detail = str(exc)
            st.error(f"Erro na API: {detail}")
        else:
            st.error(f"Erro na API: {exc}")
        return None


def call_predict_prices(
    prices: list[float], steps: int = 1, symbol: str | None = None
) -> dict | None:
    """Chama POST /predict com lista de preços informada pelo usuário."""
    try:
        payload: dict = {"prices": prices, "steps": steps}
        if symbol:
            payload["symbol"] = symbol.upper()
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as exc:
        if hasattr(exc, "response") and exc.response is not None:
            try:
                detail = exc.response.json().get("detail", str(exc))
            except Exception:
                detail = str(exc)
            st.error(f"Erro na API: {detail}")
        else:
            st.error(f"Erro na API: {exc}")
        return None


def _example_prices_text(lookback: int) -> str:
    """Gera lookback preços de exemplo para teste da previsão manual."""
    base = 150.0
    return ", ".join(f"{base + i * 0.1:.1f}" for i in range(lookback))


def show_prediction_result(result: dict) -> None:
    """Renderiza previsão, avisos, gráfico histórico e JSON da resposta."""
    preds = result["predicted_close"]
    sym = result["symbol"]
    model_sym = result.get("model_symbol", sym)

    st.success(
        f"Previsão(ões) para **{sym}**: "
        + ", ".join(f"${p:.2f}" for p in preds)
    )

    if result.get("warning"):
        st.warning(result["warning"])

    if model_sym != sym:
        st.info(f"Modelo em produção treinado para: **{model_sym}**")

    hist = result.get("historical_close")
    if hist:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                y=hist,
                mode="lines",
                name=f"Últimos {len(hist)} fechamentos ({sym})",
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    st.json(result)


st.set_page_config(page_title="LSTM Stock Predictor", page_icon="📈", layout="wide")
st.title("📈 Previsão de Preços — LSTM")
st.caption("Dashboard consumindo a API FastAPI — busca de cotações feita pela API")

health = fetch_health()
trained_symbol = (health or {}).get("model_symbol") or (health or {}).get("symbol") or DEFAULT_SYMBOL

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("API", "Online" if health else "Offline")
with col2:
    st.metric("Modelo", "Carregado" if health and health.get("model_loaded") else "Não carregado")
with col3:
    st.metric("Treinado para", trained_symbol)
with col4:
    metrics = (health or {}).get("test_metrics") or {}
    st.metric("MAPE (teste)", f"{metrics.get('mape', 0):.2f}%" if metrics else "—")

st.sidebar.header("Configuração")
symbol = st.sidebar.text_input("Símbolo (Yahoo Finance)", value=DEFAULT_SYMBOL).strip().upper()
steps = st.sidebar.slider("Dias à prever", min_value=1, max_value=10, value=1)
period = st.sidebar.selectbox("Período histórico", ["6mo", "1y", "2y", "5y"], index=1)

if symbol and trained_symbol and symbol != trained_symbol:
    st.sidebar.warning(
        f"O modelo foi treinado para **{trained_symbol}**. "
        f"Outros tickers usam o mesmo modelo (resultado apenas ilustrativo)."
    )

tab1, tab2 = st.tabs(["Previsão automática (via API)", "Previsão manual"])

with tab1:
    st.subheader(f"Histórico e previsão — {symbol}")
    st.caption("A API busca os dados no Yahoo Finance e executa o modelo.")
    if st.button("Buscar dados e prever", type="primary"):
        with st.spinner(f"Consultando API para {symbol}..."):
            result = call_predict_symbol(symbol, steps=steps, period=period)
            if result:
                show_prediction_result(result)

with tab2:
    st.subheader("Enviar histórico manualmente")
    lookback = int((health or {}).get("lookback") or 60)
    st.caption(
        f"A previsão será rotulada como **{symbol}** na resposta da API. "
        f"O modelo exige **pelo menos {lookback}** preços de fechamento (lookback)."
    )
    if "manual_prices_text" not in st.session_state:
        st.session_state.manual_prices_text = _example_prices_text(lookback)

    col_fill, col_clear = st.columns(2)
    with col_fill:
        if st.button(f"Preencher exemplo ({lookback} preços)"):
            st.session_state.manual_prices_text = _example_prices_text(lookback)
    with col_clear:
        if st.button("Limpar"):
            st.session_state.manual_prices_text = ""

    raw = st.text_area(
        "Preços de fechamento (separados por vírgula)",
        height=120,
        help=f"Mínimo de {lookback} valores positivos, sem NaN.",
        key="manual_prices_text",
    )
    if st.button("Prever com histórico manual", type="primary"):
        try:
            prices = [float(x.strip()) for x in raw.split(",") if x.strip()]
        except ValueError:
            st.error("Valores inválidos. Use números separados por vírgula.")
        else:
            if len(prices) < lookback:
                st.error(
                    f"Foram informados **{len(prices)}** preços, mas o modelo precisa de "
                    f"**pelo menos {lookback}**. Use o botão acima para preencher um exemplo "
                    "ou cole mais valores."
                )
            elif any(p <= 0 for p in prices):
                st.error("Todos os preços devem ser números positivos.")
            else:
                result = call_predict_prices(prices, steps=steps, symbol=symbol)
                if result:
                    show_prediction_result(result)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**API:** `{API_URL}`")
if health:
    st.sidebar.json(health)
