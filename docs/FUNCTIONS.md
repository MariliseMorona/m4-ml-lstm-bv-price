# Referência de funções e classes

Documentação das funções, classes e endpoints do projeto **m4-ml-lstm-bv-price**.

---

## `src/config.py`

Configuração central carregada de variáveis de ambiente (`.env`).

| Nome | Tipo | Descrição |
|------|------|-----------|
| `SYMBOL` | `str` | Ticker Yahoo Finance (padrão: `DIS`) |
| `START_DATE` / `END_DATE` | `str` | Intervalo de coleta |
| `LOOKBACK` | `int` | Janela temporal em dias (padrão: 60) |
| `LSTM_UNITS` | `int` | Neurônios por camada LSTM |
| `EPOCHS` / `BATCH_SIZE` | `int` | Hiperparâmetros de treino |
| `TRAIN_RATIO` / `VAL_RATIO` / `TEST_RATIO` | `float` | Proporções do split temporal |

### Funções

#### `ensure_dirs() -> None`
Cria os diretórios `data/raw`, `data/processed`, `models` e `reports` se não existirem.

#### `raw_csv_path(symbol=None) -> Path`
Retorna o caminho do CSV bruto: `data/raw/{SYMBOL}_historical.csv`.

#### `processed_npz_path(symbol=None) -> Path`
Retorna o caminho das sequências processadas: `data/processed/{SYMBOL}_sequences.npz`.

#### `model_path(symbol=None) -> Path`
Retorna o caminho do checkpoint PyTorch: `models/lstm_{SYMBOL}.pt`.

#### `scaler_path(symbol=None) -> Path`
Retorna o caminho do `MinMaxScaler`: `models/scaler_{SYMBOL}.pkl`.

#### `metadata_path(symbol=None) -> Path`
Retorna o caminho dos metadados JSON: `models/metadata_{SYMBOL}.json`.

#### `save_metadata(data: dict, symbol=None) -> Path`
Persiste hiperparâmetros, métricas e data do treino em JSON.

#### `load_metadata(symbol=None) -> dict`
Carrega metadados do disco; retorna `{}` se o arquivo não existir.

---

## `src/data/download.py`

Coleta de preços históricos via **yfinance**.

#### `flatten_columns(df) -> pd.DataFrame`
Achata colunas `MultiIndex` retornadas pelo yfinance em nomes simples (`Close`, `Open`, etc.).

#### `download(symbol=None, start=None, end=None) -> pd.DataFrame`
- Baixa OHLCV do ticker no Yahoo Finance.
- Limpa duplicatas, ordena por data.
- Salva CSV em `data/raw/`.
- **Raises:** `ValueError` se não houver dados.

#### `main() -> None`
Ponto de entrada CLI: `python -m src.data.download`.

---

## `src/data/preprocess.py`

Pré-processamento para sequências LSTM.

#### `load_raw(symbol=None) -> pd.DataFrame`
Lê o CSV bruto gerado pelo `download`. **Raises:** `FileNotFoundError` se ausente.

#### `build_sequences(prices, lookback) -> tuple[np.ndarray, np.ndarray]`
Constrói pares `(X, y)` deslizantes:
- `X[i]`: preços normalizados dos dias `[i-lookback : i]`
- `y[i]`: preço do dia `i`

#### `temporal_split(n_samples, train_ratio, val_ratio, test_ratio) -> tuple[slice, slice, slice]`
Divide amostras em ordem cronológica (treino → validação → teste). **Raises:** `ValueError` se as proporções não somarem 1.

#### `preprocess(symbol=None, lookback=None) -> dict`
Pipeline completo:
1. Carrega CSV bruto.
2. Ajusta `MinMaxScaler` **somente no treino**.
3. Gera sequências e faz split temporal.
4. Salva `.npz` e `scaler.pkl`.

Retorna dicionário com `X_train`, `y_train`, `X_val`, `y_val`, `X_test`, `y_test`.

#### `main() -> None`
Ponto de entrada CLI: `python -m src.data.preprocess`.

---

## `src/model/lstm.py`

Definição da rede neural.

### Classe `LSTMStockPredictor(nn.Module)`

| Método | Descrição |
|--------|-----------|
| `__init__(lookback, units=None, dropout=None)` | Duas camadas LSTM + dropout + camada linear de saída |
| `forward(x)` | Entrada `(batch, seq_len, 1)` → saída `(batch, 1)` |

#### `build_lstm_model(lookback, units=None, dropout=None) -> LSTMStockPredictor`
Factory que instancia o modelo com hiperparâmetros do `.env`.

---

## `src/model/io.py`

Persistência e dispositivo de execução.

#### `get_device() -> torch.device`
Seleciona `cuda`, `mps` (Apple Silicon) ou `cpu`.

#### `save_checkpoint(path, model, lookback, lstm_units, dropout) -> None`
Salva `state_dict`, hiperparâmetros e framework no arquivo `.pt`.

#### `load_model(path, device=None) -> LSTMStockPredictor`
Carrega checkpoint, reconstrói a arquitetura e coloca em modo `eval()`.

---

## `src/model/metrics.py`

Métricas de avaliação em escala original (após desnormalização).

| Função | Fórmula / descrição |
|--------|---------------------|
| `mae(y_true, y_pred)` | Erro absoluto médio |
| `rmse(y_true, y_pred)` | Raiz do erro quadrático médio |
| `mape(y_true, y_pred, epsilon=1e-8)` | Erro percentual absoluto médio (%) |
| `compute_all(y_true, y_pred)` | Retorna `{"mae", "rmse", "mape"}` |

---

## `src/model/train.py`

Treinamento do modelo.

#### `load_processed(symbol=None) -> dict`
Carrega arrays do `.npz` processado.

#### `_make_loader(X, y, shuffle) -> DataLoader`
Monta `DataLoader` PyTorch a partir de arrays NumPy.

#### `_run_epoch(model, loader, criterion, optimizer, device) -> float`
Executa uma época de treino ou validação; retorna loss média.

#### `train(symbol=None) -> dict`
Loop de treino com:
- Adam + MSE
- `ReduceLROnPlateau`
- Early stopping (patience 15)
- Checkpoint do melhor modelo

Retorna `{"history", "metadata"}`.

#### `main() -> None`
CLI: `python -m src.model.train`.

---

## `src/model/evaluate.py`

Avaliação no conjunto de teste.

#### `evaluate(symbol=None) -> dict`
- Carrega modelo e scaler.
- Prevê `X_test` e desnormaliza.
- Calcula MAE, RMSE, MAPE.
- Atualiza `metadata_{SYMBOL}.json` e salva gráfico PNG.

#### `_plot_predictions(symbol, y_true, y_pred, n=120) -> None`
Gera gráfico real vs previsto em `reports/{SYMBOL}_predictions.png`.

#### `main() -> None`
CLI: `python -m src.model.evaluate`.

---

## `src/model/predict.py`

Inferência em produção.

### Classe `StockPredictor`

| Método | Descrição |
|--------|-----------|
| `__init__(symbol, models_dir=None)` | Carrega modelo `.pt`, scaler e metadados |
| `predict_next(prices, steps=1)` | Prevê 1 ou mais dias à frente (recursivo) |

**Raises:** `FileNotFoundError`, `ValueError` (preços insuficientes, NaN ou ≤ 0).

#### `load_predictor(symbol) -> StockPredictor`
Atalho para instanciar o preditor.

---

## `src/api/main.py`

API REST **FastAPI**.

### Ciclo de vida

| Função | Descrição |
|--------|-----------|
| `lifespan(app)` | Carrega `StockPredictor` na subida; libera na parada |

### Modelos Pydantic

| Classe | Campos principais |
|--------|-------------------|
| `PredictRequest` | `prices`, `steps`, `symbol?` |
| `PredictBySymbolRequest` | `symbol`, `steps`, `period` |
| `PredictResponse` | `symbol`, `model_symbol`, `predicted_close`, `warning?`, `historical_close?` |

### Endpoints

| Rota | Função | Descrição |
|------|--------|-----------|
| `GET /health` | `health()` | Status da API, modelo e métricas de teste |
| `GET /metrics` | `metrics()` | Métricas Prometheus |
| `POST /predict` | `predict(body)` | Previsão a partir de lista de preços |
| `POST /predict/symbol` | `predict_by_symbol(body)` | Busca Yahoo Finance + previsão |

#### `_symbol_warning(requested) -> str | None`
Retorna aviso quando o ticker solicitado difere do modelo treinado.

#### `metrics_middleware(request, call_next)`
Middleware que registra latência e contagem de requisições.

#### `main() -> None`
Sobe o servidor Uvicorn.

---

## `src/monitoring/metrics.py`

Métricas **Prometheus**.

| Métrica | Tipo | Descrição |
|---------|------|-----------|
| `http_requests_total` | Counter | Total de requisições por método/rota/status |
| `http_request_duration_seconds` | Histogram | Latência por rota |
| `predictions_total` | Counter | Previsões servidas por símbolo |
| `model_loaded` | Gauge | 1 se o modelo está carregado |

#### `track_request(method, endpoint)`
Context manager que incrementa contadores e observa latência ao final da requisição.

---

## `dashboard/app.py`

Interface **Streamlit**.

| Função | Descrição |
|--------|-----------|
| `fetch_health()` | `GET /health` — status da API |
| `call_predict_symbol(symbol, steps, period)` | `POST /predict/symbol` |
| `call_predict_prices(prices, steps, symbol)` | `POST /predict` |
| `show_prediction_result(result)` | Exibe previsão, avisos e gráfico Plotly |

Variáveis de ambiente: `API_URL` (padrão `http://localhost:8000`), `SYMBOL`.

---

## `tests/`

| Arquivo | Cobertura |
|---------|-----------|
| `test_metrics.py` | MAE, RMSE, MAPE com previsão perfeita e com erro |
| `test_api.py` | `/health`, `/metrics`, validação de `/predict` |
