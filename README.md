# m4-ml-lstm-bv-price

Projeto end-to-end de previsão de preços de fechamento de ações com **LSTM**, incluindo coleta de dados (Yahoo Finance), treinamento, API REST (FastAPI), monitoramento (Prometheus/Grafana) e dashboard (Streamlit).

## Estrutura do projeto

```
├── data/raw/              # CSV baixado do Yahoo Finance
├── data/processed/        # Sequências para o LSTM (.npz)
├── models/                # Modelo .pt, scaler .pkl, metadata .json
├── reports/               # Métricas e gráficos de avaliação
├── src/
│   ├── data/              # download + preprocess
│   ├── model/             # LSTM, train, evaluate, predict
│   ├── api/               # FastAPI
│   └── monitoring/        # Métricas Prometheus
├── dashboard/             # Streamlit
├── tests/
├── docker-compose.yml     # API + Prometheus + Grafana
└── requirements.txt
```

## Pré-requisitos

- Python 3.10+
- (Opcional) Docker e Docker Compose

## Configuração rápida

```bash
# 1. Clonar e entrar no projeto
cd m4-ml-lstm-bv-price

# 2. Ambiente virtual (Python 3.11+ recomendado; modelo usa PyTorch)
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Dependências
pip install -r requirements.txt

# Confirme que o Python do venv é 3.11:
python --version   # deve mostrar Python 3.11.x

# 4. Variáveis de ambiente
cp .env.example .env
# Edite .env se quiser outro ticker (ex.: PETR4.SA, AAPL)
```

## Pipeline ML (executar nesta ordem)

```bash
# Coleta de dados (yfinance)
python -m src.data.download

# Pré-processamento (normalização + sequências temporais)
python -m src.data.preprocess

# Treinamento LSTM (pode levar alguns minutos)
python -m src.model.train

# Avaliação no conjunto de teste (MAE, RMSE, MAPE)
python -m src.model.evaluate
```

Ou use o Makefile:

```bash
make pipeline    # download + preprocess + train + evaluate
```

Artefatos gerados:

| Arquivo | Descrição |
|---------|-----------|
| `data/raw/{SYMBOL}_historical.csv` | Dados brutos |
| `data/processed/{SYMBOL}_sequences.npz` | X/y train, val, test |
| `models/lstm_{SYMBOL}.pt` | Modelo treinado (PyTorch) |
| `models/scaler_{SYMBOL}.pkl` | MinMaxScaler |
| `models/metadata_{SYMBOL}.json` | Hiperparâmetros e métricas |
| `reports/{SYMBOL}_predictions.png` | Gráfico real vs previsto |

## API REST

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
# ou: make api
```

Documentação interativa: http://localhost:8000/docs

### Endpoints

| Método | Rota | Descrição |
|--------|------|-----------|
| `GET` | `/health` | Status da API e métricas de teste |
| `POST` | `/predict` | Previsão a partir de lista de preços |
| `POST` | `/predict/symbol` | Busca histórico no Yahoo e prevê |
| `GET` | `/metrics` | Métricas Prometheus |

### Exemplo — previsão manual

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"prices": [/* últimos 60+ preços */], "steps": 1}'
```

## Dashboard Streamlit

Com a API rodando:

```bash
streamlit run dashboard/app.py
# ou: make dashboard
```

Acesse: http://localhost:8501

## Monitoramento (Docker)

Treine o modelo localmente antes (`models/` deve existir). Depois:

```bash
docker compose up --build
```

| Serviço | URL |
|---------|-----|
| API | http://localhost:8000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin/admin) |

No Grafana, adicione Prometheus como data source (`http://prometheus:9090`) e crie painéis para:

- `http_request_duration_seconds` — latência
- `http_requests_total` — volume de requisições
- `predictions_total` — previsões servidas
- `model_loaded` — status do modelo

## Testes

```bash
pytest tests/ -v
# ou: make test
```

## Variáveis de ambiente (.env)

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `SYMBOL` | `DIS` | Ticker Yahoo Finance |
| `START_DATE` | `2018-01-01` | Início do histórico |
| `END_DATE` | `2024-07-20` | Fim do histórico |
| `LOOKBACK` | `60` | Janela temporal (dias) |
| `LSTM_UNITS` | `50` | Unidades LSTM |
| `EPOCHS` | `100` | Épocas máximas |
| `BATCH_SIZE` | `32` | Tamanho do batch |

## Checklist do Tech Challenge

- [x] Coleta com yfinance
- [x] Pré-processamento com split temporal e scaler no treino
- [x] Modelo LSTM com early stopping
- [x] Métricas MAE, RMSE, MAPE
- [x] Salvamento do modelo e scaler
- [x] API REST FastAPI
- [x] Monitoramento Prometheus
- [x] Dashboard Streamlit

## Problemas no Mac (AVX / TensorFlow)

Se aparecer `TensorFlow was compiled to use AVX instructions`, o projeto usa **PyTorch** (sem exigência AVX). Reinstale as dependências:

```bash
pip uninstall tensorflow tensorflow-io-gcs-filesystem -y
pip install -r requirements.txt
python -c "import torch; print(torch.__version__)"
```

## Aviso

Este projeto é **educacional**. Previsões de mercado financeiro são incertas; não use como recomendação de investimento.
