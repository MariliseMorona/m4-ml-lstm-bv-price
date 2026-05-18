.PHONY: install download preprocess train evaluate api dashboard test docker-up

install:
	python -m venv .venv
	.venv/bin/pip install -r requirements.txt

download:
	python -m src.data.download

preprocess:
	python -m src.data.preprocess

train:
	python -m src.model.train

evaluate:
	python -m src.model.evaluate

api:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

dashboard:
	streamlit run dashboard/app.py

test:
	pytest tests/ -v

pipeline: download preprocess train evaluate

docker-up:
	docker compose up --build
