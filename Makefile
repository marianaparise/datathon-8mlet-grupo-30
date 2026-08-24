.PHONY: help setup data train api mlflow test lint clean docker-build docker-up docker-down

VENV    := .venv
PY      := $(VENV)/bin/python
PIP     := $(VENV)/bin/pip
RAW     := data/raw/bank-additional-full.csv

help:
	@echo "setup        cria o .venv e instala as dependências"
	@echo "data         baixa a base do Kaggle para data/raw/"
	@echo "train        roda o pipeline ponta a ponta e serializa os artefatos"
	@echo "api          sobe a API local em http://localhost:8000/docs (exige make train)"
	@echo "mlflow       abre a UI do MLflow em http://localhost:5000"
	@echo "test         roda a suíte de testes"
	@echo "lint         checa estilo e erros estáticos com ruff"
	@echo "tf-check     roda terraform fmt e validate em infra/"
	@echo "docker-up    sobe API + MLflow via docker compose"
	@echo "clean        remove caches e artefatos gerados"

setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "OK. Ative com: source $(VENV)/bin/activate"

data:
	@./scripts/download_data.sh

train:
	$(PY) train.py

api:
	@test -f models/environment.joblib || (echo "models/environment.joblib ausente — rode 'make train' antes."; exit 1)
	$(VENV)/bin/uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

mlflow:
	$(VENV)/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

test:
	$(VENV)/bin/pytest tests/ -v

lint:
	$(VENV)/bin/ruff check .

tf-check:
	cd infra && terraform fmt -check -recursive && terraform validate

docker-build:
	@test -f models/environment.joblib || (echo "models/environment.joblib ausente — rode 'make train' antes."; exit 1)
	docker compose build

docker-up:
	docker compose up

docker-down:
	docker compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ipynb_checkpoints
	rm -f models/*.joblib models/*.json
