.PHONY: help setup data train api mlflow test clean docker-build docker-up docker-down

VENV    := .venv
PY      := $(VENV)/bin/python
PIP     := $(VENV)/bin/pip
RAW     := data/raw/bank-additional-full.csv

help:
	@echo "setup        cria o .venv e instala as dependências"
	@echo "data         baixa a base do Kaggle para data/raw/"
	@echo "train        roda o pipeline ponta a ponta e serializa os artefatos"
	@echo "api          sobe a API local em http://localhost:8000/docs"
	@echo "mlflow       abre a UI do MLflow em http://localhost:5000"
	@echo "test         roda a suíte de testes"
	@echo "docker-up    sobe API + MLflow via docker compose"
	@echo "clean        remove caches e artefatos gerados"

setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "OK. Ative com: source $(VENV)/bin/activate"

data:
	@if [ -f "$(RAW)" ]; then \
		echo "$(RAW) já existe."; \
	else \
		echo "Baixando de kaggle.com/datasets/henriqueyamahata/bank-marketing ..."; \
		$(VENV)/bin/kaggle datasets download -d henriqueyamahata/bank-marketing -p data/raw --unzip \
		  || (echo ""; \
		      echo "Falhou. Requer ~/.kaggle/kaggle.json (Kaggle > Settings > Create New Token)."; \
		      echo "Alternativa manual: baixe o zip do link acima e extraia em data/raw/"; \
		      exit 1); \
	fi

train:
	$(PY) train.py

api:
	$(VENV)/bin/uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

mlflow:
	$(VENV)/bin/mlflow ui --port 5000

test:
	$(VENV)/bin/pytest tests/ -v

docker-build:
	docker compose build

docker-up:
	docker compose up

docker-down:
	docker compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ipynb_checkpoints
	rm -f models/*.joblib models/*.json
