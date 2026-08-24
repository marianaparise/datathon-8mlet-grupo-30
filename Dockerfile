# Imagem de serving da Etapa 5.
#
# Multi-stage: as rodas são compiladas no primeiro estágio e só o venv pronto
# atravessa para o segundo, então as ferramentas de build não sobram na imagem
# final. Instala requirements-api.txt, não requirements.txt — MLflow, matplotlib
# e jupyter não têm o que fazer num container que só pontua clientes.

FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements-api.txt .
RUN pip install --upgrade pip && pip install -r requirements-api.txt


FROM python:3.12-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Usuário sem privilégios: o processo não tem motivo para rodar como root.
RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY src/ src/
COPY api/ api/

# O artefato vem de `make train` e é gitignored, então precisa existir no
# contexto de build. Sem ele a API sobe e reporta `degraded` no /health, em vez
# de entrar em loop de reinício.
COPY models/ models/

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; \
    sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status == 200 else 1)"

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
