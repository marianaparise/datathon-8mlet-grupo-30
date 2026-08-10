"""Central configuration: paths, schema and reproducibility constants.

Pure constants only — importing this module must have no side effects.
"""

from __future__ import annotations

from pathlib import Path

# --- Paths ---------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

RAW_CSV = DATA_RAW / "bank-additional-full.csv"

# --- Formato do arquivo bruto --------------------------------------------

# O bank-additional-full.csv do UCI usa ponto e vírgula, não vírgula. Ler com o
# separador errado devolve um DataFrame de uma única coluna sem levantar erro —
# por isso o separador é constante e `load_raw` valida o formato depois de ler.
RAW_SEPARATOR = ";"

RAW_N_ROWS = 41_188
RAW_SHA256 = "74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8"

# --- Schema --------------------------------------------------------------

TARGET = "y"
TARGET_POSITIVE = "yes"

# Proibida pelo enunciado: só é conhecida depois do desfecho da ligação.
# Nunca entra em feature set, ambiente ou API.
FORBIDDEN_COLUMNS: tuple[str, ...] = ("duration",)

# Decisões da campanha, não atributos do cliente. É daqui que saem os braços.
ACTION_COLUMNS: tuple[str, ...] = ("contact", "month", "day_of_week")

# Atributos do cliente e conjuntura: o contexto visto pela política.
CONTEXT_COLUMNS: tuple[str, ...] = (
    "age",
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
)

# Ordem exata das colunas no arquivo bruto, usada para validar a leitura.
RAW_COLUMNS: tuple[str, ...] = (
    "age",
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
    "y",
)

# --- Reprodutibilidade ---------------------------------------------------

SEED = 42
