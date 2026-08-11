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

# Indicadores macroeconômicos: iguais para todo cliente contatado no mesmo período.
# Não personalizam nada — movem a taxa-base. Isolados do resto do contexto para que a
# Fase 2 possa medir quanto do acerto vem do calendário e quanto vem do perfil.
MACRO_COLUMNS: tuple[str, ...] = (
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
)

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

# O contexto sem os indicadores macro: só o que descreve a pessoa.
CLIENT_COLUMNS: tuple[str, ...] = tuple(
    c for c in CONTEXT_COLUMNS if c not in MACRO_COLUMNS
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

# Valor que o dataset usa para "não coletado". Não é imputado: no bank-marketing
# ele carrega informação (quem não respondeu converte diferente de quem respondeu).
UNKNOWN_TOKEN = "unknown"

# `pdays` usa 999 para "nunca contatado antes". É sentinela, não distância: a
# média da coluna crua é ficção, e um modelo que a trate como número lê 999 dias
# como "contato muito antigo" em vez de "primeiro contato".
PDAYS_SENTINEL = 999

# --- Colunas derivadas ---------------------------------------------------

WEEK_WINDOW_COLUMN = "week_window"
ARM_COLUMN = "arm"
TARGET_BINARY = "converted"
FIRST_CONTACT_COLUMN = "first_contact"

# `day_of_week` tem 5 valores; cruzado com `contact` daria 10 células, e cruzado
# também com `month` daria 100. Agregar em três janelas mantém a semântica de
# "quando abordar" com suporte amostral por célula.
WEEK_WINDOWS: dict[str, str] = {
    "mon": "early",
    "tue": "mid",
    "wed": "mid",
    "thu": "mid",
    "fri": "late",
}

# --- Espaço de braços ----------------------------------------------------

# Pisos por braço. 1.000 eventos dão erro-padrão de ~1 p.p. numa CVR de 11%;
# 100 conversões é o piso para a taxa não oscilar com um punhado de casos.
MIN_EVENTS_PER_ARM = 1_000
MIN_CONVERSIONS_PER_ARM = 100

# Avaliados do mais grosso ao mais fino. `month` fica de fora: é o confundidor
# temporal principal, e cruzá-lo esvaziaria as células.
ARM_SPACE_CANDIDATES: tuple[tuple[str, ...], ...] = (
    ("contact",),
    ("contact", WEEK_WINDOW_COLUMN),
    ("contact", "day_of_week"),
)

# Espaço definitivo, escolhido na Fase 1 pelo suporte observado — ver README.
ARM_COLUMNS: tuple[str, ...] = ("contact", WEEK_WINDOW_COLUMN)

# --- Reprodutibilidade ---------------------------------------------------

SEED = 42
TEST_SIZE = 0.2
