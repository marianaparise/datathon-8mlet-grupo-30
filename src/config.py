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

# --- Ambiente calibrado (Fase 2) -----------------------------------------

# Um modelo único com o braço como feature, não um por braço: o menor braço tem
# ~111 conversões no treino, pouco para calibrar isoladamente.
ENV_MAX_ITER = 300
ENV_LEARNING_RATE = 0.05
ENV_CALIBRATION_METHOD = "isotonic"
ENV_CALIBRATION_CV = 5

# Piso de qualidade do ambiente, em **Brier Skill Score**: 1 - brier/brier_ref,
# onde brier_ref = p(1-p) é o Brier de quem chuta a taxa-base para todo mundo.
#
# Relativo, não absoluto. Um piso absoluto de Brier não transfere entre recortes
# com taxa-base diferente: 0,10 é bom numa base que converte 11% e trivialmente
# atingível numa que converte 3%. Skill negativo significa que o modelo é pior
# que chutar a média — aí sim o ambiente não presta.
MIN_BRIER_SKILL = 0.05

# Sobreposição (positividade): abaixo desta propensão o braço praticamente não
# foi jogado naquela região do contexto, e prever ali é extrapolação.
MIN_ARM_PROPENSITY = 0.01

ENVIRONMENT_ARTIFACT = MODELS_DIR / "environment.joblib"

# --- Políticas de bandit (Fase 3) ----------------------------------------

# Valores escolhidos por sweep sobre o ambiente calibrado — ver README.
# O padrão de livro-texto do UCB1 é c=1.0, mas com taxa-base de 11% o bônus
# sqrt(2 ln t / n) domina médias pequenas e a política explora sem parar.
EPSILON = 0.05
UCB_C = 0.25

# Prior não-informativo: Beta(1,1) é a uniforme em [0,1].
TS_ALPHA_PRIOR = 1.0
TS_BETA_PRIOR = 1.0

# Prior informativo: mesma média da taxa-base observada (11,27%) com a força de
# 10 observações. Entra como variante para a análise de escolha de prior.
TS_ALPHA_PRIOR_INFORMED = 1.13
TS_BETA_PRIOR_INFORMED = 8.87

# Escala da exploração no LinTS, também por sweep. Com 41 features x 6 braços são
# 246 parâmetros a estimar sobre recompensa binária rara, então a posterior
# precisa ser estreita para a política sair da exploração dentro do horizonte.
LINTS_V = 0.05
LINTS_LAMBDA = 1.0

# --- Experimento ---------------------------------------------------------

N_ROUNDS = 20_000
N_SEEDS = 10

MLFLOW_EXPERIMENT = "tc5-bandit"

# SQLite, não o file store: a partir do MLflow 3 o backend de arquivos está em
# modo de manutenção e levanta exceção. `mlflow ui` precisa do mesmo URI.
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB}"
