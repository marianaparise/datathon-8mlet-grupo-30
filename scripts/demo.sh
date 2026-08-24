#!/usr/bin/env bash
#
# Roteiro da demo do vídeo, executado passo a passo.
#
# Sobe a API, espera ficar saudável, e para a cada passo esperando Enter — para
# você narrar sem correr atrás do terminal. Derruba o que subiu ao sair, mesmo
# se a gravação for interrompida com Ctrl-C.
#
# Narração e saídas esperadas: docs/DEMO.md

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${DEMO_PORT:-8000}"
BASE="http://localhost:${PORT}"
PY="${ROOT}/.venv/bin/python"

BOLD=$'\033[1m'; DIM=$'\033[2m'; CYAN=$'\033[36m'; GREEN=$'\033[32m'
YELLOW=$'\033[33m'; RED=$'\033[31m'; OFF=$'\033[0m'

STARTED_API=""

cleanup() {
    if [ -n "$STARTED_API" ]; then
        printf '\n%s\n' "${DIM}encerrando a API...${OFF}"
        kill "$STARTED_API" 2>/dev/null || true
        wait "$STARTED_API" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

step() {
    printf '\n%s\n' "${CYAN}${BOLD}── $1 ${OFF}"
    [ $# -gt 1 ] && printf '%s\n' "${DIM}$2${OFF}"
    printf '%s' "${DIM}   [Enter para continuar]${OFF}"
    read -r _ || true
    printf '\n'
}

fail() { printf '%s\n' "${RED}$1${OFF}" >&2; exit 1; }

recommend() {
    local body="$1" query="${2:-}"
    curl -s -X POST "${BASE}/recommend${query}" \
        -H 'Content-Type: application/json' -d "$body"
}

show_ranking() {
    "$PY" -c "
import json, sys
d = json.load(sys.stdin)
tie = 'SIM' if d['is_tie'] else 'nao'
print(f\"  recomendado: {d['recommended_arm']}  ({d['probability']:.2%})   empate no topo: {tie}\")
print()
for i, a in enumerate(d['ranking']):
    mark = '  <--' if i == 0 else ''
    print(f\"    {a['arm']:<18s} {a['probability']:>7.2%}{mark}\")
"
}

# --- Clientes do roteiro ----------------------------------------------------

CLIENT_A='{"age":18,"job":"student","marital":"single","education":"high.school",
"default":"no","housing":"no","loan":"no","campaign":1,"pdays":999,
"previous":0,"poutcome":"nonexistent"}'

CLIENT_B='{"age":49,"job":"technician","marital":"married","education":"professional.course",
"default":"no","housing":"yes","loan":"no","campaign":6,"pdays":999,
"previous":0,"poutcome":"nonexistent"}'

CLIENT_C='{"age":35,"job":"management","marital":"married","education":"university.degree",
"default":"no","housing":"no","loan":"no","campaign":1,"pdays":6,
"previous":1,"poutcome":"success"}'

CLIENT_BAD='{"age":49,"job":"astronauta","marital":"married","education":"professional.course",
"default":"no","housing":"yes","loan":"no","campaign":6,"pdays":999,
"previous":0,"poutcome":"nonexistent"}'

# --- Preparação -------------------------------------------------------------

cd "$ROOT"

[ -x "$PY" ] || fail "venv ausente. Rode: make setup"
[ -f models/environment.joblib ] || fail "models/environment.joblib ausente. Rode: make train"

if curl -sf "${BASE}/health" >/dev/null 2>&1; then
    printf '%s\n' "${GREEN}API já está no ar em ${BASE}${OFF}"
else
    printf '%s\n' "${DIM}subindo a API em ${BASE}...${OFF}"
    "${ROOT}/.venv/bin/uvicorn" api.app:app --host 127.0.0.1 --port "$PORT" \
        >/tmp/tc5-demo.log 2>&1 &
    STARTED_API=$!

    for _ in $(seq 1 30); do
        curl -sf "${BASE}/health" >/dev/null 2>&1 && break
        sleep 1
    done
    curl -sf "${BASE}/health" >/dev/null 2>&1 \
        || fail "a API não respondeu. Veja /tmp/tc5-demo.log"
    printf '%s\n' "${GREEN}no ar${OFF}"
fi

printf '\n%s\n' "${BOLD}Demo TC5 — plataforma de experimentação adaptativa${OFF}"
printf '%s\n' "${DIM}Narração e saídas esperadas em docs/DEMO.md${OFF}"

# --- Passos -----------------------------------------------------------------

step "1. O serviço está de pé" \
     "Seis braços: dois canais x tres janelas de contato."
curl -s "${BASE}/health" | "$PY" -m json.tool

step "2. Cliente A — estudante, 18 anos" \
     "A API devolve o ranking inteiro, nao so o vencedor."
recommend "$CLIENT_A" | show_ranking

step "3. Cliente B — tecnico, 49 anos, ja com 6 ligacoes" \
     "Mesma API, cliente diferente. O ranking vira."
recommend "$CLIENT_B" | show_ranking
printf '\n%s\n' "${YELLOW}   telephone|early assume a lideranca. cellular|mid, o melhor${OFF}"
printf '%s\n'   "${YELLOW}   braco na media da base, cai para quarto lugar.${OFF}"

step "4. Cliente C — ja converteu em campanha anterior" \
     "O sinal mais forte da base: 72% contra 11% da media."
recommend "$CLIENT_C" | show_ranking
printf '\n%s\n' "${YELLOW}   is_tie = SIM: o sistema admite que aqui tanto faz,${OFF}"
printf '%s\n'   "${YELLOW}   e quem decide deveria ser o custo do canal.${OFF}"

step "5. O contrato recusa lixo" \
     "Categoria desconhecida e barrada na porta, com 422."
recommend "$CLIENT_BAD" | "$PY" -c "
import json, sys
d = json.load(sys.stdin)
e = d['detail'][0]
print(f\"  HTTP 422 — campo '{e['loc'][-1]}'\")
print(f\"  {e['msg'][:100]}...\")
"

step "6. Exploracao x explotacao" \
     "Mesmo cliente, tres chamadas. Em 5% delas o sistema nao joga o melhor braco."
for seed in 3 53 65; do
    # A seed vai por argv, não interpolada: `${seed:<3}` dentro de aspas duplas
    # é expansão de parâmetro do bash, não format spec do Python.
    recommend "$CLIENT_A" "?explore=true&seed=${seed}" | "$PY" -c '
import json, sys
seed = sys.argv[1]
d = json.load(sys.stdin)
arm = d["recommended_arm"]
probability = d["probability"]
tag = "EXPLOROU" if d["explored"] else "explotou"
print(f"    seed={seed:<3}  {arm:<18s} {probability:>7.2%}   {tag}")
' "$seed"
done
printf '\n%s\n' "${YELLOW}   Parece desperdicio, e e — a curto prazo. E o preco de${OFF}"
printf '%s\n'   "${YELLOW}   continuar aprendendo, que um teste A/B congelado nao paga.${OFF}"

step "7. Swagger" \
     "Abra ${BASE}/docs — repare que 'duration' nao existe no formulario."
printf '%s\n' "  ${BASE}/docs"

printf '\n%s\n' "${GREEN}${BOLD}Fim do roteiro.${OFF}"
printf '%s\n' "${DIM}Frase de fechamento sugerida em docs/DEMO.md${OFF}"
