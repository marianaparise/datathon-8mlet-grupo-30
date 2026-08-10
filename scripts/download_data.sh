#!/usr/bin/env bash
#
# Baixa o bank-additional-full.csv para data/raw/.
#
# Tenta o Kaggle primeiro (fonte citada no enunciado) e cai no UCI, que é a origem
# real do dataset e não exige credencial. Seja qual for a via, o arquivo é conferido
# por SHA-256 — duas fontes só servem se entregarem exatamente o mesmo arquivo.
#
# Uso: ./scripts/download_data.sh   (ou `make data`)

set -euo pipefail

RAW_DIR="data/raw"
TARGET="${RAW_DIR}/bank-additional-full.csv"
NAMES="${RAW_DIR}/bank-additional-names.txt"

# SHA-256 do bank-additional-full.csv obtido do UCI em 2026-08-10.
EXPECTED_SHA256="74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8"

UCI_URL="https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
KAGGLE_DATASET="henriqueyamahata/bank-marketing"

log()  { printf '  %s\n' "$*"; }
fail() { printf 'ERRO: %s\n' "$*" >&2; exit 1; }

# Global: o trap de EXIT roda fora do escopo de main(), então não pode ser `local`.
TMP_DIR=""
cleanup() {
    if [ -n "$TMP_DIR" ] && [ -d "$TMP_DIR" ]; then
        rm -rf "$TMP_DIR"
    fi
}
trap cleanup EXIT

checksum_ok() {
    [ -f "$1" ] || return 1
    echo "${EXPECTED_SHA256}  $1" | sha256sum --check --status 2>/dev/null
}

# Localiza o CSV dentro de um diretório e o move para data/raw/.
harvest() {
    local src_dir="$1" found
    found="$(find "$src_dir" -name 'bank-additional-full.csv' -not -path '*__MACOSX*' -print -quit)"
    [ -n "$found" ] || return 1
    mkdir -p "$RAW_DIR"
    mv "$found" "$TARGET"

    # O .txt com a descrição oficial das colunas vem junto quando disponível.
    local names
    names="$(find "$src_dir" -name 'bank-additional-names.txt' -not -path '*__MACOSX*' -print -quit)"
    [ -n "$names" ] && mv "$names" "$NAMES"
    return 0
}

try_kaggle() {
    local kaggle_bin="" tmp="$1"
    if   [ -x ".venv/bin/kaggle" ]; then kaggle_bin=".venv/bin/kaggle"
    elif command -v kaggle >/dev/null 2>&1; then kaggle_bin="kaggle"
    else
        log "kaggle CLI não encontrado — pulando."
        return 1
    fi

    if [ ! -f "${HOME}/.kaggle/kaggle.json" ] && [ -z "${KAGGLE_USERNAME:-}" ]; then
        log "sem credencial em ~/.kaggle/kaggle.json — pulando."
        return 1
    fi

    log "tentando Kaggle (${KAGGLE_DATASET})..."
    "$kaggle_bin" datasets download -d "$KAGGLE_DATASET" -p "$tmp/kaggle" --unzip >/dev/null 2>&1 \
        || { log "download do Kaggle falhou."; return 1; }
    harvest "$tmp/kaggle"
}

try_uci() {
    local tmp="$1"
    log "baixando do UCI (origem do dataset, CC BY 4.0)..."
    mkdir -p "$tmp/uci"
    curl -sSL --max-time 180 -o "$tmp/uci/bank.zip" "$UCI_URL" \
        || fail "não foi possível baixar de ${UCI_URL}"

    # O zip do UCI contém outros dois zips aninhados; o que interessa é o bank-additional.
    unzip -o -q "$tmp/uci/bank.zip" -d "$tmp/uci" || fail "zip do UCI corrompido"
    [ -f "$tmp/uci/bank-additional.zip" ] || fail "bank-additional.zip não encontrado no pacote do UCI"
    unzip -o -q "$tmp/uci/bank-additional.zip" -d "$tmp/uci" || fail "bank-additional.zip corrompido"

    harvest "$tmp/uci"
}

main() {
    if checksum_ok "$TARGET"; then
        log "${TARGET} já existe e o checksum confere."
        exit 0
    fi

    [ -f "$TARGET" ] && log "arquivo existente com checksum divergente — vai ser rebaixado."

    TMP_DIR="$(mktemp -d)"

    if try_kaggle "$TMP_DIR" && checksum_ok "$TARGET"; then
        log "OK via Kaggle."
    else
        # O Kaggle é um espelho: se estiver indisponível, sem credencial, ou servir um
        # arquivo diferente do esperado, o UCI é a fonte de verdade.
        rm -f "$TARGET"
        try_uci "$TMP_DIR" || fail "não foi possível obter o arquivo de nenhuma das fontes"
        checksum_ok "$TARGET" || fail "checksum não confere após o download do UCI.
  esperado: ${EXPECTED_SHA256}
  obtido:   $(sha256sum "$TARGET" | cut -d' ' -f1)"
        log "OK via UCI."
    fi

    printf '\n%s pronto — %s linhas.\n' "$TARGET" "$(($(wc -l < "$TARGET") - 1))"
}

main "$@"
