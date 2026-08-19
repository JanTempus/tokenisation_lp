#!/usr/bin/env bash

set -euo pipefail

usage() {
    echo "Usage: $0 DATASET_PATH [OUTPUT_DIR]" >&2
    echo >&2
    echo "Optional environment variables:" >&2
    echo "  VOCAB_SIZES=8192,16384  Comma-separated vocabulary sizes" >&2
    echo "  NUM_PROC=4              Pretokenization workers (default: detected CPUs)" >&2
    echo "  PYTHON=python3           Python interpreter to use" >&2
    echo "  REBUILD_PICKY_CORPUS=1   Recreate an existing PickyBPE JSONL corpus" >&2
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRAIN_DATASET_PATH="$1"
OUTPUT_ROOT="${2:-${REPO_DIR}/local_tokenizer_runs}"
VOCAB_SIZES="${VOCAB_SIZES:-8192}"
PYTHON="${PYTHON:-python3}"

if [[ ! -e "${TRAIN_DATASET_PATH}" ]]; then
    echo "Dataset path does not exist: ${TRAIN_DATASET_PATH}" >&2
    exit 2
fi

if [[ -z "${NUM_PROC:-}" ]]; then
    if command -v getconf >/dev/null 2>&1; then
        NUM_PROC="$(getconf _NPROCESSORS_ONLN)"
    else
        NUM_PROC=1
    fi
fi

if ! [[ "${NUM_PROC}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_PROC must be a positive integer; received: ${NUM_PROC}" >&2
    exit 2
fi

IFS=',' read -r -a VOCAB_SIZE_LIST <<< "${VOCAB_SIZES}"
if [[ ${#VOCAB_SIZE_LIST[@]} -eq 0 ]]; then
    echo "VOCAB_SIZES must contain at least one vocabulary size" >&2
    exit 2
fi
for raw_size in "${VOCAB_SIZE_LIST[@]}"; do
    size="${raw_size//[[:space:]]/}"
    if ! [[ "${size}" =~ ^[1-9][0-9]*$ ]]; then
        echo "Invalid vocabulary size: ${raw_size}" >&2
        exit 2
    fi
done

FREQUENT_DIR="${OUTPUT_ROOT}/frequent"
PICKY_DIR="${OUTPUT_ROOT}/picky"
SHARED_DIR="${OUTPUT_ROOT}/shared"
PICKY_CORPUS_PATH="${PICKY_CORPUS_PATH:-${SHARED_DIR}/picky_corpus.jsonl}"

mkdir -p "${FREQUENT_DIR}" "${PICKY_DIR}" "${SHARED_DIR}"

export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "Dataset:          ${TRAIN_DATASET_PATH}"
echo "Output directory: ${OUTPUT_ROOT}"
echo "Vocabulary sizes: ${VOCAB_SIZES}"
echo "Workers:          ${NUM_PROC}"

if [[ ! -f "${PICKY_CORPUS_PATH}" || "${REBUILD_PICKY_CORPUS:-0}" == "1" ]]; then
    echo
    echo "Preparing PickyBPE JSONL corpus"
    env \
        TRAIN_DATASET_PATH="${TRAIN_DATASET_PATH}" \
        PICKY_CORPUS_PATH="${PICKY_CORPUS_PATH}" \
        NUM_PROC="${NUM_PROC}" \
        "${PYTHON}" -u "${REPO_DIR}/training_files/prepare_picky_corpus.py"
else
    echo "Reusing PickyBPE corpus: ${PICKY_CORPUS_PATH}"
fi

PICKY_NUM_LINES="$(wc -l < "${PICKY_CORPUS_PATH}")"
if [[ "${PICKY_NUM_LINES}" -eq 0 ]]; then
    echo "PickyBPE corpus is empty: ${PICKY_CORPUS_PATH}" >&2
    exit 1
fi

echo
echo "Training frequent-pretoken tokenizers"
env \
    TRAIN_DATASET_PATH="${TRAIN_DATASET_PATH}" \
    VOCAB_SIZES="${VOCAB_SIZES}" \
    SAVE_DIR="${FREQUENT_DIR}" \
    NUM_PROC="${NUM_PROC}" \
    BATCH_SIZE="${BATCH_SIZE:-1000}" \
    RAYON_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    "${PYTHON}" -u "${REPO_DIR}/hf_baseline_tokenizers/train_frequent_pretokens.py"

for raw_size in "${VOCAB_SIZE_LIST[@]}"; do
    size="${raw_size//[[:space:]]/}"
    echo
    echo "Training PickyBPE tokenizer with vocabulary size ${size}"
    env \
        PICKY_CORPUS_PATH="${PICKY_CORPUS_PATH}" \
        PICKY_NUM_LINES="${PICKY_NUM_LINES}" \
        VOCAB_SIZE="${size}" \
        SAVE_DIR="${PICKY_DIR}" \
        "${PYTHON}" -u "${REPO_DIR}/boundlessbpe/runpickybpe.py"
done

echo
echo "Finished. Tokenizers are under: ${OUTPUT_ROOT}"
