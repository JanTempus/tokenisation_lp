#!/usr/bin/env bash

# Run frequent-pretoken and PickyBPE training inside an already allocated
# interactive node. This script does not submit a Slurm job and does not call
# srun; stdout and stderr remain attached to the current terminal.

set -Eeuo pipefail

# ---------------------------------------------------------------------------
# Configuration: edit values here, then execute this file without arguments.
# ---------------------------------------------------------------------------
TRAIN_DATASET_PATH="/iopsstor/scratch/cscs/jtempus/datasets/climbmix_first7"
OUTPUT_ROOT="/iopsstor/scratch/cscs/jtempus/tokenizer_runs/nanochat_comparison_interactive"
VOCAB_SIZES="8192,16384,32768,65536,131072,262144"
NUM_PROC=256
BATCH_SIZE=10000

# PickyBPE is single-process. It prints one status line after this many merges.
# Set PICKY_VERBOSE=1 only when you want a line for every individual merge.
PICKY_PROGRESS_EVERY=1000
PICKY_VERBOSE=0

# Set these switches to 0 to skip a tokenizer family. Set the rebuild switch
# to 1 when the source dataset changed and the PickyBPE JSONL must be recreated.
RUN_FREQUENT=1
RUN_PICKY=1
REBUILD_PICKY_CORPUS=0

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

banner() {
    echo
    echo "======================================================================"
    echo "[$(timestamp)] $*"
    echo "======================================================================"
}

fail() {
    echo "[$(timestamp)] ERROR: $*" >&2
    exit 1
}

on_error() {
    local exit_code=$?
    echo >&2
    echo "[$(timestamp)] FAILED at line ${BASH_LINENO[0]} (exit ${exit_code})" >&2
    exit "${exit_code}"
}
trap on_error ERR

[[ $# -eq 0 ]] || fail "This script takes no arguments; edit its configuration block instead."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DETECTED_CPUS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"

[[ -e "${TRAIN_DATASET_PATH}" ]] || fail "Dataset not found: ${TRAIN_DATASET_PATH}"
command -v python >/dev/null 2>&1 || fail \
    "No 'python' executable found. Activate the intended Conda or virtual environment first."
PYTHON_BIN="$(command -v python)"

for integer_setting in NUM_PROC BATCH_SIZE PICKY_PROGRESS_EVERY; do
    value="${!integer_setting}"
    [[ "${value}" =~ ^[0-9]+$ ]] || fail "${integer_setting} must be an integer: ${value}"
done
[[ "${NUM_PROC}" -ge 1 ]] || fail "NUM_PROC must be at least 1"
[[ "${BATCH_SIZE}" -ge 1 ]] || fail "BATCH_SIZE must be at least 1"

IFS=',' read -r -a VOCAB_SIZE_LIST <<< "${VOCAB_SIZES}"
[[ ${#VOCAB_SIZE_LIST[@]} -gt 0 ]] || fail "VOCAB_SIZES is empty"
for raw_size in "${VOCAB_SIZE_LIST[@]}"; do
    size="${raw_size//[[:space:]]/}"
    [[ "${size}" =~ ^[1-9][0-9]*$ ]] || fail "Invalid vocabulary size: ${raw_size}"
done

FREQUENT_DIR="${OUTPUT_ROOT}/frequent"
PICKY_DIR="${OUTPUT_ROOT}/picky"
SHARED_DIR="${OUTPUT_ROOT}/shared"
LOG_DIR="${OUTPUT_ROOT}/logs"
PICKY_CORPUS_PATH="${PICKY_CORPUS_PATH:-${SHARED_DIR}/picky_corpus.jsonl}"
mkdir -p "${FREQUENT_DIR}" "${PICKY_DIR}" "${SHARED_DIR}" "${LOG_DIR}"

export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export RAYON_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export HF_MAP_LOAD_FROM_CACHE=0

RUN_LOG="${LOG_DIR}/interactive_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "${RUN_LOG}") 2>&1

banner "Interactive tokenizer run configuration"
echo "Host:                       $(hostname)"
echo "Current time:               $(timestamp)"
echo "Repository:                 ${REPO_DIR}"
echo "Dataset:                    ${TRAIN_DATASET_PATH}"
echo "Output root:                ${OUTPUT_ROOT}"
echo "Run log:                    ${RUN_LOG}"
echo "Vocabulary sizes:           ${VOCAB_SIZES}"
echo "Detected logical CPUs:      ${DETECTED_CPUS}"
echo "Configured workers:         ${NUM_PROC}"
echo "Slurm job ID:               ${SLURM_JOB_ID:-not running under Slurm allocation}"
echo "Slurm CPUs per task:        ${SLURM_CPUS_PER_TASK:-not set}"
echo "Frequent map batch size:    ${BATCH_SIZE}"
echo "Picky progress interval:    ${PICKY_PROGRESS_EVERY}"
echo "Picky per-merge verbosity:  ${PICKY_VERBOSE}"
echo "Run frequent tokenizer:     ${RUN_FREQUENT}"
echo "Run PickyBPE:               ${RUN_PICKY}"
echo "Rebuild Picky corpus:       ${REBUILD_PICKY_CORPUS}"
echo "CPU affinity:               $(awk '/Cpus_allowed_list/ {print $2}' /proc/$$/status 2>/dev/null || echo unknown)"

banner "Python and package checks"
echo "Python executable: ${PYTHON_BIN}"
"${PYTHON_BIN}" --version
"${PYTHON_BIN}" - <<'PY'
import datasets
import regex
import tokenizers
import transformers

print(f"datasets={datasets.__version__}")
print(f"tokenizers={tokenizers.__version__}")
print(f"transformers={transformers.__version__}")
print(f"regex={regex.__version__}")
PY

if command -v free >/dev/null 2>&1; then
    echo
    echo "Memory available:"
    free -h
fi
echo
echo "Output filesystem:"
df -h "${OUTPUT_ROOT}"

if [[ "${RUN_PICKY}" == "1" ]]; then
    if [[ ! -f "${PICKY_CORPUS_PATH}" || "${REBUILD_PICKY_CORPUS}" == "1" ]]; then
        banner "Stage 1/3: preparing PickyBPE JSONL corpus"
        echo "Source dataset: ${TRAIN_DATASET_PATH}"
        echo "Destination:    ${PICKY_CORPUS_PATH}"
        echo "Workers:        ${NUM_PROC}"
        stage_start=${SECONDS}
        env \
            TRAIN_DATASET_PATH="${TRAIN_DATASET_PATH}" \
            PICKY_CORPUS_PATH="${PICKY_CORPUS_PATH}" \
            NUM_PROC="${NUM_PROC}" \
            "${PYTHON_BIN}" -u "${REPO_DIR}/training_files/prepare_picky_corpus.py"
        echo "Stage completed in $((SECONDS - stage_start)) seconds"
    else
        banner "Stage 1/3: reusing existing PickyBPE corpus"
        echo "Corpus: ${PICKY_CORPUS_PATH}"
        echo "Set REBUILD_PICKY_CORPUS=1 to recreate it."
    fi

    PICKY_NUM_LINES="$(wc -l < "${PICKY_CORPUS_PATH}")"
    PICKY_BYTES="$(wc -c < "${PICKY_CORPUS_PATH}")"
    [[ "${PICKY_NUM_LINES}" -gt 0 ]] || fail "PickyBPE corpus is empty"
    echo "PickyBPE corpus documents: ${PICKY_NUM_LINES}"
    echo "PickyBPE corpus bytes:     ${PICKY_BYTES}"
else
    banner "Stage 1/3: skipped because RUN_PICKY=${RUN_PICKY}"
fi

if [[ "${RUN_FREQUENT}" == "1" ]]; then
    banner "Stage 2/3: training frequent-pretoken tokenizers"
    echo "This phase should show: Pretokenizing corpus (num_proc=${NUM_PROC})"
    echo "The later frequency merge and vocabulary saves are single-process."
    stage_start=${SECONDS}
    env \
        TRAIN_DATASET_PATH="${TRAIN_DATASET_PATH}" \
        VOCAB_SIZES="${VOCAB_SIZES}" \
        SAVE_DIR="${FREQUENT_DIR}" \
        NUM_PROC="${NUM_PROC}" \
        BATCH_SIZE="${BATCH_SIZE}" \
        HF_MAP_LOAD_FROM_CACHE=0 \
        "${PYTHON_BIN}" -u "${REPO_DIR}/hf_baseline_tokenizers/train_frequent_pretokens.py"
    echo "Frequent-pretoken stage completed in $((SECONDS - stage_start)) seconds"
else
    banner "Stage 2/3: skipped because RUN_FREQUENT=${RUN_FREQUENT}"
fi

if [[ "${RUN_PICKY}" == "1" ]]; then
    banner "Stage 3/3: training PickyBPE tokenizers"
    echo "PickyBPE is single-process; vocabulary sizes run sequentially."
    for raw_size in "${VOCAB_SIZE_LIST[@]}"; do
        size="${raw_size//[[:space:]]/}"
        banner "PickyBPE vocabulary ${size}"
        stage_start=${SECONDS}
        env \
            PICKY_CORPUS_PATH="${PICKY_CORPUS_PATH}" \
            PICKY_NUM_LINES="${PICKY_NUM_LINES}" \
            VOCAB_SIZE="${size}" \
            SAVE_DIR="${PICKY_DIR}" \
            PICKY_VERBOSE="${PICKY_VERBOSE}" \
            PICKY_PROGRESS_EVERY="${PICKY_PROGRESS_EVERY}" \
            "${PYTHON_BIN}" -u "${REPO_DIR}/boundlessbpe/runpickybpe.py"
        echo "PickyBPE ${size} completed in $((SECONDS - stage_start)) seconds"
    done
else
    banner "Stage 3/3: skipped because RUN_PICKY=${RUN_PICKY}"
fi

banner "Run completed successfully"
echo "Frequent tokenizers: ${FREQUENT_DIR}"
echo "PickyBPE tokenizers: ${PICKY_DIR}"
echo "Full terminal log:  ${RUN_LOG}"
echo "Finished at:        $(timestamp)"
