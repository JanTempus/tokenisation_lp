#!/bin/bash
# Local (no Slurm) LP tokenizer run on the first NUM_SHARDS shards of FineWeb sample/10BT.
# Mirrors multilingual_run_lp.sbatch. The CELEX morphology penalty is disabled.
#
#   bash training_files/fineweb_run_lp.sh
#   CUDA_VISIBLE_DEVICES=2 NUM_SHARDS=3 VOCAB_SIZES=65536 bash training_files/fineweb_run_lp.sh
#   CUDA_VISIBLE_DEVICES=0,1 LP_NUM_GPUS=2 nohup bash training_files/fineweb_run_lp.sh &

set -euo pipefail

# User-tunable settings
NAME="${NAME:-FineWeb}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${HOME}/tokenisation_lp_data}"
VENV_DIR="${VENV_DIR:-${REPO_DIR}/.venv}"
DATASET_ID="${DATASET_ID:-HuggingFaceFW/fineweb}"
PARQUET_PREFIX="${PARQUET_PREFIX:-sample/10BT/}"
NUM_SHARDS="${NUM_SHARDS:-4}"
PRETOKENIZER_MODE="${PRETOKENIZER_MODE:-nanochat}"
VOCAB_SIZES="${VOCAB_SIZES:-32768,65536}"
NUM_PROC="${NUM_PROC:-32}"
BATCH_SIZE="${BATCH_SIZE:-10000}"
RUN_TOKENIZER_TESTS="${RUN_TOKENIZER_TESTS:-1}"
BYTE_TEST_BEHAVIOR="${BYTE_TEST_BEHAVIOR:-not_all_unk}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
LP_NUM_GPUS="${LP_NUM_GPUS:-1}"
PENTLY_RHO=0  # 0 disables the morphology (CELEX) and Unicode penalties

RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)_$$"
DATASET_TAG="${NAME}_s${NUM_SHARDS}"
RUN_DIR="${DATA_ROOT}/runs/${DATASET_TAG}_${RUN_ID}"
TRAIN_DATASET_PATH="${TRAIN_DATASET_PATH:-${DATA_ROOT}/datasets/${DATASET_TAG}}"
RAW_VOCAB_PATH="${RAW_VOCAB_PATH:-${DATA_ROOT}/raw_vocabs/${DATASET_TAG}}"
LP_GRAPH_CACHE_DIR="${LP_GRAPH_CACHE_DIR:-${RUN_DIR}/lp_graph}"
SAVE_TOKENIZER_DIR="${SAVE_TOKENIZER_DIR:-${REPO_DIR}/rounded_tokenizers/${DATASET_TAG}}"

TMPDIR="${DATA_ROOT}/tmp/${RUN_ID}"
TMP="${TMPDIR}"
TEMP="${TMPDIR}"
PYTHONPYCACHEPREFIX="${RUN_DIR}/pycache"
XDG_CACHE_HOME="${DATA_ROOT}/cache"
HF_HOME="${XDG_CACHE_HOME}/huggingface"
HF_DATASETS_CACHE="${HF_HOME}/datasets"
HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
CUDA_CACHE_PATH="${XDG_CACHE_HOME}/cuda"
NUMBA_CACHE_DIR="${XDG_CACHE_HOME}/numba"
MPLCONFIGDIR="${XDG_CACHE_HOME}/matplotlib"

mkdir -p \
    "${DATA_ROOT}/logs" \
    "${RUN_DIR}" \
    "${TMPDIR}" \
    "${PYTHONPYCACHEPREFIX}" \
    "${LP_GRAPH_CACHE_DIR}" \
    "${DATA_ROOT}/datasets" \
    "${DATA_ROOT}/raw_vocabs" \
    "${HF_DATASETS_CACHE}" \
    "${HUGGINGFACE_HUB_CACHE}" \
    "${CUDA_CACHE_PATH}" \
    "${NUMBA_CACHE_DIR}" \
    "${MPLCONFIGDIR}"

LOG_FILE="${DATA_ROOT}/logs/fineweb_output_${RUN_ID}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

source "${VENV_DIR}/bin/activate"
cd "${RUN_DIR}"

export NAME REPO_DIR RUN_DIR DATASET_ID PARQUET_PREFIX NUM_SHARDS TRAIN_DATASET_PATH
export PRETOKENIZER_MODE VOCAB_SIZES NUM_PROC BATCH_SIZE LP_GRAPH_CACHE_DIR
export TMPDIR TMP TEMP PYTHONPYCACHEPREFIX XDG_CACHE_HOME
export HF_HOME HF_DATASETS_CACHE HUGGINGFACE_HUB_CACHE
export CUDA_CACHE_PATH NUMBA_CACHE_DIR MPLCONFIGDIR
export RAW_VOCAB_PATH SAVE_TOKENIZER_DIR RUN_TOKENIZER_TESTS BYTE_TEST_BEHAVIOR
export CUDA_VISIBLE_DEVICES LP_NUM_GPUS PENTLY_RHO

echo "Run tag:              ${DATASET_TAG}"
echo "Repository:           ${REPO_DIR}"
echo "Log file:             ${LOG_FILE}"
echo "Job working dir:      ${RUN_DIR}"
echo "Input shards:         first ${NUM_SHARDS} of ${DATASET_ID}/${PARQUET_PREFIX}"
echo "Penalty rho:          ${PENTLY_RHO} (morphology disabled)"
echo "Training dataset:     ${TRAIN_DATASET_PATH}"
echo "Raw vocabulary:       ${RAW_VOCAB_PATH}"
echo "LP graph cache:       ${LP_GRAPH_CACHE_DIR}"
echo "Final tokenizers:     ${SAVE_TOKENIZER_DIR}"
echo "Vocab sizes:          ${VOCAB_SIZES}"
echo "GPUs:                 ${CUDA_VISIBLE_DEVICES} (LP_NUM_GPUS=${LP_NUM_GPUS})"
echo "Map workers:          ${NUM_PROC}"
echo "Map batch size:       ${BATCH_SIZE}"

echo "Step 1/3: preparing first ${NUM_SHARDS} shards of ${DATASET_ID}/${PARQUET_PREFIX} (reuse if dataset exists)"
python3 -u "${REPO_DIR}/prepare_climbmix_shards.py"

echo "Step 2/3: training LP vocab(s) -> ${RAW_VOCAB_PATH}"
python3 -u "${REPO_DIR}/train_tokenizer.py"

echo "Step 3/3: rounding/exporting LP tokenizer(s) -> ${SAVE_TOKENIZER_DIR}"
python3 -u "${REPO_DIR}/rounding_vocabs_to_tokenizer.py"
