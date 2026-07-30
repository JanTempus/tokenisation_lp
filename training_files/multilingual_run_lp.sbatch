#!/bin/bash
#SBATCH -A a0229
#SBATCH -p normal
#SBATCH -t 03:30:00
#SBATCH --cpus-per-task=32
#SBATCH --output=/capstor/store/cscs/swissai/a0229/jtempus/tokenisation_lp/logs/slurm-%j.out
#SBATCH --error=/capstor/store/cscs/swissai/a0229/jtempus/tokenisation_lp/logs/slurm-%j.err


set -euo pipefail

# User-tunable settings
NAME="${NAME:-Multilingual}"
REPO_DIR="${REPO_DIR:-/capstor/store/cscs/swissai/a0229/jtempus/tokenisation_lp}"
IOPS_ROOT="${IOPS_ROOT:-/iopsstor/scratch/cscs/jtempus/tokenisation_lp}"
TARGET_ROWS="${TARGET_ROWS:-60000}"
PRETOKENIZER_MODE="${PRETOKENIZER_MODE:-nanochat}"
VOCAB_SIZES="${VOCAB_SIZES:-131072}"
NUM_PROC="${NUM_PROC:-32}"
BATCH_SIZE="${BATCH_SIZE:-10000}"
RUN_TOKENIZER_TESTS="${RUN_TOKENIZER_TESTS:-1}"
BYTE_TEST_BEHAVIOR="${BYTE_TEST_BEHAVIOR:-not_all_unk}"

if [[ -z "${NAME}" ]]; then
    echo "ERROR: NAME is required. Submit with --export=NAME=<experiment>" >&2
    exit 1
fi

DATASET_TAG="${NAME}_n${TARGET_ROWS}"
RUN_DIR="${IOPS_ROOT}/runs/${DATASET_TAG}_${SLURM_JOB_ID}"
OUTPUT_DATASET_DIR="${OUTPUT_DATASET_DIR:-${IOPS_ROOT}/datasets/${DATASET_TAG}}"
TRAIN_DATASET_PATH="${TRAIN_DATASET_PATH:-${OUTPUT_DATASET_DIR}}"
RAW_VOCAB_PATH="${RAW_VOCAB_PATH:-${IOPS_ROOT}/raw_vocabs/${DATASET_TAG}}"
LP_GRAPH_CACHE_DIR="${LP_GRAPH_CACHE_DIR:-${RUN_DIR}/lp_graph}"
SAVE_TOKENIZER_DIR="${SAVE_TOKENIZER_DIR:-${REPO_DIR}/rounded_tokenizers/${DATASET_TAG}}"

TMPDIR="${RUN_DIR}/tmp"
TMP="${TMPDIR}"
TEMP="${TMPDIR}"
PYTHONPYCACHEPREFIX="${RUN_DIR}/pycache"
XDG_CACHE_HOME="${IOPS_ROOT}/cache"
HF_HOME="${XDG_CACHE_HOME}/huggingface"
HF_DATASETS_CACHE="${HF_HOME}/datasets"
HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
CUDA_CACHE_PATH="${XDG_CACHE_HOME}/cuda"
NUMBA_CACHE_DIR="${XDG_CACHE_HOME}/numba"
MPLCONFIGDIR="${XDG_CACHE_HOME}/matplotlib"

mkdir -p \
    "${RUN_DIR}" \
    "${TMPDIR}" \
    "${PYTHONPYCACHEPREFIX}" \
    "${LP_GRAPH_CACHE_DIR}" \
    "${IOPS_ROOT}/datasets" \
    "${IOPS_ROOT}/raw_vocabs" \
    "${HF_DATASETS_CACHE}" \
    "${HUGGINGFACE_HUB_CACHE}" \
    "${CUDA_CACHE_PATH}" \
    "${NUMBA_CACHE_DIR}" \
    "${MPLCONFIGDIR}"

source /users/jtempus/miniconda3/etc/profile.d/conda.sh
conda activate primer-py311
cd "${RUN_DIR}"

export NAME
export REPO_DIR
export IOPS_ROOT
export RUN_DIR
export TARGET_ROWS
export OUTPUT_DATASET_DIR
export TRAIN_DATASET_PATH
export PRETOKENIZER_MODE
export VOCAB_SIZES
export NUM_PROC
export BATCH_SIZE
export LP_GRAPH_CACHE_DIR
export TMPDIR
export TMP
export TEMP
export PYTHONPYCACHEPREFIX
export XDG_CACHE_HOME
export HF_HOME
export HF_DATASETS_CACHE
export HUGGINGFACE_HUB_CACHE
export CUDA_CACHE_PATH
export NUMBA_CACHE_DIR
export MPLCONFIGDIR
export RAW_VOCAB_PATH
export SAVE_TOKENIZER_DIR
export RUN_TOKENIZER_TESTS
export BYTE_TEST_BEHAVIOR

echo "Run tag:              ${DATASET_TAG}"
echo "Repository:           ${REPO_DIR}"
echo "Job working dir:      ${RUN_DIR}"
echo "Training dataset:     ${TRAIN_DATASET_PATH}"
echo "Raw vocabulary:       ${RAW_VOCAB_PATH}"
echo "LP graph cache:       ${LP_GRAPH_CACHE_DIR}"
echo "Final tokenizers:     ${SAVE_TOKENIZER_DIR}"
echo "Map workers:          ${NUM_PROC}"
echo "Map batch size:       ${BATCH_SIZE}"

if [[ -f "${TRAIN_DATASET_PATH}/sampling_manifest.json" ]]; then
    echo "Step 1/3: reusing existing dataset at ${TRAIN_DATASET_PATH}"
elif [[ "${TRAIN_DATASET_PATH}" != "${OUTPUT_DATASET_DIR}" ]]; then
    echo "ERROR: overridden TRAIN_DATASET_PATH has no sampling manifest: ${TRAIN_DATASET_PATH}" >&2
    exit 1
else
    echo "Step 1/3: downloading ordered FineWeb/FineWeb2 prefixes -> ${OUTPUT_DATASET_DIR}"
    python3 -u "${REPO_DIR}/sample_tokenizer_data.py" \
        --source fineweb2 \
        --name "${NAME}" \
        --target-rows "${TARGET_ROWS}" \
        --output-dataset-dir "${OUTPUT_DATASET_DIR}"
fi

echo "Step 2/3: training LP vocab(s) -> ${RAW_VOCAB_PATH}"
python3 -u "${REPO_DIR}/train_tokenizer.py"

echo "Step 3/3: rounding/exporting LP tokenizer(s) -> ${SAVE_TOKENIZER_DIR}"
python3 -u "${REPO_DIR}/rounding_vocabs_to_tokenizer.py"
