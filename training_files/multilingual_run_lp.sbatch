#!/bin/bash
#SBATCH -A a0229
#SBATCH -p debug
#SBATCH -t 01:30:00
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err


set -euo pipefail

# User-tunable settings
NAME="${NAME:-Multilingual}"
REPO_DIR="${REPO_DIR:-/capstor/store/cscs/swissai/a0229/jtempus/tokenisation_lp}"
TARGET_ROWS="${TARGET_ROWS:-60000}"
PRETOKENIZER_MODE="${PRETOKENIZER_MODE:-nanochat}"
VOCAB_SIZES="${VOCAB_SIZES:-131072}"
NUM_PROC="${NUM_PROC:-16}"
BATCH_SIZE="${BATCH_SIZE:-10000}"
LP_NUM_PROC="${LP_NUM_PROC:-${SLURM_CPUS_PER_TASK:-16}}"
LP_MAP_BATCH_SIZE="${LP_MAP_BATCH_SIZE:-1000}"
LP_GRAPH_CACHE_DIR="${LP_GRAPH_CACHE_DIR:-}"
RUN_TOKENIZER_TESTS="${RUN_TOKENIZER_TESTS:-1}"
BYTE_TEST_BEHAVIOR="${BYTE_TEST_BEHAVIOR:-not_all_unk}"

if [[ -z "${NAME}" ]]; then
    echo "ERROR: NAME is required. Submit with --export=NAME=<experiment>" >&2
    exit 1
fi

DATASET_TAG="${NAME}_n${TARGET_ROWS}"
OUTPUT_DATASET_DIR="${OUTPUT_DATASET_DIR:-${DATASET_TAG}}"
TRAIN_DATASET_PATH="${TRAIN_DATASET_PATH:-${OUTPUT_DATASET_DIR}}"
RAW_VOCAB_PATH="${RAW_VOCAB_PATH:-rounding_vocabs/${DATASET_TAG}}"
SAVE_TOKENIZER_DIR="${SAVE_TOKENIZER_DIR:-rounded_tokenizers/${DATASET_TAG}}"

source /users/jtempus/miniconda3/etc/profile.d/conda.sh
conda activate primer-py311
cd "${REPO_DIR}"

export NAME
export TARGET_ROWS
export OUTPUT_DATASET_DIR
export TRAIN_DATASET_PATH
export PRETOKENIZER_MODE
export VOCAB_SIZES
export NUM_PROC
export BATCH_SIZE
export LP_NUM_PROC
export LP_MAP_BATCH_SIZE
export LP_GRAPH_CACHE_DIR
export RAW_VOCAB_PATH
export SAVE_TOKENIZER_DIR
export RUN_TOKENIZER_TESTS
export BYTE_TEST_BEHAVIOR

echo "Run tag:              ${DATASET_TAG}"
echo "Training dataset:     ${TRAIN_DATASET_PATH}"
echo "LP map workers:       ${LP_NUM_PROC}"
echo "LP map batch size:    ${LP_MAP_BATCH_SIZE}"
if [[ -n "${LP_GRAPH_CACHE_DIR}" ]]; then
    echo "LP graph cache:       ${LP_GRAPH_CACHE_DIR}"
else
    echo "LP graph cache:       ${TMPDIR:-/tmp}/tokenisation_lp_${SLURM_JOB_ID:-manual}"
fi

if [[ -f "${TRAIN_DATASET_PATH}/sampling_manifest.json" ]]; then
    echo "Step 1/3: reusing existing dataset at ${TRAIN_DATASET_PATH}"
elif [[ "${TRAIN_DATASET_PATH}" != "${OUTPUT_DATASET_DIR}" ]]; then
    echo "ERROR: overridden TRAIN_DATASET_PATH has no sampling manifest: ${TRAIN_DATASET_PATH}" >&2
    exit 1
else
    echo "Step 1/3: downloading ordered FineWeb/FineWeb2 prefixes -> ${OUTPUT_DATASET_DIR}"
    python3 -u sample_tokenizer_data.py \
        --source fineweb2 \
        --name "${NAME}" \
        --target-rows "${TARGET_ROWS}" \
        --output-dataset-dir "${OUTPUT_DATASET_DIR}"
fi

echo "Step 2/3: training LP vocab(s) -> ${RAW_VOCAB_PATH}"
python3 -u train_tokenizer.py

echo "Step 3/3: rounding/exporting LP tokenizer(s) -> ${SAVE_TOKENIZER_DIR}"
python3 -u rounding_vocabs_to_tokenizer.py
