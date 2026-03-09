#!/bin/bash
#SBATCH -A a139
#SBATCH -p normal
#SBATCH --gres=gpu:1
#SBATCH -t 01:00:00

set -euo pipefail

if [[ -z "${NAME:-}" ]]; then
    echo "ERROR: NAME env var is required. Submit with: NAME=myexp sbatch run_finewebedu.sbatch"
    exit 1
fi

# User-tunable settings
REPO_DIR="/iopsstor/scratch/cscs/jtempus/tokenisation_lp"
TARGET_ROWS="${TARGET_ROWS:-60000}"
SEED="${SEED:-42}"
PRETOKENIZER_MODE="${PRETOKENIZER_MODE:-custom}"
VOCAB_SIZES="${VOCAB_SIZES:-131072}"
NUM_PROC="${NUM_PROC:-16}"
BATCH_SIZE="${BATCH_SIZE:-10000}"
RUN_TOKENIZER_TESTS="${RUN_TOKENIZER_TESTS:-1}"
BYTE_TEST_BEHAVIOR="${BYTE_TEST_BEHAVIOR:-not_all_unk}"

# Property-based paths — no dates, no apertus_2
DATASET_TAG="${NAME}_finewebedu20B_n${TARGET_ROWS}_seed${SEED}"
OUTPUT_DATASET_DIR="${DATASET_TAG}"
TRAIN_DATASET_PATH="${DATASET_TAG}"
RAW_VOCAB_PATH="rounding_vocabs/${DATASET_TAG}"
SAVE_TOKENIZER_DIR="rounded_tokenizers/${DATASET_TAG}"

# IMPORTANT: load conda manually
source /users/jtempus/miniconda3/etc/profile.d/conda.sh

# activate env
conda activate primer-py311

# go to repo
cd "${REPO_DIR}"

export NAME
export TARGET_ROWS
export SEED
export OUTPUT_DATASET_DIR
export TRAIN_DATASET_PATH
export PRETOKENIZER_MODE
export VOCAB_SIZES
export NUM_PROC
export BATCH_SIZE
export RAW_VOCAB_PATH
export SAVE_TOKENIZER_DIR
export RUN_TOKENIZER_TESTS
export BYTE_TEST_BEHAVIOR

echo "Run tag: ${DATASET_TAG}"

echo "Step 1/3: sampling ${TARGET_ROWS} rows from finewebedu-20B -> ${OUTPUT_DATASET_DIR}"
python3 -u sample_tokenizer_data.py --source finewebedu --name "${NAME}"

echo "Step 2/3: training LP vocab(s) from ${TRAIN_DATASET_PATH} -> ${RAW_VOCAB_PATH}"
python3 -u train_tokenizer.py

echo "Step 3/3: rounding/exporting tokenizers -> ${SAVE_TOKENIZER_DIR}"
python3 -u rounding_vocabs_to_tokenizer.py
