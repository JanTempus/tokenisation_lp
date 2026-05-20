#!/bin/bash
#SBATCH -A a139
#SBATCH -p normal
#SBATCH --gres=gpu:1
#SBATCH -t 04:00:00

set -euo pipefail

# User-tunable settings
NAME="${NAME:-apertus_lp}"
REPO_DIR="${REPO_DIR:-/iopsstor/scratch/cscs/jtempus/tokenisation_lp}"
TOKENIZER_DATASET_BASE="${TOKENIZER_DATASET_BASE:-/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/tokenizer_training_dataset}"
TARGET_ROWS="${TARGET_ROWS:-120000}"
SEED="${SEED:-42}"
PRETOKENIZER_MODE="${PRETOKENIZER_MODE:-apertus}"
VOCAB_SIZES="${VOCAB_SIZES:-32768}"
NUM_PROC="${NUM_PROC:-16}"
BATCH_SIZE="${BATCH_SIZE:-10000}"
RUN_TOKENIZER_TESTS="${RUN_TOKENIZER_TESTS:-1}"
BYTE_TEST_BEHAVIOR="${BYTE_TEST_BEHAVIOR:-not_all_unk}"

# Property-based paths
DATASET_TAG="${NAME}_apertus_n${TARGET_ROWS}_seed${SEED}"
OUTPUT_DATASET_DIR="${DATASET_TAG}"
TRAIN_DATASET_PATH="${OUTPUT_DATASET_DIR}"
RAW_VOCAB_PATH="rounding_vocabs/${DATASET_TAG}"
SAVE_TOKENIZER_DIR="rounded_tokenizers/${DATASET_TAG}"

# IMPORTANT: load conda manually
source /users/jtempus/miniconda3/etc/profile.d/conda.sh

# activate env
conda activate primer-py311

# go to repo
cd "${REPO_DIR}"

export NAME
export TOKENIZER_DATASET_BASE
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
echo "PRETOKENIZER_MODE: ${PRETOKENIZER_MODE}"
echo "TOKENIZER_DATASET_BASE: ${TOKENIZER_DATASET_BASE}"
echo "TARGET_ROWS: ${TARGET_ROWS}"
echo "VOCAB_SIZES: ${VOCAB_SIZES}"

echo "Step 1/3: sampling Apertus tokenizer data from ${TOKENIZER_DATASET_BASE} -> ${OUTPUT_DATASET_DIR}"
python3 -u sample_tokenizer_data.py \
    --source parquet \
    --target-rows "${TARGET_ROWS}" \
    --seed "${SEED}" \
    --output-dataset-dir "${OUTPUT_DATASET_DIR}"

echo "Step 2/3: training Apertus LP vocab(s) from ${TRAIN_DATASET_PATH} -> ${RAW_VOCAB_PATH}"
python3 -u train_tokenizer.py

echo "Step 3/3: rounding/exporting Apertus tokenizers -> ${SAVE_TOKENIZER_DIR}"
python3 -u rounding_vocabs_to_tokenizer.py
