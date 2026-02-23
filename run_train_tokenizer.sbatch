#!/bin/bash
#SBATCH -A a139
#SBATCH -p normal
#SBATCH --gres=gpu:1
#SBATCH -t 01:00:00

set -euo pipefail

# User-tunable settings
REPO_DIR="/iopsstor/scratch/cscs/jtempus/tokenisation_lp"
TOKENIZER_DATASET_BASE="${TOKENIZER_DATASET_BASE:-/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/tokenizer_training_dataset}"
TARGET_ROWS="${TARGET_ROWS:-120000}"
SEED="${SEED:-42}"
OUTPUT_DATASET_DIR="${OUTPUT_DATASET_DIR:-sampled_tokenizer_data}"
TRAIN_DATASET_PATH="${TRAIN_DATASET_PATH:-${OUTPUT_DATASET_DIR}}"
NUM_PROC="${NUM_PROC:-16}"
BATCH_SIZE="${BATCH_SIZE:-10000}"
PRETOKENIZER_MODE="${PRETOKENIZER_MODE:-pythia}"
VOCAB_SIZES="${VOCAB_SIZES:-131072}"
RAW_VOCAB_PATH="${RAW_VOCAB_PATH:-rounding_vocabs_apertus_2}"
SAVE_TOKENIZER_DIR="${SAVE_TOKENIZER_DIR:-rounded_tokenizers_apertus_2}"
RUN_TOKENIZER_TESTS="${RUN_TOKENIZER_TESTS:-1}"
BYTE_TEST_BEHAVIOR="${BYTE_TEST_BEHAVIOR:-not_all_unk}"

# IMPORTANT: load conda manually
source /users/jtempus/miniconda3/etc/profile.d/conda.sh

# activate env
conda activate primer-py311

# go to repo
cd "${REPO_DIR}"
mkdir -p "${OUTPUT_DATASET_DIR}"

export TOKENIZER_DATASET_BASE
export TARGET_ROWS
export SEED
export OUTPUT_DATASET_DIR
export TRAIN_DATASET_PATH
export NUM_PROC
export BATCH_SIZE
export PRETOKENIZER_MODE
export VOCAB_SIZES
export RAW_VOCAB_PATH
export SAVE_TOKENIZER_DIR
export RUN_TOKENIZER_TESTS
export BYTE_TEST_BEHAVIOR

echo "Step 1/3: sampling dataset from ${TOKENIZER_DATASET_BASE} -> ${OUTPUT_DATASET_DIR} (rows=${TARGET_ROWS}, seed=${SEED})"
python3 -u sample_tokenizer_data.py

echo "Step 2/3: training LP vocab(s) from ${TRAIN_DATASET_PATH} -> ${RAW_VOCAB_PATH}"
python3 -u train_tokenizer.py

echo "Step 3/3: rounding/exporting tokenizers -> ${SAVE_TOKENIZER_DIR}"
python3 -u rounding_vocabs_to_tokenizer.py
