#!/bin/bash
#SBATCH -A a139
#SBATCH -p normal
#SBATCH --gres=gpu:1
#SBATCH -t 01:00:00

set -euo pipefail

# User-tunable settings
REPO_DIR="/iopsstor/scratch/cscs/jtempus/tokenisation_lp"
TRAIN_DATASET_PATH="${TRAIN_DATASET_PATH:-/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/tokenizer_training_dataset}"
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

export TRAIN_DATASET_PATH
export NUM_PROC
export BATCH_SIZE
export PRETOKENIZER_MODE
export VOCAB_SIZES
export RAW_VOCAB_PATH
export SAVE_TOKENIZER_DIR
export RUN_TOKENIZER_TESTS
export BYTE_TEST_BEHAVIOR

echo "Step 1/2: training LP vocab(s) from ${TRAIN_DATASET_PATH} -> ${RAW_VOCAB_PATH}"
python3 -u train_tokenizer.py

echo "Step 2/2: rounding/exporting tokenizers -> ${SAVE_TOKENIZER_DIR}"
python3 -u rounding_vocabs_to_tokenizer.py
