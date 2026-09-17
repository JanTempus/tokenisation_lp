#!/bin/bash
#SBATCH -A a139
#SBATCH -p normal
#SBATCH --gres=gpu:1
#SBATCH -t 04:00:00

set -euo pipefail

# User-tunable settings
NAME=please_work                              # experiment tag (required)
REPO_DIR="/iopsstor/scratch/cscs/jtempus/tokenisation_lp"
NUM_SHARDS="${NUM_SHARDS:-8}"
DATASET_ID="${DATASET_ID:-karpathy/climbmix-400b-shuffle}"
PRETOKENIZER_MODE="${PRETOKENIZER_MODE:-nanochat}"
LP_TRAINING_MODE="${LP_TRAINING_MODE:-standard}"
LP_MAX_TOKEN_BYTES="${LP_MAX_TOKEN_BYTES:-64}"
VOCAB_SIZES="${VOCAB_SIZES:-32768}"
NUM_PROC="${NUM_PROC:-16}"
if [[ "${LP_TRAINING_MODE}" == "standard" ]]; then
    BATCH_SIZE="${BATCH_SIZE:-10000}"
else
    BATCH_SIZE="${BATCH_SIZE:-64}"
fi
RUN_TOKENIZER_TESTS="${RUN_TOKENIZER_TESTS:-1}"
BYTE_TEST_BEHAVIOR="${BYTE_TEST_BEHAVIOR:-not_all_unk}"

# Property-based paths
DATASET_TAG="${NAME}_climbmix400b_s${NUM_SHARDS}"
OUTPUT_DIR="${DATASET_TAG}"
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
export NUM_SHARDS
export DATASET_ID
export OUTPUT_DIR
export TRAIN_DATASET_PATH
export PRETOKENIZER_MODE
export LP_TRAINING_MODE LP_MAX_TOKEN_BYTES
# LP_SUPER_BASE_VOCAB_SIZE, when supplied by the caller, is inherited.
export VOCAB_SIZES
export NUM_PROC
export BATCH_SIZE
export RAW_VOCAB_PATH
export SAVE_TOKENIZER_DIR
export RUN_TOKENIZER_TESTS
export BYTE_TEST_BEHAVIOR

echo "Run tag: ${DATASET_TAG}"

echo "Step 1/3: materializing first ${NUM_SHARDS} shards of ${DATASET_ID} -> ${OUTPUT_DIR}"
python3 -u prepare_climbmix_shards.py

echo "Step 2/3: training LP vocab(s) from ${TRAIN_DATASET_PATH} -> ${RAW_VOCAB_PATH}"
python3 -u train_tokenizer.py

echo "Step 3/3: rounding/exporting tokenizers -> ${SAVE_TOKENIZER_DIR}"
python3 -u rounding_vocabs_to_tokenizer.py
