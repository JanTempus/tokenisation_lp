#!/bin/bash
#SBATCH -A a0229
#SBATCH -p debug
#SBATCH -t 01:30:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail

# User-tunable settings
NAME="${NAME:-Multilingual}"
REPO_DIR="${REPO_DIR:-/capstor/store/cscs/swissai/a0229/jtempus/tokenisation_lp}"
TARGET_ROWS="${TARGET_ROWS:-60000}"
PRETOKENIZER_MODE="${PRETOKENIZER_MODE:-nanochat}"
VOCAB_SIZES="${VOCAB_SIZES:-131072}"

if [[ -z "${NAME}" ]]; then
    echo "ERROR: NAME is required. Submit with --export=NAME=<experiment>" >&2
    exit 1
fi

DATASET_TAG="${NAME}_n${TARGET_ROWS}"
TRAIN_DATASET_PATH="${TRAIN_DATASET_PATH:-${DATASET_TAG}}"
SAVE_DIR="${SAVE_DIR:-bpe_tokenizers/${DATASET_TAG}}"

source /users/jtempus/miniconda3/etc/profile.d/conda.sh
conda activate primer-py311
cd "${REPO_DIR}"

export TARGET_ROWS
export TRAIN_DATASET_PATH
export PRETOKENIZER_MODE
export VOCAB_SIZES
export SAVE_DIR

echo "Run tag:          ${DATASET_TAG}"
echo "Training dataset: ${TRAIN_DATASET_PATH}"
echo "BPE output:       ${SAVE_DIR}"
echo "Vocabulary sizes: ${VOCAB_SIZES}"

python3 -u bpe_tokenizer/train_sampled.py
