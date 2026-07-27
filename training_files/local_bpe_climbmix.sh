#!/bin/bash

set -euo pipefail

# User-tunable settings - UPDATE REPO_DIR to your local path
REPO_DIR="${REPO_DIR:/iopsstor/scratch/cscs/jtempus/tokenisation_lp/}"
VOCAB_SIZES="${VOCAB_SIZES:-8196,16384,32768,65536,131072,262144}"
NUM_SHARDS="${NUM_SHARDS:-7}" # Ensure your local machine has 8+ cores, or lower this
MAX_CHARS="${MAX_CHARS:-2000000000}"
SAVE_DIR="${SAVE_DIR:-bpe_tokenizers_climbmix}"
NANOCHAT_REPO="${NANOCHAT_REPO:-/iopsstor/scratch/cscs/jtempus/nanochat}"

# IMPORTANT: load conda manually - UPDATE this to your local conda installation path
# IMPORTANT: load conda manually - UPDATE this to your local conda installation path
source /users/jtempus/miniconda3/etc/profile.d/conda.sh
conda activate primer-py311

#go to r
# go to repo
cd "${REPO_DIR}"

export VOCAB_SIZES
export NUM_SHARDS
export MAX_CHARS
export SAVE_DIR
export NANOCHAT_REPO

echo "REPO_DIR:    ${REPO_DIR}"
echo "VOCAB_SIZES: ${VOCAB_SIZES}"
echo "NUM_SHARDS:  ${NUM_SHARDS}"
echo "SAVE_DIR:    ${SAVE_DIR}"

python3 -u bpe_tokenizer/train_nano_bpe.py


