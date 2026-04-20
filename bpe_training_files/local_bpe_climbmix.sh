#!/bin/bash

set -euo pipefail

# User-tunable settings - UPDATE REPO_DIR to your local path
REPO_DIR="${REPO_DIR:-$HOME/Desktop/Projects/NLP/tokenisation_lp}"
VOCAB_SIZES="${VOCAB_SIZES:-8196,16384,32768,65536,131072,262144}"
NUM_SHARDS="${NUM_SHARDS:-7}" # Ensure your local machine has 8+ cores, or lower this
MAX_CHARS="${MAX_CHARS:-2000000000}"
SAVE_DIR="${SAVE_DIR:-bpe_tokenizers_climbmix}"

# IMPORTANT: load conda manually - UPDATE this to your local conda installation path
source "/home/jantempus/Desktop/Projects/NLP/tokenisation_lp/.venv/bin/activate"

# go to repo
cd "${REPO_DIR}"

export VOCAB_SIZES
export NUM_SHARDS
export MAX_CHARS
export SAVE_DIR

echo "REPO_DIR:    ${REPO_DIR}"
echo "VOCAB_SIZES: ${VOCAB_SIZES}"
echo "NUM_SHARDS:  ${NUM_SHARDS}"
echo "SAVE_DIR:    ${SAVE_DIR}"

python3 -u bpe_tokenizer/train_nano_bpe.py


