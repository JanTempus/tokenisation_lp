#!/bin/bash
#SBATCH -A a139
#SBATCH -p normal
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH -t 08:00:00

set -euo pipefail

# User-tunable settings
REPO_DIR="${REPO_DIR:-/iopsstor/scratch/cscs/jtempus/tokenisation_lp}"
VOCAB_SIZE="${VOCAB_SIZE:-32768}"
NUM_SHARDS="${NUM_SHARDS:-8}"
MAX_CHARS="${MAX_CHARS:-2000000000}"
SAVE_DIR="${SAVE_DIR:-bpe_tokenizers_climbmix}"

# IMPORTANT: load conda manually
source /users/jtempus/miniconda3/etc/profile.d/conda.sh

# activate env
conda activate primer-py311

# go to repo
cd "${REPO_DIR}"

export VOCAB_SIZE
export NUM_SHARDS
export MAX_CHARS
export SAVE_DIR

echo "REPO_DIR:    ${REPO_DIR}"
echo "VOCAB_SIZE:  ${VOCAB_SIZE}"
echo "NUM_SHARDS:  ${NUM_SHARDS}"
echo "MAX_CHARS:   ${MAX_CHARS}"
echo "SAVE_DIR:    ${SAVE_DIR}"

python3 -u bpe_tokenizer/train_climbmix.py
