#!/bin/bash
#SBATCH -A a139
#SBATCH -p normal
#SBATCH --cpus-per-task=8
#SBATCH -t 00:45:00

set -euo pipefail

# ---------------------------------------------------------------------------
# Experiment parameters — fill in before submitting
# ---------------------------------------------------------------------------
REPO_DIR=/iopsstor/scratch/cscs/jtempus/tokenisation_lp
TRAIN_DATASET_PATH=/iopsstor/scratch/cscs/jtempus/tokenisation_lp/basic_finewebedu20B_n200000_seed42
VOCAB_SIZES="8196,16384,32768,65536,131072,262144"
SAVE_DIR=/iopsstor/scratch/cscs/jtempus/tokenisation_lp/bpe_tokenizer
# UPDATE this path to where the nanochat repo lives on the cluster
NANOCHAT_REPO=/iopsstor/scratch/cscs/jtempus/nanochat

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
source /users/jtempus/miniconda3/etc/profile.d/conda.sh
conda activate primer-py311
cd "${REPO_DIR}"

export TRAIN_DATASET_PATH
export VOCAB_SIZES
export SAVE_DIR
export NANOCHAT_REPO

echo "TRAIN_DATASET_PATH: ${TRAIN_DATASET_PATH}"
echo "VOCAB_SIZES:        ${VOCAB_SIZES}"
echo "SAVE_DIR:           ${SAVE_DIR}"
echo "NANOCHAT_REPO:      ${NANOCHAT_REPO}"

python3 -u bpe_tokenizer/train_nano_bpe.py
