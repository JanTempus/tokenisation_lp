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
TRAIN_DATASET_PATH=/iopsstor/scratch/cscs/jtempus/tokenisation_lp/basic_finewebedu20B_n200000_seed42  # path to dataset saved by sample_tokenizer_data.py
VOCAB_SIZES="8196,16384,32768,65536,131072,262144"           # comma-separated, e.g. "4096,16384,32768"
SAVE_DIR=/iopsstor/scratch/cscs/jtempus/tokenisation_lp/bpe_tokenizer
             # where to save the trained tokenizers

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
source /users/jtempus/miniconda3/etc/profile.d/conda.sh
conda activate primer-py311
cd "${REPO_DIR}"

export TRAIN_DATASET_PATH
export VOCAB_SIZES
export SAVE_DIR

echo "TRAIN_DATASET_PATH: ${TRAIN_DATASET_PATH}"
echo "VOCAB_SIZES:        ${VOCAB_SIZES}"
echo "SAVE_DIR:           ${SAVE_DIR}"

python3 -u bpe_tokenizer/train_sampled.py
