#!/bin/bash
#SBATCH -A a139
#SBATCH -p normal
#SBATCH --gres=gpu:1
#SBATCH -t 1:00:00

set -euo pipefail

# User-tunable settings
REPO_DIR="/iopsstor/scratch/cscs/jtempus/tokenisation_lp"
DATASET_BASE="/capstor/store/cscs/swissai/a139/datasets/tokenizer_training"
TARGET_ROWS="${TARGET_ROWS:-60000}"
SEED="${SEED:-42}"
NUM_PROC="${NUM_PROC:-16}"
BATCH_SIZE="${BATCH_SIZE:-10000}"

# IMPORTANT: load conda manually
source /users/jtempus/miniconda3/etc/profile.d/conda.sh

# activate env
conda activate primer-py311

# go to repo
cd "${REPO_DIR}"

export TOKENIZER_DATASET_BASE="${DATASET_BASE}"
export TARGET_ROWS
export SEED
export NUM_PROC
export BATCH_SIZE

# run
python3 train_tokenizer.py
