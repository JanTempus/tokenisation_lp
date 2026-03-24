#!/bin/bash
#SBATCH -A a139
#SBATCH -p normal
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00

set -euo pipefail

if [[ -z "${NAME:-}" ]]; then
    echo "ERROR: NAME env var is required. Submit with: NAME=myexp sbatch run_sampling_experiment.sbatch"
    exit 1
fi

# ---------------------------------------------------------------------------
# Experiment parameters — edit these before submitting
# ---------------------------------------------------------------------------
T="${T:-5}"                          # number of independent samples
SAMPLE_SIZES="${SAMPLE_SIZES:-60000}"  # comma-separated rows per sample, e.g. "8192,60000"
VOCAB_SIZES="${VOCAB_SIZES:-32768}"  # comma-separated, e.g. "4096,16384,32768"
SEED_BASE="${SEED_BASE:-42}"         # sample i uses seed SEED_BASE+i
SOURCE="${SOURCE:-finewebedu}"       # finewebedu | parquet

# ---------------------------------------------------------------------------
# Infrastructure settings — usually no need to change
# ---------------------------------------------------------------------------
REPO_DIR="/iopsstor/scratch/cscs/jtempus/tokenisation_lp"
PRETOKENIZER_MODE="${PRETOKENIZER_MODE:-custom}"
NUM_PROC="${NUM_PROC:-16}"
BATCH_SIZE="${BATCH_SIZE:-10000}"

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
source /users/jtempus/miniconda3/etc/profile.d/conda.sh
conda activate primer-py311
cd "${REPO_DIR}"

export NAME
export T
export SAMPLE_SIZES
export VOCAB_SIZES
export SEED_BASE
export SOURCE
export PRETOKENIZER_MODE
export NUM_PROC
export BATCH_SIZE

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "Experiment : ${NAME}"
echo "T          : ${T}"
echo "SAMPLE_SIZES: ${SAMPLE_SIZES}"
echo "VOCAB_SIZES: ${VOCAB_SIZES}"
echo "SEED_BASE  : ${SEED_BASE}"
echo "SOURCE     : ${SOURCE}"
echo "Output dir : experiment_${NAME}"

python3 -u sampling_experiment.py
