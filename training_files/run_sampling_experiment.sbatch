#!/bin/bash
#SBATCH -A a139
#SBATCH -p normal
#SBATCH --gres=gpu:1
#SBATCH -t 6:00:00

set -euo pipefail


# ---------------------------------------------------------------------------
# Experiment parameters — edit these before submitting
# ---------------------------------------------------------------------------
NAME="${NAME:-climbmix_sampling}"  
T="${T:-5}"                          # number of independent samples
SAMPLE_SIZES="${SAMPLE_SIZES:-8192,16384,32768,65536,131072}"  # comma-separated rows per sample, e.g. "8192,60000"
VOCAB_SIZES="${VOCAB_SIZES:-8192,16384,32768,65536,131072,262144}"  # comma-separated, e.g. "4096,16384,32768"
SEED_BASE="${SEED_BASE:-42}"         # sample i uses seed SEED_BASE+i
SOURCE="${SOURCE:-climbmix}"       # finewebedu | parquet | climbmix

# climbmix-only settings (two-tier sampling from HF parquet shards)
DATASET_ID="${DATASET_ID:-karpathy/climbmix-400b-shuffle}"
NUM_SHARDS="${NUM_SHARDS:-2}"        # shards to download per independent sample

# ---------------------------------------------------------------------------
# Storage layout
# ---------------------------------------------------------------------------
REPO_DIR="/capstor/store/cscs/swissai/a0229/jtempus/tokenisation_lp"
IOPS_ROOT="/iopsstor/scratch/cscs/jtempus"
WORK_DIR="${IOPS_ROOT}/sampling_experiments/experiment_${NAME}"
RESULTS_DIR="${REPO_DIR}/experiment_${NAME}"
CLIMBMIX_SHARD_TMP="${WORK_DIR}/downloaded_shards"

TMPDIR="${IOPS_ROOT}/tmp/${SLURM_JOB_ID}"
TMP="${TMPDIR}"
TEMP="${TMPDIR}"
XDG_CACHE_HOME="${IOPS_ROOT}/cache"
HF_HOME="${XDG_CACHE_HOME}/huggingface"
HF_DATASETS_CACHE="${HF_HOME}/datasets"
HF_HUB_CACHE="${HF_HOME}/hub"
PYTHONPYCACHEPREFIX="${XDG_CACHE_HOME}/python"
MPLCONFIGDIR="${XDG_CACHE_HOME}/matplotlib"
CUDA_CACHE_PATH="${XDG_CACHE_HOME}/cuda"
NUMBA_CACHE_DIR="${XDG_CACHE_HOME}/numba"
PYTHONPATH="${REPO_DIR}"

PRETOKENIZER_MODE="${PRETOKENIZER_MODE:-nanochat}"
NUM_PROC="${NUM_PROC:-16}"
BATCH_SIZE="${BATCH_SIZE:-10000}"

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
source /users/jtempus/miniconda3/etc/profile.d/conda.sh
conda activate primer-py311

mkdir -p \
    "${WORK_DIR}" \
    "${RESULTS_DIR}" \
    "${CLIMBMIX_SHARD_TMP}" \
    "${TMPDIR}" \
    "${HF_DATASETS_CACHE}" \
    "${HF_HUB_CACHE}" \
    "${PYTHONPYCACHEPREFIX}" \
    "${MPLCONFIGDIR}" \
    "${CUDA_CACHE_PATH}" \
    "${NUMBA_CACHE_DIR}"

cd "${WORK_DIR}"

export NAME
export T
export SAMPLE_SIZES
export VOCAB_SIZES
export SEED_BASE
export SOURCE
export DATASET_ID
export NUM_SHARDS
export CLIMBMIX_SHARD_TMP
export WORK_DIR
export RESULTS_DIR
export TMPDIR
export TMP
export TEMP
export XDG_CACHE_HOME
export HF_HOME
export HF_DATASETS_CACHE
export HF_HUB_CACHE
export PYTHONPYCACHEPREFIX
export MPLCONFIGDIR
export CUDA_CACHE_PATH
export NUMBA_CACHE_DIR
export PYTHONPATH
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
echo "Repository : ${REPO_DIR}"
echo "Work dir   : ${WORK_DIR}"
echo "Results dir: ${RESULTS_DIR}"
echo "Shard tmp  : ${CLIMBMIX_SHARD_TMP}"
echo "HF cache   : ${HF_HOME}"
echo "TMPDIR     : ${TMPDIR}"
echo "Working cwd: $(pwd)"

python -u "${REPO_DIR}/sampling_experiment.py"
