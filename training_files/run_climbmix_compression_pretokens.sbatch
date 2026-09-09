#!/bin/bash
#SBATCH -A a0229
#SBATCH -p normal
#SBATCH --job-name=climbmix-compression-pretokens
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH -t 00:30:00
#SBATCH --output=/iopsstor/scratch/cscs/jtempus/compression-pretokens-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/jtempus/compression-pretokens-%j.err

set -euo pipefail

# Submit: sbatch training_files/run_climbmix_compression_pretokens.sbatch
# Rank by frequency * (encoded pretoken length - 1).
# Reuse the ClimbMix dataset prepared by run_parallel_tokenizer_suite.sbatch.
REPO_DIR="${REPO_DIR:-/capstor/store/cscs/swissai/a0229/jtempus/tokenisation_lp}"
WORKING_DIR="${WORKING_DIR:-/iopsstor/scratch/cscs/jtempus}"
TRAIN_DATASET_PATH="${TRAIN_DATASET_PATH:-${WORKING_DIR}/datasets/climbmix_first7}"
SAVE_DIR="${SAVE_DIR:-${WORKING_DIR}/tokenizer_runs/nanochat_comparison/compression_pretokens}"
VOCAB_SIZES="${VOCAB_SIZES:-8192,16384,32768,65536,131072,262144}"
NUM_PROC="${SLURM_CPUS_PER_TASK:-64}"
BATCH_SIZE="${BATCH_SIZE:-10000}"
WEIGHT_BY_COMPRESSION=1

TMPDIR="${WORKING_DIR}/tmp/${SLURM_JOB_ID}"
XDG_CACHE_HOME="${WORKING_DIR}/cache"
HF_HOME="${XDG_CACHE_HOME}/huggingface"
HF_DATASETS_CACHE="${HF_HOME}/datasets"
HF_HUB_CACHE="${HF_HOME}/hub"
PYTHONPYCACHEPREFIX="${XDG_CACHE_HOME}/python"
PYTHONPATH="${REPO_DIR}"
export TMPDIR XDG_CACHE_HOME HF_HOME HF_DATASETS_CACHE HF_HUB_CACHE PYTHONPYCACHEPREFIX PYTHONPATH
export TRAIN_DATASET_PATH SAVE_DIR VOCAB_SIZES NUM_PROC BATCH_SIZE WEIGHT_BY_COMPRESSION
export RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

source /users/jtempus/miniconda3/etc/profile.d/conda.sh
conda activate primer-py311
cd "${REPO_DIR}"

mkdir -p \
    "${SAVE_DIR}" \
    "${TMPDIR}" \
    "${HF_DATASETS_CACHE}" \
    "${HF_HUB_CACHE}" \
    "${PYTHONPYCACHEPREFIX}"

echo "Training dataset: ${TRAIN_DATASET_PATH}"
echo "Output directory: ${SAVE_DIR}"
echo "Vocabulary sizes: ${VOCAB_SIZES}"
echo "Ranking: frequency * (encoded length - 1)"

srun --exclusive --exact --nodes=1 --ntasks=1 \
    --cpus-per-task="${NUM_PROC}" --cpu-bind=cores \
    /usr/bin/time -v \
    python3 -u "${REPO_DIR}/hf_baseline_tokenizers/train_frequent_pretokens.py"
