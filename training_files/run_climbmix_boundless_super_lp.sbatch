#!/bin/bash
#SBATCH -A a0229
#SBATCH -p debug
#SBATCH --job-name=climbmix-document-lp
#SBATCH --array=0-1%1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH -t 01:20:00
#SBATCH --output=/iopsstor/scratch/cscs/jtempus/document-lp-%A_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/jtempus/document-lp-%A_%a.err

set -euo pipefail

# Submit once to train both modes, one at a time:
# sbatch training_files/run_climbmix_boundless_super_lp.sbatch
case "${SLURM_ARRAY_TASK_ID:?Submit this file with sbatch}" in
    0) LP_TRAINING_MODE=boundless ;;
    1) LP_TRAINING_MODE=super ;;
    *) echo "Unexpected array task: ${SLURM_ARRAY_TASK_ID}" >&2; exit 1 ;;
esac

REPO_DIR="${REPO_DIR:-/capstor/store/cscs/swissai/a0229/jtempus/tokenisation_lp}"
IOPS_ROOT="${IOPS_ROOT:-/iopsstor/scratch/cscs/jtempus/tokenisation_lp}"
CONDA_ROOT="${CONDA_ROOT:-/users/jtempus/miniconda3}"
DATASET_ID="${DATASET_ID:-karpathy/climbmix-400b-shuffle}"
NUM_SHARDS="${NUM_SHARDS:-1}"
PRETOKENIZER_MODE="${PRETOKENIZER_MODE:-nanochat}"
VOCAB_SIZES="16384,65536,262144"
LP_MAX_TOKEN_BYTES="${LP_MAX_TOKEN_BYTES:-64}"
LP_NUM_GPUS=1
NUM_PROC="${NUM_PROC:-${SLURM_CPUS_PER_TASK}}"
BATCH_SIZE="${BATCH_SIZE:-64}"
PENTLY_RHO=0
VOCAB_UTILISATION_WEIGHT=0
RUN_TOKENIZER_TESTS="${RUN_TOKENIZER_TESTS:-1}"

EXPERIMENT_TAG="climbmix400b_s${NUM_SHARDS}_${PRETOKENIZER_MODE}_document_lp"
RUN_DIR="${IOPS_ROOT}/runs/${EXPERIMENT_TAG}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
TRAIN_DATASET_PATH="${TRAIN_DATASET_PATH:-${IOPS_ROOT}/datasets/climbmix400b_s${NUM_SHARDS}}"
RAW_VOCAB_PATH="${RAW_VOCAB_PATH:-${IOPS_ROOT}/raw_vocabs/${EXPERIMENT_TAG}}"
SAVE_TOKENIZER_DIR="${SAVE_TOKENIZER_DIR:-${REPO_DIR}/rounded_tokenizers/${EXPERIMENT_TAG}}"
LP_GRAPH_CACHE_DIR="${RUN_DIR}/lp_graph"

TMPDIR="${IOPS_ROOT}/tmp/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
PYTHONPYCACHEPREFIX="${RUN_DIR}/pycache"
XDG_CACHE_HOME="${IOPS_ROOT}/cache"
HF_HOME="${XDG_CACHE_HOME}/huggingface"
HF_DATASETS_CACHE="${HF_HOME}/datasets"
HF_HUB_CACHE="${HF_HOME}/hub"
HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
CUDA_CACHE_PATH="${XDG_CACHE_HOME}/cuda"
NUMBA_CACHE_DIR="${XDG_CACHE_HOME}/numba"
MPLCONFIGDIR="${XDG_CACHE_HOME}/matplotlib"

mkdir -p \
    "${RUN_DIR}" "${TMPDIR}" "${PYTHONPYCACHEPREFIX}" \
    "${LP_GRAPH_CACHE_DIR}" "${RAW_VOCAB_PATH}" "${SAVE_TOKENIZER_DIR}" \
    "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}" "${CUDA_CACHE_PATH}" \
    "${NUMBA_CACHE_DIR}" "${MPLCONFIGDIR}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate primer-py311
cd "${RUN_DIR}"

export DATASET_ID NUM_SHARDS TRAIN_DATASET_PATH PRETOKENIZER_MODE VOCAB_SIZES
export LP_TRAINING_MODE LP_MAX_TOKEN_BYTES LP_NUM_GPUS NUM_PROC BATCH_SIZE
export PENTLY_RHO VOCAB_UTILISATION_WEIGHT RAW_VOCAB_PATH SAVE_TOKENIZER_DIR
export LP_GRAPH_CACHE_DIR RUN_TOKENIZER_TESTS TMPDIR PYTHONPYCACHEPREFIX
export XDG_CACHE_HOME HF_HOME HF_DATASETS_CACHE HF_HUB_CACHE HUGGINGFACE_HUB_CACHE
export CUDA_CACHE_PATH NUMBA_CACHE_DIR MPLCONFIGDIR

echo "Mode: ${LP_TRAINING_MODE}; vocabulary sizes: ${VOCAB_SIZES}"
echo "Training dataset: ${TRAIN_DATASET_PATH}"
echo "Raw vocabularies: ${RAW_VOCAB_PATH}/${LP_TRAINING_MODE}"
echo "Final tokenizers: ${SAVE_TOKENIZER_DIR}/${LP_TRAINING_MODE}"

echo "Step 1/3: preparing ClimbMix dataset if needed"
python3 -u "${REPO_DIR}/prepare_climbmix_shards.py"

echo "Step 2/3: training ${LP_TRAINING_MODE} LP vocabularies"
python3 -u "${REPO_DIR}/train_tokenizer.py"

echo "Step 3/3: rounding, exporting, and checking tokenizers"
python3 -u "${REPO_DIR}/rounding_vocabs_to_tokenizer.py"
