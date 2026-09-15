#!/bin/bash
#SBATCH -A a0229
#SBATCH -p normal
#SBATCH --job-name=climbmix-lp-morphology
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH -t 08:30:00
#SBATCH --output=/iopsstor/scratch/cscs/jtempus/morphology-lp-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/jtempus/morphology-lp-%j.err

set -euo pipefail

# Submit: sbatch training_files/run_climbmix_lp_morphology.sbatch
# Override the shared CELEX/Unicode penalty weight:
# sbatch --export=ALL,PENTLY_RHO=1 training_files/run_climbmix_lp_morphology.sbatch
# CELEX_DIR must contain english/eml/eml.cd and english/emw/emw.cd.
NAME="${NAME:-morphology}"
REPO_DIR="${REPO_DIR:-/capstor/store/cscs/swissai/a0229/jtempus/tokenisation_lp}"
IOPS_ROOT="${IOPS_ROOT:-/iopsstor/scratch/cscs/jtempus/tokenisation_lp}"
CONDA_ROOT="${CONDA_ROOT:-/users/jtempus/miniconda3}"
DATASET_ID="${DATASET_ID:-karpathy/climbmix-400b-shuffle}"
NUM_SHARDS="${NUM_SHARDS:-7}"
PRETOKENIZER_MODE="${PRETOKENIZER_MODE:-nanochat}"
VOCAB_SIZES="${VOCAB_SIZES:-8192,16384,32768,65536,131072,262144}"
PENTLY_RHO="${PENTLY_RHO:-0.5}"
CELEX_DIR="${CELEX_DIR:-${REPO_DIR}/CELEX_V2}"
VOCAB_UTILISATION_WEIGHT="${VOCAB_UTILISATION_WEIGHT:-0}"
LP_NUM_GPUS=1
NUM_PROC="${NUM_PROC:-${SLURM_CPUS_PER_TASK:-128}}"
BATCH_SIZE="${BATCH_SIZE:-10000}"
RUN_TOKENIZER_TESTS="${RUN_TOKENIZER_TESTS:-1}"
BYTE_TEST_BEHAVIOR="${BYTE_TEST_BEHAVIOR:-not_all_unk}"

DATASET_TAG="climbmix400b_s${NUM_SHARDS}"
EXPERIMENT_TAG="${NAME}_${PRETOKENIZER_MODE}_${DATASET_TAG}_rho${PENTLY_RHO}"
RUN_DIR="${IOPS_ROOT}/runs/${EXPERIMENT_TAG}_${SLURM_JOB_ID}"
TRAIN_DATASET_PATH="${TRAIN_DATASET_PATH:-${IOPS_ROOT}/datasets/${DATASET_TAG}}"
RAW_VOCAB_PATH="${RAW_VOCAB_PATH:-${IOPS_ROOT}/raw_vocabs/${EXPERIMENT_TAG}}"
SAVE_TOKENIZER_DIR="${SAVE_TOKENIZER_DIR:-${REPO_DIR}/rounded_tokenizers/${EXPERIMENT_TAG}}"
LP_GRAPH_CACHE_DIR="${LP_GRAPH_CACHE_DIR:-${RUN_DIR}/lp_graph}"

TMPDIR="${RUN_DIR}/tmp"
TMP="${TMPDIR}"
TEMP="${TMPDIR}"
PYTHONPYCACHEPREFIX="${RUN_DIR}/pycache"
XDG_CACHE_HOME="${IOPS_ROOT}/cache"
HF_HOME="${XDG_CACHE_HOME}/huggingface"
HF_DATASETS_CACHE="${HF_HOME}/datasets"
HF_HUB_CACHE="${HF_HOME}/hub"
HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
CUDA_CACHE_PATH="${XDG_CACHE_HOME}/cuda"
NUMBA_CACHE_DIR="${XDG_CACHE_HOME}/numba"
MPLCONFIGDIR="${XDG_CACHE_HOME}/matplotlib"

# Check morphology data before preparing the training corpus.
for CELEX_FILE in "${CELEX_DIR}/english/eml/eml.cd" "${CELEX_DIR}/english/emw/emw.cd"; do
    if [[ ! -r "${CELEX_FILE}" ]]; then
        echo "ERROR: CELEX file is missing or unreadable: ${CELEX_FILE}" >&2
        exit 1
    fi
done

mkdir -p \
    "${RUN_DIR}" "${TMPDIR}" "${PYTHONPYCACHEPREFIX}" \
    "${LP_GRAPH_CACHE_DIR}" "${RAW_VOCAB_PATH}" "${SAVE_TOKENIZER_DIR}" \
    "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}" "${CUDA_CACHE_PATH}" \
    "${NUMBA_CACHE_DIR}" "${MPLCONFIGDIR}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate primer-py311
cd "${RUN_DIR}"

export DATASET_ID NUM_SHARDS TRAIN_DATASET_PATH PRETOKENIZER_MODE VOCAB_SIZES
export PENTLY_RHO CELEX_DIR VOCAB_UTILISATION_WEIGHT LP_NUM_GPUS NUM_PROC BATCH_SIZE
export RAW_VOCAB_PATH SAVE_TOKENIZER_DIR LP_GRAPH_CACHE_DIR
export RUN_TOKENIZER_TESTS BYTE_TEST_BEHAVIOR
export TMPDIR TMP TEMP PYTHONPYCACHEPREFIX XDG_CACHE_HOME
export HF_HOME HF_DATASETS_CACHE HF_HUB_CACHE HUGGINGFACE_HUB_CACHE
export CUDA_CACHE_PATH NUMBA_CACHE_DIR MPLCONFIGDIR

echo "Run tag:          ${EXPERIMENT_TAG}"
echo "Penalty rho:      ${PENTLY_RHO} (CELEX + Unicode)"
echo "CELEX directory:  ${CELEX_DIR}"
echo "Training dataset: ${TRAIN_DATASET_PATH}"
echo "Vocabulary sizes: ${VOCAB_SIZES}"
echo "Raw vocabularies: ${RAW_VOCAB_PATH}"
echo "Final tokenizers: ${SAVE_TOKENIZER_DIR}"

echo "Step 1/3: preparing first ${NUM_SHARDS} ClimbMix shards (reuse if dataset exists)"
python3 -u "${REPO_DIR}/prepare_climbmix_shards.py"

echo "Step 2/3: training morphologically aligned LP vocabularies"
python3 -u "${REPO_DIR}/train_tokenizer.py"

echo "Step 3/3: rounding, exporting, and checking tokenizers"
python3 -u "${REPO_DIR}/rounding_vocabs_to_tokenizer.py"
