#!/bin/bash
#SBATCH -A a0229
#SBATCH -p debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH -t 01:20:00
#SBATCH --output=/capstor/store/cscs/swissai/a0229/jtempus/tokenisation_lp/logs/multilingual_baselines_%j.out
#SBATCH --error=/capstor/store/cscs/swissai/a0229/jtempus/tokenisation_lp/logs/multilingual_baselines_%j.err

set -euo pipefail

# User-tunable settings
NAME="${NAME:-Multilingual}"
REPO_DIR="${REPO_DIR:-/capstor/store/cscs/swissai/a0229/jtempus/tokenisation_lp}"
IOPS_ROOT="${IOPS_ROOT:-/iopsstor/scratch/cscs/jtempus/tokenisation_lp}"
TMP_ROOT="${TMP_ROOT:-/iopsstor/scratch/cscs/jtempus/tmp}"
CONDA_ROOT="${CONDA_ROOT:-/users/jtempus/miniconda3}"
TARGET_ROWS="${TARGET_ROWS:-8000}"
PRETOKENIZER_MODE="${PRETOKENIZER_MODE:-nanochat}"
VOCAB_SIZES="${VOCAB_SIZES:-131072,196608,262144}"
RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-32}}"
TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

if [[ -z "${NAME}" ]]; then
    echo "ERROR: NAME is required. Submit with --export=NAME=<experiment>" >&2
    exit 1
fi

DATASET_TAG="${NAME}_n${TARGET_ROWS}"
RUN_DIR="${IOPS_ROOT}/runs/${DATASET_TAG}_baselines_${SLURM_JOB_ID}"
OUTPUT_DATASET_DIR="${OUTPUT_DATASET_DIR:-${IOPS_ROOT}/datasets/${DATASET_TAG}}"
TRAIN_DATASET_PATH="${TRAIN_DATASET_PATH:-${OUTPUT_DATASET_DIR}}"
BPE_SAVE_DIR="${BPE_SAVE_DIR:-${SAVE_DIR:-${REPO_DIR}/bpe_tokenizers/${DATASET_TAG}}}"
UNIGRAM_SAVE_DIR="${UNIGRAM_SAVE_DIR:-${REPO_DIR}/unigram_tokenizers/${DATASET_TAG}}"

TMPDIR="${TMP_ROOT}/${SLURM_JOB_ID}"
TMP="${TMPDIR}"
TEMP="${TMPDIR}"
PYTHONPYCACHEPREFIX="${RUN_DIR}/pycache"
XDG_CACHE_HOME="${IOPS_ROOT}/cache"
HF_HOME="${XDG_CACHE_HOME}/huggingface"
HF_DATASETS_CACHE="${HF_HOME}/datasets"
HF_HUB_CACHE="${HF_HOME}/hub"
HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"

mkdir -p \
    "${RUN_DIR}" \
    "${TMPDIR}" \
    "${PYTHONPYCACHEPREFIX}" \
    "${BPE_SAVE_DIR}" \
    "${UNIGRAM_SAVE_DIR}" \
    "${HF_DATASETS_CACHE}" \
    "${HF_HUB_CACHE}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate primer-py311
cd "${RUN_DIR}"

export NAME REPO_DIR IOPS_ROOT TARGET_ROWS OUTPUT_DATASET_DIR
export TRAIN_DATASET_PATH PRETOKENIZER_MODE VOCAB_SIZES
export TMPDIR TMP TEMP PYTHONPYCACHEPREFIX XDG_CACHE_HOME
export HF_HOME HF_DATASETS_CACHE HF_HUB_CACHE HUGGINGFACE_HUB_CACHE
export RAYON_NUM_THREADS TOKENIZERS_PARALLELISM

echo "Run tag:           ${DATASET_TAG}"
echo "Job working dir:   ${RUN_DIR}"
echo "Training dataset:  ${TRAIN_DATASET_PATH}"
echo "BPE output:        ${BPE_SAVE_DIR}"
echo "Unigram output:    ${UNIGRAM_SAVE_DIR}"
echo "Vocabulary sizes:  ${VOCAB_SIZES}"
echo "Tokenizer threads: ${RAYON_NUM_THREADS}"

if [[ -f "${TRAIN_DATASET_PATH}/sampling_manifest.json" ]]; then
    echo "Step 1/3: reusing existing dataset at ${TRAIN_DATASET_PATH}"
elif [[ "${TRAIN_DATASET_PATH}" != "${OUTPUT_DATASET_DIR}" ]]; then
    echo "ERROR: overridden TRAIN_DATASET_PATH has no sampling manifest: ${TRAIN_DATASET_PATH}" >&2
    exit 1
else
    echo "Step 1/3: downloading ordered FineWeb/FineWeb2 prefixes -> ${OUTPUT_DATASET_DIR}"
    python3 -u "${REPO_DIR}/sample_tokenizer_data.py" \
        --source fineweb2 \
        --name "${NAME}" \
        --target-rows "${TARGET_ROWS}" \
        --output-dataset-dir "${OUTPUT_DATASET_DIR}"
fi

echo "Step 2/3: training BPE tokenizers"
TOKENIZER_TYPE=bpe SAVE_DIR="${BPE_SAVE_DIR}" \
    python3 -u "${REPO_DIR}/hf_baseline_tokenizers/train_sampled.py"

echo "Step 3/3: training Unigram tokenizers"
TOKENIZER_TYPE=unigram SAVE_DIR="${UNIGRAM_SAVE_DIR}" \
    python3 -u "${REPO_DIR}/hf_baseline_tokenizers/train_sampled.py"
