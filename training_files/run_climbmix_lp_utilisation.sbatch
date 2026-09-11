#!/bin/bash
#SBATCH -A a0229
#SBATCH -p normal
#SBATCH --job-name=climbmix-lp-utilisation
#SBATCH --gres=gpu:1
#SBATCH -t 03:30:00
#SBATCH --cpus-per-task=32
#SBATCH --output=/capstor/store/cscs/swissai/a0229/jtempus/tokenisation_lp/logs/slurm-%j.out
#SBATCH --error=/capstor/store/cscs/swissai/a0229/jtempus/tokenisation_lp/logs/slurm-%j.err


set -euo pipefail

# Submit: sbatch training_files/run_climbmix_lp_utilisation.sbatch
# Defaults resume export of every saved size from util_bias, weight 0.01.
# Set EXPORT_ONLY=0 below to run dataset preparation and LP training again.
# Override weight: sbatch --export=ALL,VOCAB_UTILISATION_WEIGHT=10 training_files/run_climbmix_lp_utilisation.sbatch
# Resume export for every saved vocabulary in the util_bias run:
# sbatch --export=ALL,EXPORT_ONLY=1,NAME=util_bias,VOCAB_UTILISATION_WEIGHT=0.01 training_files/run_climbmix_lp_utilisation.sbatch
# To restrict export, set VOCAB_SIZES in the export-only branch below.
# User-tunable settings
NAME="${NAME:-util_bias}"
REPO_DIR="${REPO_DIR:-/capstor/store/cscs/swissai/a0229/jtempus/tokenisation_lp}"
IOPS_ROOT="${IOPS_ROOT:-/iopsstor/scratch/cscs/jtempus/tokenisation_lp}"
NUM_SHARDS="${NUM_SHARDS:-7}"
DATASET_ID="${DATASET_ID:-karpathy/climbmix-400b-shuffle}"
# Unused-fraction penalty: lambda * sum_c (t_c - U_c / N_c).
VOCAB_UTILISATION_WEIGHT="${VOCAB_UTILISATION_WEIGHT:-0.01}"
LP_NUM_GPUS=1
PRETOKENIZER_MODE="${PRETOKENIZER_MODE:-nanochat}"
EXPORT_ONLY="${EXPORT_ONLY:-1}"
if [[ "${EXPORT_ONLY}" == "1" ]]; then
    # Empty means discover every saved vocabulary, including sizes from earlier jobs.
    VOCAB_SIZES=""
else
    VOCAB_SIZES="${VOCAB_SIZES:-8192,16384,32768,65536,131072,262144}"
fi
NUM_PROC="${NUM_PROC:-32}"
BATCH_SIZE="${BATCH_SIZE:-10000}"
RUN_TOKENIZER_TESTS="${RUN_TOKENIZER_TESTS:-1}"
BYTE_TEST_BEHAVIOR="${BYTE_TEST_BEHAVIOR:-not_all_unk}"

if [[ "${EXPORT_ONLY}" != "0" && "${EXPORT_ONLY}" != "1" ]]; then
    echo "ERROR: EXPORT_ONLY must be 0 (full pipeline) or 1 (round/export saved vocabularies)." >&2
    exit 1
fi

if [[ -z "${NAME}" ]]; then
    echo "ERROR: NAME is required. Submit with --export=NAME=<experiment>" >&2
    exit 1
fi

DATASET_TAG="${NAME}_climbmix400b_s${NUM_SHARDS}"
EXPERIMENT_TAG="${DATASET_TAG}_unused_fraction_w${VOCAB_UTILISATION_WEIGHT}"
RUN_DIR="${IOPS_ROOT}/runs/${EXPERIMENT_TAG}_${SLURM_JOB_ID}"
OUTPUT_DATASET_DIR="${OUTPUT_DATASET_DIR:-${IOPS_ROOT}/datasets/${DATASET_TAG}}"
TRAIN_DATASET_PATH="${TRAIN_DATASET_PATH:-${OUTPUT_DATASET_DIR}}"
RAW_VOCAB_PATH="${RAW_VOCAB_PATH:-${IOPS_ROOT}/raw_vocabs/${EXPERIMENT_TAG}}"
LP_GRAPH_CACHE_DIR="${LP_GRAPH_CACHE_DIR:-${RUN_DIR}/lp_graph}"
SAVE_TOKENIZER_DIR="${SAVE_TOKENIZER_DIR:-${REPO_DIR}/rounded_tokenizers/${EXPERIMENT_TAG}}"

TMPDIR="/iopsstor/scratch/cscs/jtempus/tmp/${SLURM_JOB_ID}"
TMP="${TMPDIR}"
TEMP="${TMPDIR}"
PYTHONPYCACHEPREFIX="${RUN_DIR}/pycache"
XDG_CACHE_HOME="${IOPS_ROOT}/cache"
HF_HOME="${XDG_CACHE_HOME}/huggingface"
HF_DATASETS_CACHE="${HF_HOME}/datasets"
HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
CUDA_CACHE_PATH="${XDG_CACHE_HOME}/cuda"
NUMBA_CACHE_DIR="${XDG_CACHE_HOME}/numba"
MPLCONFIGDIR="${XDG_CACHE_HOME}/matplotlib"

mkdir -p \
    "${RUN_DIR}" \
    "${TMPDIR}" \
    "${PYTHONPYCACHEPREFIX}" \
    "${LP_GRAPH_CACHE_DIR}" \
    "${IOPS_ROOT}/datasets" \
    "${IOPS_ROOT}/raw_vocabs" \
    "${HF_DATASETS_CACHE}" \
    "${HUGGINGFACE_HUB_CACHE}" \
    "${CUDA_CACHE_PATH}" \
    "${NUMBA_CACHE_DIR}" \
    "${MPLCONFIGDIR}"

source /users/jtempus/miniconda3/etc/profile.d/conda.sh
conda activate primer-py311
cd "${RUN_DIR}"

export NAME
export REPO_DIR
export IOPS_ROOT
export RUN_DIR
export NUM_SHARDS
export DATASET_ID
export VOCAB_UTILISATION_WEIGHT
export LP_NUM_GPUS
export OUTPUT_DATASET_DIR
export TRAIN_DATASET_PATH
export PRETOKENIZER_MODE
export VOCAB_SIZES
export NUM_PROC
export BATCH_SIZE
export LP_GRAPH_CACHE_DIR
export TMPDIR
export TMP
export TEMP
export PYTHONPYCACHEPREFIX
export XDG_CACHE_HOME
export HF_HOME
export HF_DATASETS_CACHE
export HUGGINGFACE_HUB_CACHE
export CUDA_CACHE_PATH
export NUMBA_CACHE_DIR
export MPLCONFIGDIR
export RAW_VOCAB_PATH
export SAVE_TOKENIZER_DIR
export RUN_TOKENIZER_TESTS
export BYTE_TEST_BEHAVIOR

echo "Run tag:              ${EXPERIMENT_TAG}"
echo "ClimbMix dataset:     ${DATASET_ID}"
echo "Number of shards:     ${NUM_SHARDS}"
echo "Unused-fraction weight: ${VOCAB_UTILISATION_WEIGHT}"
echo "Repository:           ${REPO_DIR}"
echo "Job working dir:      ${RUN_DIR}"
echo "Training dataset:     ${TRAIN_DATASET_PATH}"
echo "Raw vocabulary:       ${RAW_VOCAB_PATH}"
echo "LP graph cache:       ${LP_GRAPH_CACHE_DIR}"
echo "Final tokenizers:     ${SAVE_TOKENIZER_DIR}"
echo "Map workers:          ${NUM_PROC}"
echo "Map batch size:       ${BATCH_SIZE}"
echo "Vocabulary sizes:     ${VOCAB_SIZES:-all saved sizes}"
echo "Export only:          ${EXPORT_ONLY}"

if [[ "${EXPORT_ONLY}" == "1" ]]; then
    echo "Skipping steps 1/3 and 2/3; exporting saved vocabularies from ${RAW_VOCAB_PATH}"
else
    echo "Step 1/3: preparing first ${NUM_SHARDS} ClimbMix shards (reuse if dataset exists)"
    python3 -u "${REPO_DIR}/prepare_climbmix_shards.py"

    echo "Step 2/3: training LP vocab(s) -> ${RAW_VOCAB_PATH}"
    python3 -u "${REPO_DIR}/train_tokenizer.py"
fi

echo "Step 3/3: rounding/exporting LP tokenizer(s) -> ${SAVE_TOKENIZER_DIR}"
if [[ "${EXPORT_ONLY}" == "1" && -z "${VOCAB_SIZES}" ]]; then
    unset VOCAB_SIZES
fi
python3 -u "${REPO_DIR}/rounding_vocabs_to_tokenizer.py"
