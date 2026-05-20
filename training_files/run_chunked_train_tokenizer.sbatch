#!/bin/bash
#SBATCH -A a139
#SBATCH -p normal
#SBATCH --gres=gpu:4
#SBATCH -t 24:00:00

set -euo pipefail

# User-tunable settings
REPO_DIR="/iopsstor/scratch/cscs/jtempus/tokenisation_lp"
TOKENIZER_DATASET_BASE="${TOKENIZER_DATASET_BASE:-/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/tokenizer_training_dataset}"
CHUNK_SIZE="${CHUNK_SIZE:-50000}"
CHUNK_OUTPUT_DIR="${CHUNK_OUTPUT_DIR:-dataset_chunks}"
CHUNK_VOCAB_DIR="${CHUNK_VOCAB_DIR:-chunk_vocabs}"
PRETOKENIZER_MODE="${PRETOKENIZER_MODE:-custom}"
VOCAB_SIZES="${VOCAB_SIZES:-131072}"
RAW_VOCAB_PATH="${RAW_VOCAB_PATH:-rounding_vocabs_apertus_2}"
SAVE_TOKENIZER_DIR="${SAVE_TOKENIZER_DIR:-rounded_tokenizers_apertus_2}"
RUN_TOKENIZER_TESTS="${RUN_TOKENIZER_TESTS:-1}"
BYTE_TEST_BEHAVIOR="${BYTE_TEST_BEHAVIOR:-not_all_unk}"
NUM_PROC="${NUM_PROC:-4}"
BATCH_SIZE="${BATCH_SIZE:-10000}"

# IMPORTANT: load conda manually
source /users/jtempus/miniconda3/etc/profile.d/conda.sh

# activate env
conda activate primer-py311

# go to repo
cd "${REPO_DIR}"

export TOKENIZER_DATASET_BASE
export CHUNK_SIZE
export CHUNK_OUTPUT_DIR
export CHUNK_VOCAB_DIR
export PRETOKENIZER_MODE
export VOCAB_SIZES
export RAW_VOCAB_PATH
export SAVE_TOKENIZER_DIR
export RUN_TOKENIZER_TESTS
export BYTE_TEST_BEHAVIOR
export NUM_PROC
export BATCH_SIZE

# Step 1: Split full dataset into 50k-row chunks
echo "Step 1/4: chunking dataset from ${TOKENIZER_DATASET_BASE} -> ${CHUNK_OUTPUT_DIR} (chunk_size=${CHUNK_SIZE})"
python3 -u chunk_dataset.py

MANIFEST="${CHUNK_OUTPUT_DIR}/manifest.json"
NUM_CHUNKS=$(python3 -c "import json; d=json.load(open('${MANIFEST}')); print(d['num_chunks'])")
echo "  Created ${NUM_CHUNKS} chunks"

# Step 2: Train LP tokenizer on each chunk, 4 at a time (one per GPU)
echo "Step 2/4: training LP vocab on each chunk (4 chunks in parallel)"

CHUNK_PATHS=$(python3 -c "
import json
d = json.load(open('${MANIFEST}'))
print('\n'.join(d['chunks']))
")

GPU_ID=0
PIDS=()

while IFS= read -r CHUNK_PATH; do
    CHUNK_NAME=$(basename "${CHUNK_PATH}")
    CHUNK_SAVE_DIR="${CHUNK_VOCAB_DIR}/${CHUNK_NAME}"

    echo "  Launching training on ${CHUNK_NAME} (GPU ${GPU_ID})"
    CUDA_VISIBLE_DEVICES=${GPU_ID} \
    TRAIN_DATASET_PATH="${CHUNK_PATH}" \
    RAW_VOCAB_PATH="${CHUNK_SAVE_DIR}" \
        python3 -u train_tokenizer.py &

    PIDS+=($!)
    GPU_ID=$(( (GPU_ID + 1) % 4 ))

    # Wait when all 4 GPUs are busy
    if [ ${#PIDS[@]} -eq 4 ]; then
        echo "  Waiting for batch of 4 to finish..."
        for PID in "${PIDS[@]}"; do
            wait "${PID}"
        done
        PIDS=()
    fi
done <<< "${CHUNK_PATHS}"

# Wait for any remaining jobs (last partial batch)
if [ ${#PIDS[@]} -gt 0 ]; then
    echo "  Waiting for final batch (${#PIDS[@]} job(s))..."
    for PID in "${PIDS[@]}"; do
        wait "${PID}"
    done
fi

echo "  All chunk training jobs completed"

# Step 3: Merge unrounded vocabs (sum lp_value per token across chunks)
echo "Step 3/4: merging chunk vocabs from ${CHUNK_VOCAB_DIR} -> ${RAW_VOCAB_PATH}"
CHUNK_VOCAB_DIR="${CHUNK_VOCAB_DIR}" \
RAW_VOCAB_PATH="${RAW_VOCAB_PATH}" \
VOCAB_SIZES="${VOCAB_SIZES}" \
    python3 -u merge_vocabs.py

# Step 4: Round and export tokenizers
echo "Step 4/4: rounding/exporting tokenizers -> ${SAVE_TOKENIZER_DIR}"
python3 -u rounding_vocabs_to_tokenizer.py
