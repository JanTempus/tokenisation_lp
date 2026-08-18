#!/bin/bash
#SBATCH -A a0229
#SBATCH -p normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH -t 24:00:00

set -e

REPO_DIR=/capstor/store/cscs/swissai/a0229/jtempus/tokenisation_lp
TRAIN_DATASET_PATH=${REPO_DIR}/datasets/climbmix_first7
OUTPUT_ROOT=${REPO_DIR}/tokenizer_runs/nanochat_comparison
VOCAB_SIZES_CSV=8192,16384,32768,65536,131072,262144
VOCAB_SIZES=(8192 16384 32768 65536 131072 262144)

source /users/jtempus/miniconda3/etc/profile.d/conda.sh
conda activate primer-py311
cd "${REPO_DIR}"

UNIGRAM_DIR=${OUTPUT_ROOT}/unigram
FREQUENT_DIR=${OUTPUT_ROOT}/frequent
PICKY_DIR=${OUTPUT_ROOT}/picky
SHARED_DIR=${OUTPUT_ROOT}/shared
LOG_DIR=${OUTPUT_ROOT}/logs
PICKY_CORPUS_PATH=${SHARED_DIR}/picky_corpus.jsonl

mkdir -p \
    "${UNIGRAM_DIR}" \
    "${FREQUENT_DIR}" \
    "${PICKY_DIR}" \
    "${SHARED_DIR}" \
    "${LOG_DIR}"

srun \
    --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=64 --cpu-bind=cores \
    --output="${LOG_DIR}/prepare_dataset.out" \
    --error="${LOG_DIR}/prepare_dataset.err" \
    /usr/bin/time -v \
    env TRAIN_DATASET_PATH="${TRAIN_DATASET_PATH}" \
    python3 -u prepare_climbmix_shards.py

srun \
    --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=64 --cpu-bind=cores \
    --output="${LOG_DIR}/prepare_corpus.out" \
    --error="${LOG_DIR}/prepare_corpus.err" \
    /usr/bin/time -v \
    env \
        TRAIN_DATASET_PATH="${TRAIN_DATASET_PATH}" \
        PICKY_CORPUS_PATH="${PICKY_CORPUS_PATH}" \
    python3 -u training_files/prepare_picky_corpus.py

PICKY_NUM_LINES=$(wc -l < "${PICKY_CORPUS_PATH}")

launch() {
    name=$1
    cpus=$2
    shift 2
    srun \
        --exclusive --exact --nodes=1 --ntasks=1 \
        --cpus-per-task="${cpus}" --cpu-bind=cores \
        --output="${LOG_DIR}/${name}.out" \
        --error="${LOG_DIR}/${name}.err" \
        /usr/bin/time -v "$@" &
}

launch frequent_all 64 \
    env \
        TRAIN_DATASET_PATH="${TRAIN_DATASET_PATH}" \
        VOCAB_SIZES="${VOCAB_SIZES_CSV}" \
        SAVE_DIR="${FREQUENT_DIR}" \
        PRETOKENIZER_MODE=nanochat \
        NUM_PROC=64 \
        BATCH_SIZE=10000 \
        RAYON_NUM_THREADS=1 \
        TOKENIZERS_PARALLELISM=false \
        OMP_NUM_THREADS=1 \
        MKL_NUM_THREADS=1 \
        OPENBLAS_NUM_THREADS=1 \
    python3 -u hf_baseline_tokenizers/train_frequent_pretokens.py

for size in "${VOCAB_SIZES[@]}"; do
    launch "unigram_${size}" 36 \
        env \
            TRAIN_DATASET_PATH="${TRAIN_DATASET_PATH}" \
            VOCAB_SIZES="${size}" \
            SAVE_DIR="${UNIGRAM_DIR}" \
            TOKENIZER_TYPE=unigram \
            PRETOKENIZER_MODE=nanochat \
            RAYON_NUM_THREADS=36 \
            TOKENIZERS_PARALLELISM=true \
            OMP_NUM_THREADS=1 \
            MKL_NUM_THREADS=1 \
            OPENBLAS_NUM_THREADS=1 \
        python3 -u hf_baseline_tokenizers/train_sampled.py

    launch "picky_${size}" 1 \
        env \
            PICKY_CORPUS_PATH="${PICKY_CORPUS_PATH}" \
            PICKY_NUM_LINES="${PICKY_NUM_LINES}" \
            VOCAB_SIZE="${size}" \
            SAVE_DIR="${PICKY_DIR}" \
            RAYON_NUM_THREADS=1 \
            TOKENIZERS_PARALLELISM=false \
            OMP_NUM_THREADS=1 \
            MKL_NUM_THREADS=1 \
            OPENBLAS_NUM_THREADS=1 \
        python3 -u boundlessbpe/runpickybpe.py
done

wait
