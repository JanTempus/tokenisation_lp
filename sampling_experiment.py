"""
Unified sampling experiment pipeline.

For each of T independent samples it:
  1. Samples SAMPLE_SIZE rows from the dataset (using a unique seed per sample)
  2. Trains the LP tokenizer for every requested vocab size
  3. Trains the BPE tokenizer for every requested vocab size
  4. Computes pairwise inter-sample Jaccard distances (same vocab size, different samples)
     for the four LP rounding schemes and for BPE, and saves the results.

All settings are read from environment variables (see run_sampling_experiment.sbatch).
"""

import json
import os

import numpy as np
from datasets import load_from_disk
from transformers import PreTrainedTokenizerFast

# ---------------------------------------------------------------------------
# Configuration (read before any imports that consume env-vars at load time)
# ---------------------------------------------------------------------------
NAME = os.environ.get("NAME")
if not NAME:
    raise ValueError("NAME env var is required. Submit with: NAME=myexp sbatch run_sampling_experiment.sbatch")

T = int(os.environ.get("T", "5"))
SAMPLE_SIZE = int(os.environ.get("SAMPLE_SIZE", "60000"))
VOCAB_SIZES = [int(v) for v in os.environ.get("VOCAB_SIZES", "32768").split(",") if v.strip()]
SEED_BASE = int(os.environ.get("SEED_BASE", "42"))
SOURCE = os.environ.get("SOURCE", "finewebedu").strip().lower()
NUM_PROC = int(os.environ.get("NUM_PROC", "16"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "10000"))

EXP_DIR = f"experiment_{NAME}"

# ---------------------------------------------------------------------------
# Imports that read env-vars at module load time (must come after env is set)
# ---------------------------------------------------------------------------
# train_tokenizer reads PRETOKENIZER_MODE when the module is first imported.
from train_tokenizer import (  # noqa: E402
    get_unique_chars_batch,
    pretokenizer as lp_pretokenizer,
    train_lp_tokenizer,
)
from bpe_tokenizer.bpe_tokenizer import train_bpe_tokenizer  # noqa: E402
from jaccard_distances_sampling import (  # noqa: E402
    jaccard_distance,
    jaccard_distance_different_rounding,
)


# ---------------------------------------------------------------------------
# Step 1 – Sample T independent datasets
# ---------------------------------------------------------------------------
def step1_sample_datasets():
    from datasets import load_dataset
    from sample_tokenizer_data import (
        discover_parquet_files,
        sample_dataset,
        save_sampling_outputs,
    )

    # Check if all samples already exist on disk.
    all_dirs = [os.path.join(EXP_DIR, "samples", f"sample_{i}") for i in range(T)]
    missing = [i for i, d in enumerate(all_dirs) if not os.path.exists(d)]

    if missing:
        if SOURCE == "finewebedu":
            # Load the full dataset once; shuffle+select is fast on cached Arrow data.
            print(f"Loading pietrolesci/finewebedu-20B (once for all {len(missing)} missing samples)")
            full_ds = load_dataset("pietrolesci/finewebedu-20B", split="train")
            full_ds = full_ds.select_columns(["text"])
            for i in missing:
                seed = SEED_BASE + i
                print(f"[Sample {i}] shuffle(seed={seed}).select({SAMPLE_SIZE})")
                sample = full_ds.shuffle(seed=seed).select(range(SAMPLE_SIZE))
                save_sampling_outputs(
                    all_dirs[i], sample,
                    source_counts={"finewebedu20B": SAMPLE_SIZE},
                    sampling_manifest=[{"source": "finewebedu20B", "rows": SAMPLE_SIZE}],
                    target_rows=SAMPLE_SIZE,
                    seed=seed,
                )
        elif SOURCE == "parquet":
            DATASET_BASE_DIR = os.environ.get(
                "TOKENIZER_DATASET_BASE",
                "/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/tokenizer_training_dataset",
            )
            SOURCE_DIRS = ["fineweb2", "fineweb", "megamath", "infimath", "finemath", "starcoder"]
            SOURCE_TEXT_COLUMNS = {
                "fineweb2": "text", "fineweb": "text", "megamath": "text",
                "infimath": "text", "finemath": "text", "starcoder": "content",
            }
            source_to_files = discover_parquet_files(DATASET_BASE_DIR, SOURCE_DIRS)
            for i in missing:
                seed = SEED_BASE + i
                print(f"[Sample {i}] Sampling {SAMPLE_SIZE} parquet rows with seed={seed}")
                dataset, source_counts, manifest = sample_dataset(
                    source_to_files, SAMPLE_SIZE, seed, SOURCE_TEXT_COLUMNS
                )
                save_sampling_outputs(all_dirs[i], dataset, source_counts, manifest, SAMPLE_SIZE, seed)
        else:
            raise ValueError(f"Unknown SOURCE='{SOURCE}'. Expected: finewebedu, parquet")

    samples = []
    for i in range(T):
        dataset = load_from_disk(all_dirs[i])
        print(f"[Sample {i}] Loaded {len(dataset)} rows")
        samples.append(dataset)
    return samples


# ---------------------------------------------------------------------------
# Step 2 – Train LP tokenizers
# ---------------------------------------------------------------------------
def step2_train_lp(samples):
    for i, dataset in enumerate(samples):
        lp_dir = os.path.join(EXP_DIR, "lp_raw", f"sample_{i}")

        # Skip this sample entirely if all vocab sizes are already done
        missing_sizes = [
            vs for vs in VOCAB_SIZES
            if not os.path.exists(os.path.join(lp_dir, f"lp_tokens_{vs}.pkl"))
        ]
        if not missing_sizes:
            print(f"[LP Sample {i}] All vocab sizes already exist, skipping.")
            continue

        print(f"\n[LP Sample {i}] Computing unique chars (missing vocab sizes: {missing_sizes})")
        char_chunks = dataset.map(
            get_unique_chars_batch,
            batched=True,
            batch_size=BATCH_SIZE,
            num_proc=NUM_PROC,
            remove_columns=dataset.column_names,
            desc=f"Unique chars sample {i}",
        )
        unique_chars = sorted(set().union(*map(set, char_chunks["unique_chars"])))

        for vs in missing_sizes:
            print(f"[LP Sample {i} vocab={vs}] Training")
            train_lp_tokenizer(dataset, unique_chars, vs, lp_dir, lp_pretokenizer)
            print(f"[LP Sample {i} vocab={vs}] Saved to {lp_dir}/lp_tokens_{vs}.pkl")


# ---------------------------------------------------------------------------
# Step 3 – Train BPE tokenizers
# ---------------------------------------------------------------------------
def step3_train_bpe(samples):
    for i, dataset in enumerate(samples):
        bpe_dir = os.path.join(EXP_DIR, "bpe", f"sample_{i}")
        for vs in VOCAB_SIZES:
            out_path = os.path.join(bpe_dir, f"bpe_{vs}")
            if os.path.exists(out_path):
                print(f"[BPE Sample {i} vocab={vs}] Already exists, skipping.")
                continue
            print(f"[BPE Sample {i} vocab={vs}] Training")
            train_bpe_tokenizer(vs, dataset, bpe_dir)
            print(f"[BPE Sample {i} vocab={vs}] Saved to {out_path}")


# ---------------------------------------------------------------------------
# Step 4 – Compute pairwise inter-sample Jaccard distances
# ---------------------------------------------------------------------------
def step4_jaccard():
    lp_keys = ["all_ones", "det", "bias", "prob"]
    results = {}

    for vs in VOCAB_SIZES:
        print(f"\n[Jaccard vocab={vs}]")
        results[vs] = {}

        # --- LP rounding schemes ---
        token_sets = []
        for i in range(T):
            pkl_path = os.path.join(EXP_DIR, "lp_raw", f"sample_{i}", f"lp_tokens_{vs}.pkl")
            token_sets.append(jaccard_distance_different_rounding(vs, pkl_path))

        for key in lp_keys:
            mat = np.zeros((T, T))
            for i in range(T):
                for j in range(i + 1, T):
                    d = jaccard_distance(token_sets[i][key], token_sets[j][key])
                    mat[i][j] = d
                    mat[j][i] = d
            results[vs][f"lp_{key}"] = mat.tolist()
            upper = mat[np.triu_indices(T, k=1)]
            print(f"  LP {key}: mean={upper.mean():.4f}  min={upper.min():.4f}  max={upper.max():.4f}")

        # --- BPE ---
        bpe_vocabs = []
        for i in range(T):
            tok_path = os.path.join(EXP_DIR, "bpe", f"sample_{i}", f"bpe_{vs}")
            tok = PreTrainedTokenizerFast.from_pretrained(tok_path)
            bpe_vocabs.append(set(tok.get_vocab().keys()))

        mat = np.zeros((T, T))
        for i in range(T):
            for j in range(i + 1, T):
                d = jaccard_distance(list(bpe_vocabs[i]), list(bpe_vocabs[j]))
                mat[i][j] = d
                mat[j][i] = d
        results[vs]["bpe"] = mat.tolist()
        upper = mat[np.triu_indices(T, k=1)]
        print(f"  BPE:    mean={upper.mean():.4f}  min={upper.min():.4f}  max={upper.max():.4f}")

    # --- Persist results ---
    jaccard_dir = os.path.join(EXP_DIR, "jaccard")
    os.makedirs(jaccard_dir, exist_ok=True)

    json_path = os.path.join(jaccard_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    print(f"\nSaved JSON results to {json_path}")

    txt_path = os.path.join(jaccard_dir, "results.txt")
    with open(txt_path, "w") as f:
        f.write(f"Experiment : {NAME}\n")
        f.write(f"T          : {T} samples\n")
        f.write(f"sample_size: {SAMPLE_SIZE}\n")
        f.write(f"vocab_sizes: {VOCAB_SIZES}\n")
        f.write(f"seed_base  : {SEED_BASE}\n\n")
        for vs in VOCAB_SIZES:
            f.write(f"=== Vocab size {vs} ===\n")
            for key in lp_keys:
                mat = np.array(results[vs][f"lp_{key}"])
                upper = mat[np.triu_indices(T, k=1)]
                f.write(f"  LP {key:8s}: mean={upper.mean():.4f}  min={upper.min():.4f}  max={upper.max():.4f}\n")
                f.write(f"    {np.array2string(mat, precision=4)}\n")
            mat = np.array(results[vs]["bpe"])
            upper = mat[np.triu_indices(T, k=1)]
            f.write(f"  BPE       : mean={upper.mean():.4f}  min={upper.min():.4f}  max={upper.max():.4f}\n")
            f.write(f"    {np.array2string(mat, precision=4)}\n\n")
    print(f"Saved human-readable summary to {txt_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"=== Sampling Experiment: {NAME} ===")
    print(f"T={T}, SAMPLE_SIZE={SAMPLE_SIZE}, VOCAB_SIZES={VOCAB_SIZES}, SEED_BASE={SEED_BASE}, SOURCE={SOURCE}")
    print(f"Output directory: {EXP_DIR}\n")

    print("--- Step 1/4: Sampling datasets ---")
    samples = step1_sample_datasets()

    print("\n--- Step 2/4: Training LP tokenizers ---")
    step2_train_lp(samples)

    print("\n--- Step 3/4: Training BPE tokenizers ---")
    step3_train_bpe(samples)

    print("\n--- Step 4/4: Computing Jaccard distances ---")
    step4_jaccard()

    print("\nDone.")
