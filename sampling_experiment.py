"""
Unified sampling experiment pipeline.

For each sample size in SAMPLE_SIZES and each of T independent samples it:
  1. Samples SAMPLE_SIZE rows from the dataset (using a unique seed per sample)
  2. Trains the LP tokenizer for every requested vocab size
  3. Trains the BPE tokenizer for every requested vocab size
  4. Computes pairwise inter-sample Jaccard distances (same vocab size, different samples)
     for the four LP rounding schemes and for BPE, and saves the results.

All settings are read from environment variables (see run_sampling_experiment.sbatch).
"""

import json
import os
import time

import numpy as np
from datasets import load_dataset, load_from_disk
from transformers import PreTrainedTokenizerFast

# ---------------------------------------------------------------------------
# Configuration (read before any imports that consume env-vars at load time)
# ---------------------------------------------------------------------------
def _require_env(name):
    val = os.environ.get(name)
    if not val:
        raise ValueError(f"{name} env var is required but not set.")
    return val


NAME = os.environ.get("NAME")
if not NAME:
    raise ValueError("NAME env var is required. Submit with: NAME=myexp sbatch run_sampling_experiment.sbatch")

T            = int(_require_env("T"))
SAMPLE_SIZES = [int(v) for v in _require_env("SAMPLE_SIZES").split(",") if v.strip()]
VOCAB_SIZES  = [int(v) for v in _require_env("VOCAB_SIZES").split(",") if v.strip()]
SEED_BASE    = int(_require_env("SEED_BASE"))
SOURCE       = _require_env("SOURCE").strip().lower()
NUM_PROC     = int(_require_env("NUM_PROC"))
BATCH_SIZE   = int(_require_env("BATCH_SIZE"))

EXP_DIR = f"experiment_{NAME}"

# ---------------------------------------------------------------------------
# Imports that read env-vars at module load time (must come after env is set)
# ---------------------------------------------------------------------------
# train_tokenizer reads PRETOKENIZER_MODE when the module is first imported.
from train_tokenizer import (  # noqa: E402
    PRETOKENIZER_MODE,
    get_special_tokens,
    get_unique_chars_batch,
    pretokenizer as lp_pretokenizer,
    train_lp_tokenizer,
)
from bpe_tokenizer.bpe_tokenizer import train_bpe_tokenizer  # noqa: E402
import pickle

from lp_tokenizer.lp_functions import (  # noqa: E402
    biased_rounding,
    deterministic_rounding,
    probabilistic_rounding,
)


def jaccard_distance(a, b):
    inter = len(set(a) & set(b))
    union = len(set(a) | set(b))
    return inter / union if union > 0 else 0.0


def jaccard_distance_different_rounding(vocab_size, raw_tokens_path):
    with open(raw_tokens_path, "rb") as f:
        tokens = pickle.load(f)
    n_special = len(tokens["special_tokens"])
    target = vocab_size - n_special
    det_tokens  = deterministic_rounding(tokens["possible_tokens"], tokens["unique_chars"], target)
    bias_tokens = biased_rounding(tokens["possible_tokens"], tokens["unique_chars"], target)
    prob_tokens = probabilistic_rounding(tokens["possible_tokens"], tokens["unique_chars"], target)
    ones_tokens = [t.token for t in tokens["possible_tokens"] if t.lp_value >= 0.99]
    det_tokens  = list(set(det_tokens  + tokens["special_tokens"]))
    bias_tokens = list(set(bias_tokens + tokens["special_tokens"]))
    prob_tokens = list(set(prob_tokens + tokens["special_tokens"]))
    ones_tokens = list(set(ones_tokens + tokens["unique_chars"] + tokens["special_tokens"]))
    return {"all_ones": ones_tokens, "det": det_tokens, "bias": bias_tokens, "prob": prob_tokens}


# ---------------------------------------------------------------------------
# Step 1 – Sample T independent datasets
# ---------------------------------------------------------------------------
def step1_sample_datasets(sample_size, ss_dir, full_ds=None):
    from sample_tokenizer_data import (
        discover_parquet_files,
        sample_dataset,
        save_sampling_outputs,
    )

    all_dirs = [os.path.join(ss_dir, "samples", f"sample_{i}") for i in range(T)]

    if SOURCE == "finewebedu":
        for i in range(T):
            seed = SEED_BASE + i
            print(f"[Sample {i}] shuffle(seed={seed}).select({sample_size})")
            sample = full_ds.shuffle(seed=seed).select(range(sample_size))
            save_sampling_outputs(
                all_dirs[i], sample,
                source_counts={"finewebedu20B": sample_size},
                sampling_manifest=[{"source": "finewebedu20B", "rows": sample_size}],
                target_rows=sample_size,
                seed=seed,
            )
    elif SOURCE == "parquet":
        DATASET_BASE_DIR = _require_env("TOKENIZER_DATASET_BASE")
        SOURCE_DIRS = ["fineweb2", "fineweb", "megamath", "infimath", "finemath", "starcoder"]
        SOURCE_TEXT_COLUMNS = {
            "fineweb2": "text", "fineweb": "text", "megamath": "text",
            "infimath": "text", "finemath": "text", "starcoder": "content",
        }
        source_to_files = discover_parquet_files(DATASET_BASE_DIR, SOURCE_DIRS)
        for i in range(T):
            seed = SEED_BASE + i
            print(f"[Sample {i}] Sampling {sample_size} parquet rows with seed={seed}")
            dataset, source_counts, manifest = sample_dataset(
                source_to_files, sample_size, seed, SOURCE_TEXT_COLUMNS
            )
            save_sampling_outputs(all_dirs[i], dataset, source_counts, manifest, sample_size, seed)
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
def step2_train_lp(samples, ss_dir):
    for i, dataset in enumerate(samples):
        lp_dir = os.path.join(ss_dir, "lp_raw", f"sample_{i}")

        print(f"\n[LP Sample {i}] Computing unique chars")
        char_chunks = dataset.map(
            get_unique_chars_batch,
            batched=True,
            batch_size=BATCH_SIZE,
            num_proc=NUM_PROC,
            remove_columns=dataset.column_names,
            desc=f"Unique chars sample {i}",
        )
        unique_chars = sorted(set().union(*map(set, char_chunks["unique_chars"])))

        for vs in VOCAB_SIZES:
            print(f"[LP Sample {i} vocab={vs}] Training")
            train_lp_tokenizer(dataset, unique_chars, vs, lp_dir, lp_pretokenizer, get_special_tokens(PRETOKENIZER_MODE))
            print(f"[LP Sample {i} vocab={vs}] Saved to {lp_dir}/lp_tokens_{vs}.pkl")


# ---------------------------------------------------------------------------
# Step 3 – Train BPE tokenizers
# ---------------------------------------------------------------------------
def step3_train_bpe(samples, ss_dir):
    for i, dataset in enumerate(samples):
        bpe_dir = os.path.join(ss_dir, "bpe", f"sample_{i}")
        for vs in VOCAB_SIZES:
            out_path = os.path.join(bpe_dir, f"bpe_{vs}")
            print(f"[BPE Sample {i} vocab={vs}] Training")
            train_bpe_tokenizer(vs, dataset, bpe_dir)
            print(f"[BPE Sample {i} vocab={vs}] Saved to {out_path}")


# ---------------------------------------------------------------------------
# Step 4 – Compute pairwise inter-sample Jaccard distances
# ---------------------------------------------------------------------------
def step4_jaccard(sample_size, ss_dir):
    lp_keys = ["all_ones", "det", "bias", "prob"]
    results = {}

    for vs in VOCAB_SIZES:
        print(f"\n[Jaccard vocab={vs}]")
        results[vs] = {}

        # --- LP rounding schemes ---
        token_sets = []
        for i in range(T):
            pkl_path = os.path.join(ss_dir, "lp_raw", f"sample_{i}", f"lp_tokens_{vs}.pkl")
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
            tok_path = os.path.join(ss_dir, "bpe", f"sample_{i}", f"bpe_{vs}")
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
    jaccard_dir = os.path.join(ss_dir, "jaccard")
    os.makedirs(jaccard_dir, exist_ok=True)

    json_path = os.path.join(jaccard_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    print(f"\nSaved JSON results to {json_path}")

    txt_path = os.path.join(jaccard_dir, "results.txt")
    with open(txt_path, "w") as f:
        f.write(f"Experiment : {NAME}\n")
        f.write(f"T          : {T} samples\n")
        f.write(f"sample_size: {sample_size}\n")
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
    print(f"T={T}, SAMPLE_SIZES={SAMPLE_SIZES}, VOCAB_SIZES={VOCAB_SIZES}, SEED_BASE={SEED_BASE}, SOURCE={SOURCE}")
    print(f"Output directory: {EXP_DIR}\n")

    full_ds = None
    if SOURCE == "finewebedu":
        print("Loading pietrolesci/finewebedu-20B (once for all sample sizes)")
        full_ds = load_dataset("pietrolesci/finewebedu-20B", split="train")
        full_ds = full_ds.select_columns(["text"])

    timings = {}
    for sample_size in SAMPLE_SIZES:
        ss_dir = os.path.join(EXP_DIR, f"ss_{sample_size}")
        print(f"\n{'='*60}")
        print(f"Sample size: {sample_size}  →  {ss_dir}")
        print(f"{'='*60}")
        t0 = time.perf_counter()

        print("\n--- Step 1/4: Sampling datasets ---")
        samples = step1_sample_datasets(sample_size, ss_dir, full_ds=full_ds)

        print("\n--- Step 2/4: Training LP tokenizers ---")
        step2_train_lp(samples, ss_dir)

        print("\n--- Step 3/4: Training BPE tokenizers ---")
        step3_train_bpe(samples, ss_dir)

        print("\n--- Step 4/4: Computing Jaccard distances ---")
        step4_jaccard(sample_size, ss_dir)

        elapsed = time.perf_counter() - t0
        timings[sample_size] = elapsed
        print(f"\n[sample_size={sample_size}] Done in {elapsed:.1f}s ({elapsed/60:.1f}min)")

    print(f"\n{'='*60}")
    print("All sample sizes complete. Timings:")
    for ss, t in timings.items():
        print(f"  ss={ss:>8d}: {t:.1f}s ({t/60:.1f}min)")
    print("Done.")
