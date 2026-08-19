"""
Unified sampling experiment pipeline.

For each sample size in SAMPLE_SIZES and each of T independent samples it:
  1. Samples SAMPLE_SIZE rows from the dataset (using a unique seed per sample)
  2. Trains the LP tokenizer for every requested vocab size
  3. Trains the BPE and Unigram tokenizers for every requested vocab size
  4. Computes pairwise inter-sample Jaccard distances (same vocab size, different samples)
     for the LP rounding schemes, BPE, and Unigram, and saves the results.

Sampled datasets and scratch artifacts are written to WORK_DIR on Iopsstor.
Trained tokenizers and final Jaccard results are written to RESULTS_DIR on
Capstor. All settings are read from environment variables (see
run_sampling_experiment.sbatch).
"""

import json
import os
import shutil
import time

import numpy as np
from datasets import load_dataset, load_from_disk
from transformers import PreTrainedTokenizerFast

from sampling_jaccard import (
    jaccard_score,
    pairwise_jaccard_by_length,
    plot_length_conditioned_jaccard,
)

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
WORK_DIR      = _require_env("WORK_DIR")
RESULTS_DIR   = _require_env("RESULTS_DIR")

# ---------------------------------------------------------------------------
# Imports that read env-vars at module load time (must come after env is set)
# ---------------------------------------------------------------------------
# train_tokenizer reads PRETOKENIZER_MODE when the module is first imported.
from train_tokenizer import (  # noqa: E402
    BYTE_LEVEL_ALPHABET,
    PRETOKENIZER_MODE,
    get_special_tokens,
    pretokenizer as lp_pretokenizer,
    train_lp_tokenizer,
)
from hf_baseline_tokenizers.hf_baseline_tokenizers import (  # noqa: E402
    train_bpe_tokenizer,
    train_unigram_tokenizer,
)
import pickle

from lp_tokenizer.lp_functions import (  # noqa: E402
    biased_rounding,
    deterministic_rounding,
)


BASELINE_TRAINERS = {
    "bpe": train_bpe_tokenizer,
    "unigram": train_unigram_tokenizer,
}


def jaccard_distance(a, b):
    return jaccard_score(a, b)


def jaccard_distance_different_rounding(vocab_size, raw_tokens_path):
    with open(raw_tokens_path, "rb") as f:
        tokens = pickle.load(f)
    n_special = len(tokens["special_tokens"])
    target = vocab_size - n_special
    det_tokens  = deterministic_rounding(tokens["possible_tokens"], tokens["unique_chars"], target)
    bias_tokens = biased_rounding(tokens["possible_tokens"], tokens["unique_chars"], target)
    ones_tokens = [t.token for t in tokens["possible_tokens"] if t.lp_value >= 0.999]
    det_tokens = list(set(det_tokens))
    bias_tokens = list(set(bias_tokens))
    ones_tokens = list(set(ones_tokens + tokens["unique_chars"]))
    return {
        "all_ones": ones_tokens,
        "det": det_tokens,
        "bias": bias_tokens,
    }


# ---------------------------------------------------------------------------
# Step 1 – Sample T independent datasets
# ---------------------------------------------------------------------------
def step1_sample_datasets(sample_size, ss_dir, full_ds=None):
    from sample_tokenizer_data import (
        discover_parquet_files,
        sample_climbmix,
        sample_dataset,
        save_sampling_outputs,
    )

    all_dirs = [os.path.join(ss_dir, "samples", f"sample_{i}") for i in range(T)]

    def _already_sampled(d):
        return os.path.exists(os.path.join(d, "dataset_info.json"))

    if SOURCE == "finewebedu":
        for i in range(T):
            if _already_sampled(all_dirs[i]):
                print(f"[Sample {i}] already sampled at {all_dirs[i]}, skipping")
                continue
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
        source_to_files = None
        for i in range(T):
            if _already_sampled(all_dirs[i]):
                print(f"[Sample {i}] already sampled at {all_dirs[i]}, skipping")
                continue
            if source_to_files is None:
                source_to_files = discover_parquet_files(DATASET_BASE_DIR, SOURCE_DIRS)
            seed = SEED_BASE + i
            print(f"[Sample {i}] Sampling {sample_size} parquet rows with seed={seed}")
            dataset, source_counts, manifest = sample_dataset(
                source_to_files, sample_size, seed, SOURCE_TEXT_COLUMNS
            )
            save_sampling_outputs(all_dirs[i], dataset, source_counts, manifest, sample_size, seed)
    elif SOURCE == "climbmix":
        DATASET_ID = os.environ.get("DATASET_ID", "karpathy/climbmix-400b-shuffle")
        NUM_SHARDS = int(_require_env("NUM_SHARDS"))
        SHARD_TMP_BASE = os.environ.get(
            "CLIMBMIX_SHARD_TMP", os.path.join(ss_dir, "_shard_tmp")
        )
        for i in range(T):
            if _already_sampled(all_dirs[i]):
                print(f"[Sample {i}] already sampled at {all_dirs[i]}, skipping")
                continue
            seed = SEED_BASE + i
            per_sample_tmp = os.path.join(SHARD_TMP_BASE, f"sample_{i}")
            print(
                f"[Sample {i}] climbmix: sampling {NUM_SHARDS} shards + "
                f"{sample_size} rows (seed={seed})"
            )
            dataset, source_counts, manifest, shard_dir = sample_climbmix(
                DATASET_ID, NUM_SHARDS, sample_size, seed, per_sample_tmp
            )
            save_sampling_outputs(
                all_dirs[i], dataset, source_counts, manifest, sample_size, seed
            )
            shutil.rmtree(shard_dir, ignore_errors=True)
            print(f"[Sample {i}] deleted downloaded shards at {shard_dir}")
    else:
        raise ValueError(f"Unknown SOURCE='{SOURCE}'. Expected: finewebedu, parquet, climbmix")

    samples = []
    for i in range(T):
        dataset = load_from_disk(all_dirs[i])
        print(f"[Sample {i}] Loaded {len(dataset)} rows")
        samples.append(dataset)
    return samples


# ---------------------------------------------------------------------------
# Step 2 – Train LP tokenizers
# ---------------------------------------------------------------------------
def step2_train_lp(samples, result_ss_dir):
    for i, dataset in enumerate(samples):
        lp_dir = os.path.join(result_ss_dir, "lp_raw", f"sample_{i}")

        missing = [
            vs for vs in VOCAB_SIZES
            if not os.path.exists(os.path.join(lp_dir, f"lp_tokens_{vs}.pkl"))
        ]
        if not missing:
            print(f"\n[LP Sample {i}] all {len(VOCAB_SIZES)} vocab sizes already trained, skipping")
            continue

        print(
            f"\n[LP Sample {i}] Using fixed ByteLevel alphabet "
            f"(training {len(missing)}/{len(VOCAB_SIZES)} vocab sizes: {missing})"
        )
        unique_chars = list(BYTE_LEVEL_ALPHABET)

        for vs in missing:
            print(f"[LP Sample {i} vocab={vs}] Training")
            train_lp_tokenizer(dataset, unique_chars, vs, lp_dir, lp_pretokenizer, get_special_tokens(PRETOKENIZER_MODE))
            print(f"[LP Sample {i} vocab={vs}] Saved to {lp_dir}/lp_tokens_{vs}.pkl")


# ---------------------------------------------------------------------------
# Step 3 – Train baseline tokenizers
# ---------------------------------------------------------------------------
def step3_train_baselines(samples, result_ss_dir):
    for baseline, train_baseline in BASELINE_TRAINERS.items():
        for i, dataset in enumerate(samples):
            baseline_dir = os.path.join(result_ss_dir, baseline, f"sample_{i}")
            for vs in VOCAB_SIZES:
                out_path = os.path.join(baseline_dir, f"{baseline}_{vs}")
                if os.path.exists(os.path.join(out_path, "tokenizer.json")):
                    print(
                        f"[{baseline.upper()} Sample {i} vocab={vs}] "
                        f"already trained at {out_path}, skipping"
                    )
                    continue
                print(f"[{baseline.upper()} Sample {i} vocab={vs}] Training")
                train_baseline(vs, dataset, baseline_dir)
                print(f"[{baseline.upper()} Sample {i} vocab={vs}] Saved to {out_path}")


# ---------------------------------------------------------------------------
# Step 4 – Compute pairwise inter-sample Jaccard distances
# ---------------------------------------------------------------------------
def step4_jaccard(sample_size, result_ss_dir):
    lp_keys = ["all_ones", "det", "bias"]
    results = {}
    jaccard_dir = os.path.join(result_ss_dir, "jaccard")
    os.makedirs(jaccard_dir, exist_ok=True)

    for vs in VOCAB_SIZES:
        print(f"\n[Jaccard vocab={vs}]")
        results[vs] = {"by_token_length": {}}

        # --- LP rounding schemes ---
        token_sets = []
        for i in range(T):
            pkl_path = os.path.join(
                result_ss_dir,
                "lp_raw",
                f"sample_{i}",
                f"lp_tokens_{vs}.pkl",
            )
            token_sets.append(jaccard_distance_different_rounding(vs, pkl_path))

        for key in lp_keys:
            mat = np.zeros((T, T))
            for i in range(T):
                for j in range(i + 1, T):
                    d = jaccard_distance(token_sets[i][key], token_sets[j][key])
                    mat[i][j] = d
                    mat[j][i] = d
            method = f"lp_{key}"
            results[vs][method] = mat.tolist()
            results[vs]["by_token_length"][method] = pairwise_jaccard_by_length(
                [sample_tokens[key] for sample_tokens in token_sets]
            )
            upper = mat[np.triu_indices(T, k=1)]
            print(f"  LP {key}: mean={upper.mean():.4f}  min={upper.min():.4f}  max={upper.max():.4f}")

        # --- Baselines ---
        for baseline in BASELINE_TRAINERS:
            baseline_vocabs = []
            for i in range(T):
                tok_path = os.path.join(
                    result_ss_dir,
                    baseline,
                    f"sample_{i}",
                    f"{baseline}_{vs}",
                )
                tok = PreTrainedTokenizerFast.from_pretrained(tok_path)
                baseline_vocabs.append(
                    set(tok.get_vocab()) - set(tok.all_special_tokens)
                )

            mat = np.zeros((T, T))
            for i in range(T):
                for j in range(i + 1, T):
                    d = jaccard_distance(
                        list(baseline_vocabs[i]), list(baseline_vocabs[j])
                    )
                    mat[i][j] = d
                    mat[j][i] = d
            results[vs][baseline] = mat.tolist()
            results[vs]["by_token_length"][baseline] = pairwise_jaccard_by_length(
                baseline_vocabs
            )
            upper = mat[np.triu_indices(T, k=1)]
            print(
                f"  {baseline.upper()}: mean={upper.mean():.4f}  "
                f"min={upper.min():.4f}  max={upper.max():.4f}"
            )

        plot_path = os.path.join(
            jaccard_dir, f"jaccard_by_token_length_vocab_{vs}.png"
        )
        plot_length_conditioned_jaccard(
            results[vs]["by_token_length"],
            plot_path,
            title=f"Jaccard by stored token length (vocab size {vs})",
        )
        print(f"  Saved length-conditioned plot to {plot_path}")

    # --- Persist results ---
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
            f.write(
                "  Length-conditioned raw pair results: "
                "results.json -> by_token_length\n"
            )
            f.write(
                "  Length-conditioned plot: "
                f"jaccard_by_token_length_vocab_{vs}.png\n"
            )
            for key in lp_keys:
                mat = np.array(results[vs][f"lp_{key}"])
                upper = mat[np.triu_indices(T, k=1)]
                f.write(f"  LP {key:8s}: mean={upper.mean():.4f}  min={upper.min():.4f}  max={upper.max():.4f}\n")
                f.write(f"    {np.array2string(mat, precision=4)}\n")
            for baseline in BASELINE_TRAINERS:
                mat = np.array(results[vs][baseline])
                upper = mat[np.triu_indices(T, k=1)]
                f.write(
                    f"  {baseline.upper():10s}: mean={upper.mean():.4f}  "
                    f"min={upper.min():.4f}  max={upper.max():.4f}\n"
                )
                f.write(f"    {np.array2string(mat, precision=4)}\n")
            f.write("\n")
    print(f"Saved human-readable summary to {txt_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"=== Sampling Experiment: {NAME} ===")
    print(f"T={T}, SAMPLE_SIZES={SAMPLE_SIZES}, VOCAB_SIZES={VOCAB_SIZES}, SEED_BASE={SEED_BASE}, SOURCE={SOURCE}")
    print(f"Iopsstor work directory: {WORK_DIR}")
    print(f"Capstor results directory: {RESULTS_DIR}\n")

    full_ds = None
    if SOURCE == "finewebedu":
        print("Loading pietrolesci/finewebedu-20B (once for all sample sizes)")
        full_ds = load_dataset("pietrolesci/finewebedu-20B", split="train")
        full_ds = full_ds.select_columns(["text"])

    timings = {}
    for sample_size in SAMPLE_SIZES:
        ss_dir = os.path.join(WORK_DIR, f"ss_{sample_size}")
        result_ss_dir = os.path.join(RESULTS_DIR, f"ss_{sample_size}")
        print(f"\n{'='*60}")
        print(f"Sample size: {sample_size}")
        print(f"Intermediate work: {ss_dir}")
        print(f"Final results: {result_ss_dir}")
        print(f"{'='*60}")
        t0 = time.perf_counter()

        print("\n--- Step 1/4: Sampling datasets ---")
        samples = step1_sample_datasets(sample_size, ss_dir, full_ds=full_ds)

        print("\n--- Step 2/4: Training LP tokenizers ---")
        step2_train_lp(samples, result_ss_dir)

        print("\n--- Step 3/4: Training BPE and Unigram tokenizers ---")
        step3_train_baselines(samples, result_ss_dir)

        print("\n--- Step 4/4: Computing Jaccard distances ---")
        step4_jaccard(sample_size, result_ss_dir)

        elapsed = time.perf_counter() - t0
        timings[sample_size] = elapsed
        print(f"\n[sample_size={sample_size}] Done in {elapsed:.1f}s ({elapsed/60:.1f}min)")

    print(f"\n{'='*60}")
    print("All sample sizes complete. Timings:")
    for ss, t in timings.items():
        print(f"  ss={ss:>8d}: {t:.1f}s ({t/60:.1f}min)")

    summary_path = os.path.join(RESULTS_DIR, "run_summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "name": NAME,
                "samples_per_size": T,
                "sample_sizes": SAMPLE_SIZES,
                "vocab_sizes": VOCAB_SIZES,
                "seed_base": SEED_BASE,
                "source": SOURCE,
                "work_dir": WORK_DIR,
                "results_dir": RESULTS_DIR,
                "timings_seconds": {
                    str(sample_size): elapsed
                    for sample_size, elapsed in timings.items()
                },
            },
            f,
            indent=2,
        )
    print(f"Run summary saved to {summary_path}")
    print("Done.")
