"""Jaccard similarity between the BPE vocabulary and the LP deterministic-rounding
vocabulary, for every (sample size, sample, vocab size) run found on disk.

Expected layout (as written by sampling_experiment.py):

    <root>/ss_<S>/lp_raw/sample_<i>/lp_tokens_<V>.pkl
    <root>/ss_<S>/bpe/sample_<i>/bpe_<V>/tokenizer.json

<root> is searched recursively, so you can point it at a directory holding
several experiments. A run is used only if both files exist; anything missing
is listed at the end.

Dependencies: Python standard library only (numpy is needed only if the pickles
contain numpy scalars). The LP pickles are loaded without importing
lp_tokenizer, so cvxpy/transformers are not required.

Usage:
    python jaccard_bpe_vs_det.py /path/to/results [more roots...] [-o out_prefix]

Writes <out_prefix>.csv (one row per run), <out_prefix>.json (rows plus
per-token-length breakdown) and prints a mean/std summary per (S, V).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import re
import statistics
import sys
from collections import defaultdict


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
class _Stub:
    """Stand-in for lp_tokenizer classes (possibleToken, tokenInstance, ...)."""

    def __setstate__(self, state):
        if isinstance(state, tuple) and len(state) == 2:  # (dict, slots)
            state = {**(state[0] or {}), **(state[1] or {})}
        self.__dict__.update(state)


class _LPUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("lp_tokenizer"):
            return _Stub
        return super().find_class(module, name)


def load_det_vocab(pkl_path: str, vocab_size: int) -> set[str]:
    """Same as lp_functions.deterministic_rounding with target = V - #special."""
    with open(pkl_path, "rb") as f:
        tokens = _LPUnpickler(f).load()
    unique_chars = list(tokens["unique_chars"])
    target = vocab_size - len(tokens["special_tokens"])
    if target < len(unique_chars):
        raise ValueError(f"{pkl_path}: more unique chars than vocab budget")
    ranked = sorted(tokens["possible_tokens"], key=lambda t: t.lp_value, reverse=True)
    chosen = [t.token for t in ranked[: target - len(unique_chars)]]
    return set(unique_chars + chosen)


def load_bpe_vocab(tokenizer_json: str) -> set[str]:
    """Vocab of an HF tokenizer.json minus special added tokens."""
    with open(tokenizer_json, encoding="utf-8") as f:
        data = json.load(f)
    vocab = set(data["model"]["vocab"])
    vocab |= {t["content"] for t in data.get("added_tokens", [])}
    special = {t["content"] for t in data.get("added_tokens", []) if t.get("special")}
    return vocab - special


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def jaccard_by_length(a: set[str], b: set[str]) -> dict[str, float]:
    by_len_a, by_len_b = defaultdict(set), defaultdict(set)
    for t in a:
        by_len_a[len(t)].add(t)
    for t in b:
        by_len_b[len(t)].add(t)
    lengths = sorted(set(by_len_a) | set(by_len_b))
    return {str(n): jaccard(by_len_a[n], by_len_b[n]) for n in lengths}


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
SS_RE = re.compile(r"^ss_(\d+)$")
SAMPLE_RE = re.compile(r"^sample_(\d+)$")
LP_RE = re.compile(r"^lp_tokens_(\d+)\.pkl$")
BPE_RE = re.compile(r"^bpe_(\d+)$")


def discover(root: str):
    """Yield (experiment_dir, ss_dir) for every ss_* directory under root."""
    for dirpath, dirnames, _ in os.walk(root):
        for d in sorted(dirnames):
            if SS_RE.match(d):
                yield dirpath, os.path.join(dirpath, d)
        # don't descend into ss_* trees, they hold no further experiments
        dirnames[:] = [d for d in dirnames if not SS_RE.match(d)]


def runs_in_ss_dir(ss_dir: str):
    """Return {(sample, vocab): {'lp': path, 'bpe': path}} for one ss_* dir."""
    runs: dict[tuple[int, int], dict[str, str]] = defaultdict(dict)

    lp_root = os.path.join(ss_dir, "lp_raw")
    if os.path.isdir(lp_root):
        for s in os.listdir(lp_root):
            m = SAMPLE_RE.match(s)
            if not m:
                continue
            for fn in os.listdir(os.path.join(lp_root, s)):
                mv = LP_RE.match(fn)
                if mv:
                    runs[(int(m[1]), int(mv[1]))]["lp"] = os.path.join(lp_root, s, fn)

    bpe_root = os.path.join(ss_dir, "bpe")
    if os.path.isdir(bpe_root):
        for s in os.listdir(bpe_root):
            m = SAMPLE_RE.match(s)
            if not m:
                continue
            for d in os.listdir(os.path.join(bpe_root, s)):
                mv = BPE_RE.match(d)
                tj = os.path.join(bpe_root, s, d, "tokenizer.json")
                if mv and os.path.isfile(tj):
                    runs[(int(m[1]), int(mv[1]))]["bpe"] = tj
    return runs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("roots", nargs="*", default=["."], help="directories to search")
    ap.add_argument("-o", "--out", default="jaccard_bpe_vs_det",
                    help="output prefix for .csv/.json (default: %(default)s)")
    args = ap.parse_args()

    rows, by_length, missing = [], {}, []
    for root in args.roots:
        for exp_dir, ss_dir in discover(root):
            ss = int(SS_RE.match(os.path.basename(ss_dir))[1])
            for (sample, vs), paths in sorted(runs_in_ss_dir(ss_dir).items()):
                if "lp" not in paths or "bpe" not in paths:
                    have = "lp" if "lp" in paths else "bpe"
                    missing.append(f"{exp_dir} ss={ss} sample={sample} vocab={vs} (only {have})")
                    continue
                try:
                    det = load_det_vocab(paths["lp"], vs)
                    bpe = load_bpe_vocab(paths["bpe"])
                except Exception as e:  # keep going on corrupt/partial files
                    missing.append(f"{exp_dir} ss={ss} sample={sample} vocab={vs} (error: {e})")
                    continue
                inter = len(det & bpe)
                union = len(det | bpe)
                j = inter / union if union else 0.0
                row = {
                    "experiment": os.path.abspath(exp_dir),
                    "sample_size": ss,
                    "sample": sample,
                    "vocab_size": vs,
                    "det_size": len(det),
                    "bpe_size": len(bpe),
                    "intersection": inter,
                    "union": union,
                    "jaccard": j,
                    "jaccard_distance": 1.0 - j,
                }
                rows.append(row)
                by_length[f"{row['experiment']}|ss_{ss}|sample_{sample}|vocab_{vs}"] = (
                    jaccard_by_length(det, bpe)
                )
                print(f"ss={ss:>7} sample={sample} vocab={vs:>7}  J={j:.4f}  d={1 - j:.4f}",
                      flush=True)

    if not rows:
        print("No complete (lp_raw + bpe) runs found.", file=sys.stderr)
        for m in missing:
            print("  missing:", m, file=sys.stderr)
        sys.exit(1)

    with open(args.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(args.out + ".json", "w") as f:
        json.dump({"runs": rows, "by_token_length": by_length, "missing": missing}, f, indent=2)

    # Summary: mean/std over samples for each (experiment, ss, vocab)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["experiment"], r["sample_size"], r["vocab_size"])].append(r["jaccard"])
    print("\n=== Jaccard(BPE, LP det) — mean ± std over samples ===")
    last_exp = None
    for (exp, ss, vs), js in sorted(groups.items()):
        if exp != last_exp:
            print(f"\n{exp}")
            last_exp = exp
        sd = statistics.pstdev(js) if len(js) > 1 else 0.0
        print(f"  ss={ss:>7} vocab={vs:>7}  n={len(js)}  J={statistics.mean(js):.4f} ± {sd:.4f}"
              f"  (distance {1 - statistics.mean(js):.4f})")

    if missing:
        print(f"\n{len(missing)} incomplete runs skipped (see {args.out}.json -> missing)")
    print(f"\nWrote {args.out}.csv and {args.out}.json")


if __name__ == "__main__":
    main()
