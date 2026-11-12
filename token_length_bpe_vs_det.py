"""Token-length distribution of the BPE and LP deterministic-rounding vocabularies,
for every (sample size, vocab size) found on disk.

Uses the same layout and loaders as jaccard_bpe_vs_det.py (keep both files in
the same directory). Unlike the Jaccard script, a scheme is reported even if the
other one is missing for that run.

Length = stored string length in byte-level characters (so 'Ġ' counts as one).
Statistics pool all samples of a (sample size, vocab size, scheme) group.

Usage:
    python token_length_bpe_vs_det.py /path/to/results [more roots...] [-o out_prefix]

Writes:
    <out_prefix>_summary.csv    one row per (experiment, ss, vocab, scheme)
    <out_prefix>_histogram.csv  mean token count per length, per group
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
from collections import Counter, defaultdict

from jaccard_bpe_vs_det import (
    SS_RE,
    discover,
    load_bpe_vocab,
    load_det_vocab,
    runs_in_ss_dir,
)

BUCKETS = [("len1", 1, 1), ("len2_4", 2, 4), ("len5_8", 5, 8),
           ("len9_16", 9, 16), ("len17plus", 17, None)]
SCHEMES = ("bpe", "det")


def percentile(sorted_vals: list[int], p: float) -> float:
    """Nearest-rank percentile on an already sorted list."""
    k = max(0, min(len(sorted_vals) - 1, round(p / 100 * (len(sorted_vals) - 1))))
    return float(sorted_vals[k])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("roots", nargs="*", default=["."], help="directories to search")
    ap.add_argument("-o", "--out", default="token_length_bpe_vs_det",
                    help="output prefix (default: %(default)s)")
    args = ap.parse_args()

    # (experiment, ss, vocab, scheme) -> list of per-sample length Counters
    groups: dict[tuple, list[Counter]] = defaultdict(list)
    errors = []
    for root in args.roots:
        for exp_dir, ss_dir in discover(root):
            exp = os.path.abspath(exp_dir)
            ss = int(SS_RE.match(os.path.basename(ss_dir))[1])
            for (sample, vs), paths in sorted(runs_in_ss_dir(ss_dir).items()):
                for scheme, key in (("bpe", "bpe"), ("det", "lp")):
                    if key not in paths:
                        continue
                    try:
                        vocab = (load_bpe_vocab(paths[key]) if scheme == "bpe"
                                 else load_det_vocab(paths[key], vs))
                    except Exception as e:
                        errors.append(f"{paths[key]}: {e}")
                        continue
                    groups[(exp, ss, vs, scheme)].append(Counter(len(t) for t in vocab))
                print(f"loaded ss={ss} sample={sample} vocab={vs} "
                      f"({', '.join(k for k in ('bpe', 'lp') if k in paths)})", flush=True)

    if not groups:
        print("No vocabularies found.", file=sys.stderr)
        sys.exit(1)

    summary_rows, hist_rows = [], []
    for (exp, ss, vs, scheme), counters in sorted(groups.items()):
        total = Counter()
        for c in counters:
            total.update(c)
        lens = sorted(total.elements())
        n_tok = len(lens)
        row = {
            "experiment": exp, "sample_size": ss, "vocab_size": vs, "scheme": scheme,
            "n_samples": len(counters),
            "tokens_per_sample": n_tok / len(counters),
            "mean": statistics.mean(lens),
            "sd": statistics.pstdev(lens),
            "median": statistics.median(lens),
            "p90": percentile(lens, 90),
            "p99": percentile(lens, 99),
            "max": lens[-1],
            # spread of the per-sample mean, to show sample-to-sample variability
            "mean_sd_across_samples": (
                statistics.pstdev([sum(l * c for l, c in ct.items()) / sum(ct.values())
                                   for ct in counters]) if len(counters) > 1 else 0.0),
        }
        for name, lo, hi in BUCKETS:
            cnt = sum(c for l, c in total.items() if l >= lo and (hi is None or l <= hi))
            row[f"pct_{name}"] = 100 * cnt / n_tok
        summary_rows.append(row)
        for length in sorted(total):
            hist_rows.append({
                "experiment": exp, "sample_size": ss, "vocab_size": vs, "scheme": scheme,
                "length": length,
                "mean_count": total[length] / len(counters),
                "pct": 100 * total[length] / n_tok,
            })

    with open(args.out + "_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        w.writeheader()
        w.writerows(summary_rows)
    with open(args.out + "_histogram.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hist_rows[0]))
        w.writeheader()
        w.writerows(hist_rows)

    # Console table
    hdr = (f"{'ss':>7} {'V':>7} {'sch':>4} {'n':>2} {'mean':>5} {'sd':>5} {'med':>4} "
           f"{'p90':>4} {'p99':>4} {'max':>4} | {'%1':>4} {'%2-4':>5} {'%5-8':>5} "
           f"{'%9-16':>5} {'%>16':>5}")
    last_exp = None
    for r in summary_rows:
        if r["experiment"] != last_exp:
            print(f"\n{r['experiment']}\n{hdr}")
            last_exp = r["experiment"]
        print(f"{r['sample_size']:>7} {r['vocab_size']:>7} {r['scheme']:>4} {r['n_samples']:>2} "
              f"{r['mean']:>5.2f} {r['sd']:>5.2f} {r['median']:>4.0f} {r['p90']:>4.0f} "
              f"{r['p99']:>4.0f} {r['max']:>4} | {r['pct_len1']:>4.1f} {r['pct_len2_4']:>5.1f} "
              f"{r['pct_len5_8']:>5.1f} {r['pct_len9_16']:>5.1f} {r['pct_len17plus']:>5.2f}")

    if errors:
        print(f"\n{len(errors)} files failed to load:")
        for e in errors:
            print("  ", e)
    print(f"\nWrote {args.out}_summary.csv and {args.out}_histogram.csv")


if __name__ == "__main__":
    main()
