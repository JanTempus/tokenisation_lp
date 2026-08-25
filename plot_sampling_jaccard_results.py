"""Regenerate averaged length-conditioned Jaccard PDFs from results.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sampling_jaccard import plot_length_conditioned_jaccard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot mean pairwise Jaccard and its +/-1 standard-deviation band "
            "from a sampling experiment results.json file."
        )
    )
    parser.add_argument("results_json", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Destination directory (default: directory containing results.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_path = args.results_json.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else results_path.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    with results_path.open(encoding="utf-8") as handle:
        results = json.load(handle)

    for vocab_size, vocab_results in results.items():
        by_token_length = vocab_results.get("by_token_length")
        if not by_token_length:
            raise ValueError(
                f"Vocabulary size {vocab_size} has no by_token_length results"
            )
        output_path = output_dir / f"jaccard_by_token_length_vocab_{vocab_size}.pdf"
        plot_length_conditioned_jaccard(
            by_token_length,
            str(output_path),
            title=f"Mean Jaccard by stored token length (vocab size {vocab_size})",
        )
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
