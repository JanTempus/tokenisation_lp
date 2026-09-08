"""Regenerate averaged length-conditioned Jaccard PDFs from results.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.markers import MarkerStyle  # noqa: E402
from matplotlib.transforms import Affine2D  # noqa: E402

from sampling_jaccard import create_length_conditioned_jaccard_figure


def mark_method_endpoints(axis, results_by_method: dict) -> None:
    """Mark every method's final mean-Jaccard point with a cross."""
    methods = list(results_by_method)
    mean_lines = {
        line.get_label(): line
        for line in axis.lines
        if line.get_label() in {
            method.replace("_", " ") for method in methods
        }
    }
    rotation_step = 20
    first_rotation = -rotation_step * (len(methods) - 1) / 2

    for method_index, method in enumerate(methods):
        label = method.replace("_", " ")
        line = mean_lines.get(label)
        if line is None or len(line.get_xdata()) == 0:
            continue

        x_data = line.get_xdata()
        y_data = line.get_ydata()
        rotation = first_rotation + method_index * rotation_step
        endpoint_marker = MarkerStyle(
            "x",
            transform=Affine2D().rotate_deg(rotation),
        )
        line.set_markevery(range(len(x_data) - 1))
        axis.plot(
            x_data[-1],
            y_data[-1],
            marker=endpoint_marker,
            markersize=12,
            markeredgewidth=2.5,
            color=line.get_color(),
            linestyle="none",
            label="_nolegend_",
            zorder=10,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot all methods' mean pairwise Jaccard and +/-1 standard-deviation "
            "bands from a sampling experiment results.json file."
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
        title = f"Mean Jaccard by stored token length (vocab size {vocab_size})"
        figure = create_length_conditioned_jaccard_figure(
            by_token_length,
            title=title,
            xscale="log",
        )
        try:
            mark_method_endpoints(figure.axes[0], by_token_length)
            figure.tight_layout(rect=(0, 0, 1, 0.96))
            figure.savefig(output_path, dpi=180, bbox_inches="tight")
        finally:
            plt.close(figure)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
