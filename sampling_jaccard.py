"""Shared Jaccard analysis helpers for tokenizer sampling experiments."""

from __future__ import annotations

import math
import os
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

import matplotlib

# Sampling experiments commonly run as headless batch jobs.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402


JaccardRecord = dict[str, int | float]
LengthConditionedResults = dict[str, list[JaccardRecord]]


def jaccard_score(tokens_a: Iterable[str], tokens_b: Iterable[str]) -> float:
    """Return the project's existing intersection-over-union Jaccard score."""
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 0.0


def group_tokens_by_length(
    tokens: Iterable[str],
) -> dict[int, set[str]]:
    """Group unique vocabulary tokens by stored string length."""
    grouped: defaultdict[int, set[str]] = defaultdict(set)
    for token in set(tokens):
        grouped[len(token)].add(token)
    return dict(grouped)


def pairwise_jaccard_by_length(
    vocabularies: Sequence[Iterable[str]],
) -> LengthConditionedResults:
    """Calculate every sample-pair Jaccard score independently by token length.

    A record is emitted when at least one side has tokens of that length. This
    keeps one-sided buckets (whose score is zero) while omitting comparisons in
    which both buckets are empty.
    """
    grouped_vocabularies = [
        group_tokens_by_length(vocabulary) for vocabulary in vocabularies
    ]

    records_by_length: defaultdict[int, list[JaccardRecord]] = defaultdict(list)
    for sample_i in range(len(grouped_vocabularies)):
        for sample_j in range(sample_i + 1, len(grouped_vocabularies)):
            grouped_i = grouped_vocabularies[sample_i]
            grouped_j = grouped_vocabularies[sample_j]
            for token_length in sorted(grouped_i.keys() | grouped_j.keys()):
                records_by_length[token_length].append(
                    {
                        "sample_i": sample_i,
                        "sample_j": sample_j,
                        "jaccard": jaccard_score(
                            grouped_i.get(token_length, ()),
                            grouped_j.get(token_length, ()),
                        ),
                    }
                )

    return {
        str(token_length): records_by_length[token_length]
        for token_length in sorted(records_by_length)
    }


def _records_to_pair_traces(
    results: Mapping[str, Sequence[Mapping[str, int | float]]],
) -> tuple[list[int], dict[tuple[int, int], dict[int, float]]]:
    lengths: list[int] = []
    traces: defaultdict[tuple[int, int], dict[int, float]] = defaultdict(dict)

    for length_key, records in sorted(results.items(), key=lambda item: int(item[0])):
        token_length = int(length_key)
        lengths.append(token_length)
        for record in records:
            pair = (int(record["sample_i"]), int(record["sample_j"]))
            traces[pair][token_length] = float(record["jaccard"])

    return lengths, dict(traces)


def create_length_conditioned_jaccard_figure(
    results_by_method: Mapping[str, LengthConditionedResults],
    title: str | None = None,
):
    """Create a figure with one exact sample-pair trace per method panel."""
    methods = list(results_by_method)
    if not methods:
        raise ValueError("At least one method is required to plot Jaccard results")

    traces_by_method = {}
    all_pairs: set[tuple[int, int]] = set()
    for method, method_results in results_by_method.items():
        lengths, traces = _records_to_pair_traces(method_results)
        traces_by_method[method] = (lengths, traces)
        all_pairs.update(traces)

    pairs = sorted(all_pairs)
    color_map = plt.get_cmap("tab20")
    pair_colors = {
        pair: color_map(index % color_map.N) for index, pair in enumerate(pairs)
    }

    ncols = min(2, len(methods))
    nrows = math.ceil(len(methods) / ncols)
    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7 * ncols, 3.8 * nrows),
        sharey=True,
        squeeze=False,
    )
    flat_axes = list(np.asarray(axes).ravel())

    for axis, method in zip(flat_axes, methods, strict=False):
        lengths, traces = traces_by_method[method]
        plotted_lengths = (
            list(range(lengths[0], lengths[-1] + 1)) if lengths else []
        )
        for pair in pairs:
            pair_values = traces.get(pair)
            if not pair_values:
                continue
            values = [
                pair_values.get(token_length, math.nan)
                for token_length in plotted_lengths
            ]
            axis.plot(
                plotted_lengths,
                values,
                color=pair_colors[pair],
                linewidth=1.0,
                marker="o",
                markersize=2.5,
            )
        axis.set_title(method.replace("_", " "))
        axis.set_xlabel("Stored token length")
        axis.set_ylabel("Jaccard (intersection / union)")
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.25)

    for unused_axis in flat_axes[len(methods):]:
        unused_axis.set_visible(False)

    if pairs:
        legend_handles = [
            Line2D(
                [0],
                [0],
                color=pair_colors[pair],
                marker="o",
                linewidth=1.0,
                markersize=3,
                label=f"sample {pair[0]} vs {pair[1]}",
            )
            for pair in pairs
        ]
        figure.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=min(5, len(legend_handles)),
            frameon=False,
        )

    if title:
        figure.suptitle(title)
    figure.tight_layout(rect=(0, 0.08 if pairs else 0, 1, 0.96 if title else 1))
    return figure


def plot_length_conditioned_jaccard(
    results_by_method: Mapping[str, LengthConditionedResults],
    output_path: str,
    title: str | None = None,
) -> str:
    """Save a headless PNG containing the length-conditioned pair traces."""
    parent = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(parent, exist_ok=True)
    figure = create_length_conditioned_jaccard_figure(results_by_method, title=title)
    try:
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
    finally:
        plt.close(figure)
    return output_path
