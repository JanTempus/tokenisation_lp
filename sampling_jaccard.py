"""Shared Jaccard analysis helpers for tokenizer sampling experiments."""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

import matplotlib

# Sampling experiments commonly run as headless batch jobs.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FormatStrFormatter, LogLocator  # noqa: E402


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

    Every sample pair is emitted for every token length observed in at least
    one vocabulary. If both samples have an empty bucket, ``jaccard_score``
    records zero, consistently with its empty-union behavior.
    """
    grouped_vocabularies = [
        group_tokens_by_length(vocabulary) for vocabulary in vocabularies
    ]

    all_token_lengths = sorted(
        set().union(*(grouped.keys() for grouped in grouped_vocabularies))
    )
    records_by_length: defaultdict[int, list[JaccardRecord]] = defaultdict(list)
    for sample_i in range(len(grouped_vocabularies)):
        for sample_j in range(sample_i + 1, len(grouped_vocabularies)):
            grouped_i = grouped_vocabularies[sample_i]
            grouped_j = grouped_vocabularies[sample_j]
            for token_length in all_token_lengths:
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


def _infer_sample_count(
    results: Mapping[str, Sequence[Mapping[str, int | float]]],
) -> int:
    sample_indices = {
        int(record[key])
        for records in results.values()
        for record in records
        for key in ("sample_i", "sample_j")
    }
    return max(sample_indices, default=-1) + 1


def summarize_jaccard_by_length(
    results: Mapping[str, Sequence[Mapping[str, int | float]]],
) -> tuple[list[int], list[float], list[float]]:
    """Return token lengths and pairwise mean/std Jaccard at each length.

    Older result files omitted a pair when both samples had no tokens of a
    given length. ``jaccard_score`` defines that empty-union case as zero, so
    this function restores those missing zero-valued comparisons before
    calculating the statistics. New result files explicitly contain every
    sample pair for every globally observed token length.
    """
    sample_count = _infer_sample_count(results)
    expected_pairs = {
        (sample_i, sample_j)
        for sample_i in range(sample_count)
        for sample_j in range(sample_i + 1, sample_count)
    }

    lengths: list[int] = []
    means: list[float] = []
    standard_deviations: list[float] = []
    for length_key, records in sorted(results.items(), key=lambda item: int(item[0])):
        scores_by_pair = {
            (int(record["sample_i"]), int(record["sample_j"])): float(
                record["jaccard"]
            )
            for record in records
        }
        scores = np.asarray(
            [scores_by_pair.get(pair, 0.0) for pair in sorted(expected_pairs)],
            dtype=float,
        )
        if scores.size == 0:
            continue
        lengths.append(int(length_key))
        means.append(float(np.mean(scores)))
        standard_deviations.append(float(np.std(scores)))

    return lengths, means, standard_deviations


def create_length_conditioned_jaccard_figure(
    results_by_method: Mapping[str, LengthConditionedResults],
    title: str | None = None,
    xscale: str = "linear",
):
    """Plot every method's pairwise mean Jaccard and population +/-1 SD."""
    methods = list(results_by_method)
    if not methods:
        raise ValueError("At least one method is required to plot Jaccard results")

    figure, axis = plt.subplots(figsize=(11, 6.5))
    color_map = plt.get_cmap("tab10")

    for method_index, method in enumerate(methods):
        lengths, means, standard_deviations = summarize_jaccard_by_length(
            results_by_method[method]
        )
        means_array = np.asarray(means)
        std_array = np.asarray(standard_deviations)
        lower_one_sd = np.clip(means_array - std_array, 0.0, 1.0)
        upper_one_sd = np.clip(means_array + std_array, 0.0, 1.0)
        color = color_map(method_index % color_map.N)

        axis.fill_between(
            lengths,
            lower_one_sd,
            upper_one_sd,
            color=color,
            alpha=0.1,
        )
        axis.plot(
            lengths,
            lower_one_sd,
            color=color,
            linewidth=0.7,
            linestyle="--",
            alpha=0.65,
        )
        axis.plot(
            lengths,
            upper_one_sd,
            color=color,
            linewidth=0.7,
            linestyle="--",
            alpha=0.65,
        )
        axis.plot(
            lengths,
            means_array,
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=3,
            label=method.replace("_", " "),
        )

    axis.set_xlabel("Stored token length")
    axis.set_xscale(xscale)
    if xscale == "log":
        axis.xaxis.set_major_locator(
            LogLocator(base=10, subs=(1.0, 2.0, 5.0))
        )
        axis.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    axis.set_ylabel("Jaccard (intersection / union)")
    axis.set_ylim(-0.02, 1.02)
    axis.grid(alpha=0.25)
    axis.legend(
        title="Method (solid: mean; dashed/band: +/- 1 SD)",
        frameon=False,
        ncol=2,
    )

    if title:
        figure.suptitle(title)
    figure.tight_layout(rect=(0, 0, 1, 0.96 if title else 1))
    return figure


def plot_length_conditioned_jaccard(
    results_by_method: Mapping[str, LengthConditionedResults],
    output_path: str,
    title: str | None = None,
    xscale: str = "linear",
) -> str:
    """Save a headless mean-and-standard-deviation Jaccard figure."""
    parent = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(parent, exist_ok=True)
    figure = create_length_conditioned_jaccard_figure(
        results_by_method,
        title=title,
        xscale=xscale,
    )
    try:
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
    finally:
        plt.close(figure)
    return output_path
