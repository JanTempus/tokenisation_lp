#!/usr/bin/env python3
"""Compare LP and BPE tokenizer output on sampled parquet documents.

By default this lists and downloads sampled parquet shards from the same
climbmix Hugging Face dataset used by the training scripts:

    python compare_tokenizer_samples.py \
        --output tokenizer_comparison.md

To compare against local parquet files instead:

    python compare_tokenizer_samples.py \
        --source local \
        --parquet-root /path/to/parquet/root \
        --output tokenizer_comparison.md
"""

from __future__ import annotations

import argparse
import os
import random
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_PARQUET_ROOT = Path(
    os.environ.get(
        "TOKENIZER_DATASET_BASE",
        "/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/tokenizer_training_dataset",
    )
)
DEFAULT_DATASET_ID = os.environ.get("DATASET_ID", "karpathy/climbmix-400b-shuffle")
DEFAULT_HF_REVISION = os.environ.get("HF_REVISION", "main")
DEFAULT_DOWNLOAD_DIR = REPO_ROOT / "downloaded_parquet_samples"
DEFAULT_LP_TOKENIZER = (
    REPO_ROOT
    / "rounded_tokenizers"
    / "cross_over_climbmix400b_s7"
    / "vocab_32768"
    / "lp_32768_det"
)
DEFAULT_BPE_TOKENIZER = (
    REPO_ROOT / "bpe_tokenizers_climbmix" / "nano_bpe_climb_mix" / "32768"
)
PREFERRED_TEXT_COLUMNS = ("text", "content", "code")


@dataclass(frozen=True)
class DocumentSample:
    file_number: int
    document_number: int
    file_path: Path
    row_index: int
    row_count: int
    text_column: str
    text: str


@dataclass(frozen=True)
class TokenizedSample:
    sample: DocumentSample
    lp_tokens: list[str]
    lp_ids: list[int]
    bpe_tokens: list[str]
    bpe_ids: list[int]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample parquet documents and compare tokenization from the local "
            "32768 LP and BPE tokenizers."
        )
    )
    parser.add_argument(
        "--source",
        choices=("hf", "local"),
        default=os.environ.get("COMPARE_SOURCE", "hf"),
        help=(
            "Where to sample parquet files from. 'hf' lists and downloads "
            "parquet shards from --dataset-id; 'local' samples --parquet-root. "
            "Default: hf."
        ),
    )
    parser.add_argument(
        "--dataset-id",
        default=DEFAULT_DATASET_ID,
        help=(
            "Hugging Face dataset repo id used when --source hf. "
            f"Default: {DEFAULT_DATASET_ID}."
        ),
    )
    parser.add_argument(
        "--revision",
        default=DEFAULT_HF_REVISION,
        help=(
            "Hugging Face dataset revision used when --source hf. "
            f"Default: {DEFAULT_HF_REVISION}."
        ),
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=DEFAULT_DOWNLOAD_DIR,
        help=(
            "Directory for downloaded Hugging Face parquet shards when --source hf. "
            f"Default: {DEFAULT_DOWNLOAD_DIR}."
        ),
    )
    parser.add_argument(
        "--parquet-root",
        type=Path,
        default=DEFAULT_PARQUET_ROOT,
        help=(
            "Local parquet file or directory to sample recursively when --source local. "
            "Defaults to TOKENIZER_DATASET_BASE, or the tokenizer-training dataset path used "
            f"by other scripts: {DEFAULT_PARQUET_ROOT}"
        ),
    )
    parser.add_argument(
        "--file-count",
        type=positive_int,
        default=5,
        help="Number of parquet files to sample. Default: 5.",
    )
    parser.add_argument(
        "--docs-per-file",
        type=positive_int,
        default=5,
        help="Number of documents to sample from each parquet file. Default: 5.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for file and document sampling. Default: 42.",
    )
    parser.add_argument(
        "--text-column",
        default=None,
        help=(
            "Text column to tokenize. If omitted, the script tries text, content, "
            "code, then the first string column."
        ),
    )
    parser.add_argument(
        "--lp-tokenizer",
        type=Path,
        default=DEFAULT_LP_TOKENIZER,
        help=f"LP tokenizer directory or tokenizer.json file. Default: {DEFAULT_LP_TOKENIZER}",
    )
    parser.add_argument(
        "--bpe-tokenizer",
        type=Path,
        default=DEFAULT_BPE_TOKENIZER,
        help=f"BPE tokenizer directory or tokenizer.json file. Default: {DEFAULT_BPE_TOKENIZER}",
    )
    parser.add_argument(
        "--max-tokens",
        type=positive_int,
        default=80,
        help="Maximum token positions to print per tokenizer per document. Default: 80.",
    )
    parser.add_argument(
        "--preview-chars",
        type=positive_int,
        default=800,
        help="Maximum source-text characters to print per document. Default: 800.",
    )
    parser.add_argument(
        "--max-token-width",
        type=positive_int,
        default=36,
        help="Maximum printed width for each token in side-by-side rows. Default: 36.",
    )
    parser.add_argument(
        "--show-ids",
        action="store_true",
        help="Include token ids alongside token strings.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for a Markdown report. The report is also printed to stdout.",
    )
    return parser.parse_args()


def import_runtime_dependencies():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install the project environment or run "
            "`pip install datasets pyarrow`."
        ) from exc

    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: tokenizers. Install the project environment or run "
            "`pip install tokenizers`."
        ) from exc

    return load_dataset, Tokenizer


def import_huggingface_dependencies():
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: huggingface_hub. Install the project environment "
            "or run `pip install huggingface_hub`."
        ) from exc

    return hf_hub_download, list_repo_files


def resolve_tokenizer_json(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_dir():
        expanded = expanded / "tokenizer.json"
    if not expanded.is_file():
        raise FileNotFoundError(f"Tokenizer JSON not found: {expanded}")
    return expanded.resolve()


def discover_parquet_files(path: Path) -> list[Path]:
    expanded = path.expanduser()
    if expanded.is_file():
        if expanded.suffix != ".parquet":
            raise ValueError(f"Expected a .parquet file, got: {expanded}")
        return [expanded.resolve()]
    if not expanded.is_dir():
        raise FileNotFoundError(f"Parquet root does not exist: {expanded}")

    parquet_files = sorted(p.resolve() for p in expanded.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under: {expanded}")
    return parquet_files


def sample_parquet_files(files: list[Path], file_count: int, seed: int) -> list[Path]:
    if file_count > len(files):
        raise ValueError(
            f"Requested {file_count} parquet files, but only found {len(files)}."
        )
    rng = random.Random(seed)
    return rng.sample(files, file_count)


def safe_path_component(value: str) -> str:
    return value.replace("/", "__").replace("\\", "__").replace(":", "_")


def list_hf_parquet_shards(
    dataset_id: str,
    revision: str | None,
    list_repo_files: Any,
) -> list[str]:
    print(f"Listing parquet shards for {dataset_id}...")
    shards = sorted(
        path
        for path in list_repo_files(
            dataset_id,
            repo_type="dataset",
            revision=revision,
        )
        if path.endswith(".parquet")
    )
    if not shards:
        raise ValueError(f"No parquet shards found in Hugging Face dataset: {dataset_id}")
    return shards


def sample_hf_parquet_shards(shards: list[str], file_count: int, seed: int) -> list[str]:
    if file_count > len(shards):
        raise ValueError(
            f"Requested {file_count} parquet shards, but {len(shards)} are available."
        )
    rng = random.Random(seed)
    return rng.sample(shards, file_count)


def download_hf_parquet_shards(
    dataset_id: str,
    revision: str | None,
    shards: list[str],
    download_dir: Path,
    hf_hub_download: Any,
) -> list[Path]:
    revision_label = safe_path_component(revision or "default")
    target_dir = (
        download_dir.expanduser()
        / safe_path_component(dataset_id)
        / revision_label
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    local_paths = []
    for shard in shards:
        print(f"Downloading {dataset_id}:{shard} -> {target_dir}")
        local_path = hf_hub_download(
            repo_id=dataset_id,
            filename=shard,
            repo_type="dataset",
            revision=revision,
            local_dir=str(target_dir),
        )
        local_paths.append(Path(local_path).resolve())
    return local_paths


def select_parquet_files(args: argparse.Namespace) -> tuple[int, list[Path], list[str]]:
    if args.source == "local":
        parquet_files = discover_parquet_files(args.parquet_root)
        selected_files = sample_parquet_files(parquet_files, args.file_count, args.seed)
        selected_labels = [display_path(path) for path in selected_files]
        return len(parquet_files), selected_files, selected_labels

    hf_hub_download, list_repo_files = import_huggingface_dependencies()
    shards = list_hf_parquet_shards(args.dataset_id, args.revision, list_repo_files)
    selected_shards = sample_hf_parquet_shards(shards, args.file_count, args.seed)
    print(f"Selected {len(selected_shards)} of {len(shards)} shards:")
    for shard in selected_shards:
        print(f"  - {shard}")
    selected_files = download_hf_parquet_shards(
        args.dataset_id,
        args.revision,
        selected_shards,
        args.download_dir,
        hf_hub_download,
    )
    return len(shards), selected_files, selected_shards


def infer_text_column(dataset: Any, configured_text_column: str | None) -> str:
    if configured_text_column is not None:
        if configured_text_column not in dataset.column_names:
            raise ValueError(
                f"Configured text column '{configured_text_column}' is not present. "
                f"Available columns: {dataset.column_names}"
            )
        return configured_text_column

    for column in PREFERRED_TEXT_COLUMNS:
        if column in dataset.column_names:
            return column

    for name, feature in dataset.features.items():
        dtype = getattr(feature, "dtype", None)
        if dtype in {"string", "large_string"}:
            return name

    if dataset.column_names:
        return dataset.column_names[0]
    raise ValueError("Dataset has no columns")


def sample_documents_from_file(
    load_dataset: Any,
    parquet_file: Path,
    file_number: int,
    docs_per_file: int,
    text_column: str | None,
    seed: int,
) -> list[DocumentSample]:
    dataset = load_dataset("parquet", data_files=str(parquet_file), split="train")
    row_count = len(dataset)
    if row_count == 0:
        raise ValueError(f"Parquet file has no rows: {parquet_file}")

    column = infer_text_column(dataset, text_column)
    docs_to_sample = min(docs_per_file, row_count)
    rng = random.Random(seed)
    row_indices = rng.sample(range(row_count), docs_to_sample)

    samples = []
    for document_number, row_index in enumerate(row_indices, start=1):
        value = dataset[int(row_index)][column]
        text = "" if value is None else str(value)
        samples.append(
            DocumentSample(
                file_number=file_number,
                document_number=document_number,
                file_path=parquet_file,
                row_index=int(row_index),
                row_count=row_count,
                text_column=column,
                text=text,
            )
        )
    return samples


def encode_without_special_tokens(tokenizer: Any, text: str) -> tuple[list[str], list[int]]:
    encoding = tokenizer.encode(text, add_special_tokens=False)
    return list(encoding.tokens), list(encoding.ids)


def first_difference(left: list[str], right: list[str]) -> int | None:
    for index in range(max(len(left), len(right))):
        if index >= len(left) or index >= len(right) or left[index] != right[index]:
            return index
    return None


def display_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def truncate(value: str, max_width: int) -> str:
    if len(value) <= max_width:
        return value
    if max_width <= 3:
        return value[:max_width]
    return value[: max_width - 3] + "..."


def render_token(token: str, token_id: int | None, show_ids: bool, max_width: int) -> str:
    value = repr(token)
    if show_ids and token_id is not None:
        value = f"{token_id}:{value}"
    return truncate(value, max_width)


def render_token_list(
    label: str,
    tokens: list[str],
    ids: list[int],
    max_tokens: int,
    show_ids: bool,
    max_token_width: int,
) -> list[str]:
    limit = min(max_tokens, len(tokens))
    items = [
        render_token(tokens[index], ids[index], show_ids, max_token_width)
        for index in range(limit)
    ]
    if limit < len(tokens):
        items.append(f"... ({len(tokens) - limit} more)")

    body = ", ".join(items) if items else "<no tokens>"
    wrapped = textwrap.wrap(
        body,
        width=120,
        initial_indent="  ",
        subsequent_indent="  ",
        break_long_words=False,
        break_on_hyphens=False,
    )
    return [f"{label} tokens ({len(tokens)} total):", *wrapped]


def render_side_by_side(
    comparison: TokenizedSample,
    max_tokens: int,
    show_ids: bool,
    max_token_width: int,
) -> list[str]:
    lp_tokens = comparison.lp_tokens
    bpe_tokens = comparison.bpe_tokens
    lp_ids = comparison.lp_ids
    bpe_ids = comparison.bpe_ids
    rows = min(max_tokens, max(len(lp_tokens), len(bpe_tokens)))
    width = max_token_width

    lines = [
        f"Side-by-side token positions (first {rows}):",
        f"{'idx':>5}  {'LP':<{width}}  {'BPE':<{width}}  match",
        f"{'-' * 5}  {'-' * width}  {'-' * width}  -----",
    ]

    for index in range(rows):
        lp_cell = ""
        bpe_cell = ""
        if index < len(lp_tokens):
            lp_cell = render_token(
                lp_tokens[index],
                lp_ids[index] if index < len(lp_ids) else None,
                show_ids,
                width,
            )
        if index < len(bpe_tokens):
            bpe_cell = render_token(
                bpe_tokens[index],
                bpe_ids[index] if index < len(bpe_ids) else None,
                show_ids,
                width,
            )
        match = (
            "yes"
            if index < len(lp_tokens)
            and index < len(bpe_tokens)
            and lp_tokens[index] == bpe_tokens[index]
            else "no"
        )
        lines.append(f"{index:>5}  {lp_cell:<{width}}  {bpe_cell:<{width}}  {match}")

    return lines


def indent_text_preview(text: str, max_chars: int) -> list[str]:
    preview = text[:max_chars]
    truncated = len(text) > max_chars
    if not preview:
        lines = ["    <empty>"]
    else:
        lines = [f"    {line}" for line in preview.splitlines()]
    if truncated:
        lines.append(f"    ... ({len(text) - max_chars} characters omitted)")
    return lines


def render_report(
    args: argparse.Namespace,
    parquet_file_count: int,
    selected_files: list[Path],
    selected_file_labels: list[str],
    lp_tokenizer_json: Path,
    bpe_tokenizer_json: Path,
    comparisons: list[TokenizedSample],
) -> str:
    lines = [
        "# LP vs BPE Tokenizer Comparison",
        "",
        f"Source: `{args.source}`",
    ]

    if args.source == "hf":
        lines.extend(
            [
                f"Hugging Face dataset: `{args.dataset_id}`",
                f"Revision: `{args.revision}`",
                f"Download dir: `{args.download_dir.expanduser()}`",
            ]
        )
    else:
        lines.append(f"Parquet root: `{args.parquet_root.expanduser()}`")

    lines.extend(
        [
            f"Discovered parquet files: `{parquet_file_count}`",
            f"Sampled parquet files: `{len(selected_files)}`",
            f"Documents per file: `{args.docs_per_file}`",
            f"Seed: `{args.seed}`",
            f"LP tokenizer: `{display_path(lp_tokenizer_json)}`",
            f"BPE tokenizer: `{display_path(bpe_tokenizer_json)}`",
            f"Text column: `{args.text_column or 'auto'}`",
            "",
            "## Sampled Files",
            "",
        ]
    )

    for index, (file_path, label) in enumerate(
        zip(selected_files, selected_file_labels),
        start=1,
    ):
        if args.source == "hf":
            lines.append(f"{index}. `{label}` -> `{display_path(file_path)}`")
        else:
            lines.append(f"{index}. `{label}`")

    for global_index, comparison in enumerate(comparisons, start=1):
        sample = comparison.sample
        lp_count = len(comparison.lp_tokens)
        bpe_count = len(comparison.bpe_tokens)
        delta = lp_count - bpe_count
        if bpe_count:
            delta_pct = (delta / bpe_count) * 100
            delta_text = f"{delta:+d} ({delta_pct:+.2f}% vs BPE)"
        else:
            delta_text = f"{delta:+d} (BPE produced no tokens)"
        diff_index = first_difference(comparison.lp_tokens, comparison.bpe_tokens)
        diff_text = "none" if diff_index is None else str(diff_index)

        lines.extend(
            [
                "",
                f"## Document {global_index}",
                "",
                f"File: `{display_path(sample.file_path)}`",
                f"File sample: `{sample.file_number}/{len(selected_files)}`",
                f"Document sample: `{sample.document_number}/{args.docs_per_file}`",
                f"Row index: `{sample.row_index}` of `{sample.row_count}`",
                f"Text column: `{sample.text_column}`",
                f"Characters: `{len(sample.text)}`",
                f"Token count delta: `{delta_text}`",
                f"First differing token position: `{diff_text}`",
                "",
                "Text preview:",
                *indent_text_preview(sample.text, args.preview_chars),
                "",
            ]
        )
        lines.extend(
            render_token_list(
                "LP",
                comparison.lp_tokens,
                comparison.lp_ids,
                args.max_tokens,
                args.show_ids,
                args.max_token_width,
            )
        )
        lines.append("")
        lines.extend(
            render_token_list(
                "BPE",
                comparison.bpe_tokens,
                comparison.bpe_ids,
                args.max_tokens,
                args.show_ids,
                args.max_token_width,
            )
        )
        lines.append("")
        lines.extend(
            render_side_by_side(
                comparison,
                args.max_tokens,
                args.show_ids,
                args.max_token_width,
            )
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    load_dataset, Tokenizer = import_runtime_dependencies()

    lp_tokenizer_json = resolve_tokenizer_json(args.lp_tokenizer)
    bpe_tokenizer_json = resolve_tokenizer_json(args.bpe_tokenizer)
    lp_tokenizer = Tokenizer.from_file(str(lp_tokenizer_json))
    bpe_tokenizer = Tokenizer.from_file(str(bpe_tokenizer_json))

    parquet_file_count, selected_files, selected_file_labels = select_parquet_files(args)

    comparisons = []
    for file_number, parquet_file in enumerate(selected_files, start=1):
        file_seed = args.seed + (file_number * 1_000_003)
        samples = sample_documents_from_file(
            load_dataset=load_dataset,
            parquet_file=parquet_file,
            file_number=file_number,
            docs_per_file=args.docs_per_file,
            text_column=args.text_column,
            seed=file_seed,
        )
        for sample in samples:
            lp_tokens, lp_ids = encode_without_special_tokens(lp_tokenizer, sample.text)
            bpe_tokens, bpe_ids = encode_without_special_tokens(bpe_tokenizer, sample.text)
            comparisons.append(
                TokenizedSample(
                    sample=sample,
                    lp_tokens=lp_tokens,
                    lp_ids=lp_ids,
                    bpe_tokens=bpe_tokens,
                    bpe_ids=bpe_ids,
                )
            )

    report = render_report(
        args=args,
        parquet_file_count=parquet_file_count,
        selected_files=selected_files,
        selected_file_labels=selected_file_labels,
        lp_tokenizer_json=lp_tokenizer_json,
        bpe_tokenizer_json=bpe_tokenizer_json,
        comparisons=comparisons,
    )

    if args.output is not None:
        output_path = args.output.expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        print(f"Wrote comparison report to {output_path}")

    print(report)


if __name__ == "__main__":
    main()
