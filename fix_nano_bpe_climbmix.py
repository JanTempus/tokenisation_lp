#!/usr/bin/env python3
"""Repair NanoChat ClimbMix BPE tokenizers that are missing ``<|unk|>``.

For each tokenizer, the pipeline removes the final BPE merge and the token
created by that merge, then assigns the freed final token ID to ``<|unk|>``.
This keeps the vocabulary size and every other token ID unchanged.

With no arguments, the repository's tokenizer sweep is repaired into:

    baseline_tokenisers/nano_bpe_climbmix/<vocab_size>/

The repaired vocabularies are then converted to equal-score Unigram models in:

    baseline_tokenisers/bpe_vocab_unigram/<vocab_size>/

Custom source and destination roots can be supplied as positional arguments.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from bpe_vocab_to_unigram import convert_tokenizer as convert_to_unigram


DEFAULT_INPUT_ROOT = Path("bpe_tokenizers_climbmix/nano_bpe_climb_mix")
DEFAULT_OUTPUT_ROOT = Path("baseline_tokenisers/nano_bpe_climbmix")
DEFAULT_UNIGRAM_OUTPUT_ROOT = Path("baseline_tokenisers/bpe_vocab_unigram")
UNK_TOKEN = "<|unk|>"
TOKENIZER_FILENAME = "tokenizer.json"
SIDECAR_FILENAMES = (
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
)


def find_tokenizers(input_root: Path) -> list[tuple[Path, Path]]:
    """Find tokenizer files and their relative output directories."""
    if input_root.is_file():
        return [(input_root, Path())]
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input does not exist: {input_root}")

    direct = input_root / TOKENIZER_FILENAME
    if direct.is_file():
        return [(direct, Path())]

    tokenizer_files = sorted(input_root.rglob(TOKENIZER_FILENAME))
    if not tokenizer_files:
        raise FileNotFoundError(
            f"No {TOKENIZER_FILENAME} files found below {input_root}"
        )
    return [
        (path, path.parent.relative_to(input_root)) for path in tokenizer_files
    ]


def final_merge_parts(merge: Any, source: Path) -> tuple[str, str]:
    """Read the final merge from either supported tokenizer JSON encoding."""
    if (
        isinstance(merge, list)
        and len(merge) == 2
        and all(isinstance(part, str) for part in merge)
    ):
        return merge[0], merge[1]
    if isinstance(merge, str):
        parts = merge.split(" ")
        if len(parts) == 2 and all(parts):
            return parts[0], parts[1]
    raise ValueError(f"Unsupported final merge representation in {source}: {merge!r}")


def register_unk_added_token(
    document: dict[str, Any], unk_id: int, source: Path
) -> None:
    """Register ``<|unk|>`` as a special token with its model vocabulary ID."""
    added_tokens = document.setdefault("added_tokens", [])
    if not isinstance(added_tokens, list):
        raise ValueError(f"added_tokens in {source} is not a list")

    for token in added_tokens:
        if isinstance(token, dict) and token.get("content") == UNK_TOKEN:
            raise ValueError(f"{source} already has an added {UNK_TOKEN} token")

    if any(
        isinstance(token, dict) and token.get("id") == unk_id
        for token in added_tokens
    ):
        raise ValueError(f"Added-token ID {unk_id} is already occupied in {source}")

    added_tokens.append(
        {
            "id": unk_id,
            "content": UNK_TOKEN,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
    )


def repair_document(
    document: dict[str, Any], source: Path
) -> tuple[dict[str, Any], int, str, Any]:
    """Replace the last learned BPE token with ``<|unk|>``."""
    model = document.get("model")
    if not isinstance(model, dict) or model.get("type") != "BPE":
        model_type = model.get("type") if isinstance(model, dict) else None
        raise ValueError(f"Expected a BPE model in {source}, found {model_type!r}")

    vocab = model.get("vocab")
    merges = model.get("merges")
    if not isinstance(vocab, dict) or not vocab:
        raise ValueError(f"Missing non-empty BPE vocabulary in {source}")
    if not isinstance(merges, list) or not merges:
        raise ValueError(f"Missing non-empty BPE merges in {source}")
    if UNK_TOKEN in vocab:
        raise ValueError(f"{source} already contains {UNK_TOKEN}")

    ids = list(vocab.values())
    if not all(isinstance(token_id, int) for token_id in ids):
        raise ValueError(f"Vocabulary in {source} contains a non-integer token ID")
    if set(ids) != set(range(len(vocab))):
        raise ValueError(f"Vocabulary IDs in {source} are not unique and contiguous")

    final_merge = merges[-1]
    left, right = final_merge_parts(final_merge, source)
    final_token = left + right
    final_id = len(vocab) - 1

    if vocab.get(final_token) != final_id:
        token_at_final_id = next(
            (token for token, token_id in vocab.items() if token_id == final_id),
            None,
        )
        raise ValueError(
            f"Final merge {final_merge!r} creates {final_token!r}, but final "
            f"vocabulary ID {final_id} belongs to {token_at_final_id!r} in {source}"
        )

    merges.pop()
    del vocab[final_token]
    vocab[UNK_TOKEN] = final_id
    model["unk_token"] = UNK_TOKEN
    register_unk_added_token(document, final_id, source)

    return document, len(vocab), final_token, final_merge


def tokenizer_config(source_dir: Path) -> dict[str, Any]:
    """Load a source config or construct the NanoChat tokenizer config."""
    config_path = source_dir / "tokenizer_config.json"
    if config_path.is_file():
        with config_path.open(encoding="utf-8") as handle:
            config = json.load(handle)
    else:
        config = {
            "backend": "tokenizers",
            "bos_token": "<|bos|>",
            "extra_special_tokens": [
                "<|user_start|>",
                "<|user_end|>",
                "<|assistant_start|>",
                "<|assistant_end|>",
                "<|python_start|>",
                "<|python_end|>",
                "<|output_start|>",
                "<|output_end|>",
            ],
            "model_max_length": 1_000_000_000_000_000_019_884_624_838_656,
            "tokenizer_class": "TokenizersBackend",
        }
    config["unk_token"] = UNK_TOKEN
    return config


def write_json(document: dict[str, Any], destination: Path) -> None:
    """Write JSON atomically."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(destination)


def repair_tokenizer(source: Path, output_dir: Path, overwrite: bool) -> None:
    """Repair one tokenizer and write its Hugging Face tokenizer directory."""
    destination = output_dir / TOKENIZER_FILENAME
    if source.resolve() == destination.resolve():
        raise ValueError(f"Refusing to modify the source in place: {source}")
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Output exists: {destination} (pass --overwrite to replace it)"
        )

    with source.open(encoding="utf-8") as handle:
        document = json.load(handle)
    document, vocab_size, removed_token, removed_merge = repair_document(
        document, source
    )

    write_json(document, destination)
    write_json(tokenizer_config(source.parent), output_dir / "tokenizer_config.json")
    for filename in SIDECAR_FILENAMES:
        sidecar = source.parent / filename
        if sidecar.is_file():
            shutil.copy2(sidecar, output_dir / filename)

    print(
        f"Repaired {source} -> {destination} (vocab={vocab_size:,}, "
        f"unk_id={vocab_size - 1}, removed_token={removed_token!r}, "
        f"removed_merge={removed_merge!r})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_root",
        type=Path,
        nargs="?",
        default=DEFAULT_INPUT_ROOT,
        help=f"Source tokenizer root (default: {DEFAULT_INPUT_ROOT})",
    )
    parser.add_argument(
        "output_root",
        type=Path,
        nargs="?",
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Destination tokenizer root (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace tokenizer files that already exist in the destination",
    )
    parser.add_argument(
        "--unigram-output-root",
        type=Path,
        default=DEFAULT_UNIGRAM_OUTPUT_ROOT,
        help=(
            "Destination for equal-score Unigram tokenizers "
            f"(default: {DEFAULT_UNIGRAM_OUTPUT_ROOT})"
        ),
    )
    parser.add_argument(
        "--skip-unigram",
        action="store_true",
        help="Only repair the BPE tokenizers; do not create Unigram tokenizers",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer_files = find_tokenizers(args.input_root)
    for source, relative_dir in tokenizer_files:
        repair_tokenizer(
            source=source,
            output_dir=args.output_root / relative_dir,
            overwrite=args.overwrite,
        )

    if not args.skip_unigram:
        for _, relative_dir in tokenizer_files:
            repaired_tokenizer = (
                args.output_root / relative_dir / TOKENIZER_FILENAME
            )
            convert_to_unigram(
                source=repaired_tokenizer,
                output_dir=args.unigram_output_root / relative_dir,
                overwrite=args.overwrite,
                default_unk_token=UNK_TOKEN,
            )


if __name__ == "__main__":
    main()
