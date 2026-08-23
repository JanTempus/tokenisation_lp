#!/usr/bin/env python3
"""Convert a Hugging Face BPE tokenizer to an equal-score Unigram tokenizer.

The BPE vocabulary and token IDs are preserved.  Only the model is changed:
BPE merges are discarded and every vocabulary item receives a score of -1.0.
If the BPE has no unknown token, ``<|unk|>`` is appended to the vocabulary so
the Unigram model has a valid ``unk_id``.  The normalizer, pre-tokenizer,
post-processor, decoder, and added-token metadata are retained from the source
``tokenizer.json``.

Examples:

    # Convert one tokenizer directory.
    python bpe_vocab_to_unigram.py \
        bpe_tokenizers_climbmix/nano_bpe_climb_mix/8192 \
        baseline_tokenisers/bpe_vocab_unigram_8192

    # Convert every tokenizer.json below a directory, preserving subdirectories.
    python bpe_vocab_to_unigram.py \
        bpe_tokenizers_climbmix/nano_bpe_climb_mix \
        baseline_tokenisers/bpe_vocab_unigram
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


TOKENIZER_FILENAME = "tokenizer.json"
SIDECAR_FILENAMES = (
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
)


def find_tokenizers(input_path: Path) -> list[tuple[Path, Path]]:
    """Return ``(tokenizer_json, relative_output_dir)`` pairs."""
    if input_path.is_file():
        return [(input_path, Path())]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input does not exist: {input_path}")

    direct_tokenizer = input_path / TOKENIZER_FILENAME
    if direct_tokenizer.is_file():
        return [(direct_tokenizer, Path())]

    tokenizer_files = sorted(input_path.rglob(TOKENIZER_FILENAME))
    if not tokenizer_files:
        raise FileNotFoundError(
            f"No {TOKENIZER_FILENAME} files found below: {input_path}"
        )
    return [
        (path, path.parent.relative_to(input_path)) for path in tokenizer_files
    ]


def ordered_bpe_tokens(model: dict[str, Any], source: Path) -> list[str]:
    """Validate a BPE model and return tokens in token-ID order."""
    if model.get("type") != "BPE":
        raise ValueError(
            f"Expected a BPE model in {source}, found {model.get('type')!r}"
        )

    vocab = model.get("vocab")
    if not isinstance(vocab, dict) or not vocab:
        raise ValueError(f"BPE model in {source} has no non-empty vocab mapping")
    if not all(isinstance(token, str) for token in vocab):
        raise ValueError(f"BPE vocab in {source} contains a non-string token")
    if not all(isinstance(token_id, int) for token_id in vocab.values()):
        raise ValueError(f"BPE vocab in {source} contains a non-integer token ID")

    expected_ids = set(range(len(vocab)))
    actual_ids = set(vocab.values())
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)[:10]
        unexpected = sorted(actual_ids - expected_ids)[:10]
        raise ValueError(
            f"BPE token IDs in {source} must be unique and contiguous from 0; "
            f"missing={missing}, unexpected={unexpected}"
        )

    return [token for token, _ in sorted(vocab.items(), key=lambda item: item[1])]


def ensure_special_added_token(
    document: dict[str, Any], token: str, token_id: int, source: Path
) -> None:
    """Ensure the unknown token is also registered as a special added token."""
    added_tokens = document.setdefault("added_tokens", [])
    if not isinstance(added_tokens, list):
        raise ValueError(f"added_tokens in {source} is not a list")

    for added_token in added_tokens:
        if isinstance(added_token, dict) and added_token.get("content") == token:
            if added_token.get("id") != token_id:
                raise ValueError(
                    f"Added token {token!r} in {source} has ID "
                    f"{added_token.get('id')}, but its vocabulary ID is {token_id}"
                )
            added_token["special"] = True
            return

    added_tokens.append(
        {
            "id": token_id,
            "content": token,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
    )


def convert_document(
    document: dict[str, Any], source: Path, default_unk_token: str
) -> tuple[dict[str, Any], int, str, bool]:
    """Replace the BPE model in a tokenizer document with Unigram."""
    model = document.get("model")
    if not isinstance(model, dict):
        raise ValueError(f"Missing model object in {source}")

    tokens = ordered_bpe_tokens(model, source)
    token_to_id = {token: token_id for token_id, token in enumerate(tokens)}

    # Use the BPE's configured unknown token when it has one; otherwise use the
    # requested default. Append it rather than inserting it so every original
    # BPE token keeps its ID.
    model_unk_token = model.get("unk_token")
    unk_token = (
        model_unk_token
        if isinstance(model_unk_token, str) and model_unk_token
        else default_unk_token
    )
    added_unk = unk_token not in token_to_id
    if added_unk:
        token_to_id[unk_token] = len(tokens)
        tokens.append(unk_token)
    unk_id = token_to_id[unk_token]
    ensure_special_added_token(document, unk_token, unk_id, source)

    document["model"] = {
        "type": "Unigram",
        "unk_id": unk_id,
        "vocab": [[token, -1.0] for token in tokens],
        "byte_fallback": bool(model.get("byte_fallback", False)),
    }
    return document, len(tokens), unk_token, added_unk


def write_json(document: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(destination)


def convert_tokenizer(
    source: Path,
    output_dir: Path,
    overwrite: bool,
    default_unk_token: str,
) -> None:
    destination = output_dir / TOKENIZER_FILENAME
    if source.resolve() == destination.resolve():
        raise ValueError(f"Refusing to overwrite the source tokenizer in place: {source}")
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {destination} (pass --overwrite to replace it)"
        )

    with source.open(encoding="utf-8") as handle:
        document = json.load(handle)
    document, vocab_size, unk_token, added_unk = convert_document(
        document, source, default_unk_token
    )
    write_json(document, destination)

    for filename in SIDECAR_FILENAMES:
        sidecar = source.parent / filename
        if sidecar.is_file():
            shutil.copy2(sidecar, output_dir / filename)

    unk_description = f"{unk_token!r} at ID {document['model']['unk_id']}"
    if added_unk:
        unk_description += " (appended)"
    print(
        f"Converted {source} -> {destination} "
        f"(vocab={vocab_size:,}, score=-1.0, unk={unk_description})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help=(
            "A BPE tokenizer.json, a tokenizer directory, or a parent directory "
            "containing tokenizer.json files"
        ),
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory (source subdirectories are preserved for batch conversion)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace tokenizer.json files that already exist in the output",
    )
    parser.add_argument(
        "--unk-token",
        default="<|unk|>",
        help=(
            "Unknown token to append when the BPE model does not configure one "
            "(default: <|unk|>)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer_files = find_tokenizers(args.input)
    for source, relative_dir in tokenizer_files:
        convert_tokenizer(
            source=source,
            output_dir=args.output_dir / relative_dir,
            overwrite=args.overwrite,
            default_unk_token=args.unk_token,
        )


if __name__ == "__main__":
    main()
