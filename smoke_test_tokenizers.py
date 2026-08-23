#!/usr/bin/env python3
"""Run lightweight loading and round-trip checks on tokenizer directories.

Each input may be a ``tokenizer.json``, one tokenizer directory, or a parent
directory containing a tokenizer sweep.

Examples:

    python3 smoke_test_tokenizers.py \
        baseline_tokenisers/nano_bpe_climbmix \
        baseline_tokenisers/bpe_vocab_unigram \
        --require-unk --require-hf

    python3 smoke_test_tokenizers.py baseline_tokenisers --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer


DEFAULT_SAMPLES = (
    "Hello, world!",
    "  leading space\tand tab\ntrailing  ",
    "naïve café 中文 Ελληνικά العربية 😀🚀",
    "def f(x):\n    return x**2  # square\n",
)


def discover_tokenizers(inputs: list[Path]) -> list[Path]:
    """Resolve files/directories to unique tokenizer.json paths."""
    discovered: dict[Path, Path] = {}
    for supplied_path in inputs:
        input_path = supplied_path
        if not input_path.exists():
            # ``\ path`` on one shell line escapes the space and produces an
            # argument with leading whitespace. Recover when the trimmed path
            # is unambiguous and actually exists.
            trimmed_path = Path(str(input_path).strip())
            if trimmed_path != input_path and trimmed_path.exists():
                print(
                    f"WARN treating {str(input_path)!r} as {str(trimmed_path)!r}",
                    file=sys.stderr,
                )
                input_path = trimmed_path

        if input_path.is_file():
            candidates = [input_path]
        elif input_path.is_dir() and (input_path / "tokenizer.json").is_file():
            candidates = [input_path / "tokenizer.json"]
        elif input_path.is_dir():
            candidates = sorted(input_path.rglob("tokenizer.json"))
        else:
            raise FileNotFoundError(
                f"Input does not exist: {input_path!s}. On one shell line, do "
                "not put a backslash before the path."
            )

        if not candidates:
            raise FileNotFoundError(f"No tokenizer.json found below: {input_path}")
        for candidate in candidates:
            discovered[candidate.resolve()] = candidate
    return [discovered[key] for key in sorted(discovered, key=str)]


def model_summary(document: dict[str, Any]) -> tuple[str, int, str | None, int | None]:
    """Return model type, model vocab size, unknown token, and unknown ID."""
    model = document.get("model")
    if not isinstance(model, dict):
        raise ValueError("tokenizer.json has no model object")

    model_type = model.get("type")
    vocab = model.get("vocab")
    if model_type == "BPE":
        if not isinstance(vocab, dict):
            raise ValueError("BPE vocabulary is not a mapping")
        unk_token = model.get("unk_token")
        unk_token = unk_token if isinstance(unk_token, str) else None
        unk_id = vocab.get(unk_token) if unk_token is not None else None
        return model_type, len(vocab), unk_token, unk_id

    if model_type == "Unigram":
        if not isinstance(vocab, list):
            raise ValueError("Unigram vocabulary is not a list")
        unk_id = model.get("unk_id")
        if unk_id is None:
            return model_type, len(vocab), None, None
        if not isinstance(unk_id, int) or not 0 <= unk_id < len(vocab):
            raise ValueError(f"Invalid Unigram unk_id: {unk_id!r}")
        entry = vocab[unk_id]
        if not isinstance(entry, list) or len(entry) != 2:
            raise ValueError(f"Invalid Unigram vocabulary entry at unk_id={unk_id}")
        return model_type, len(vocab), entry[0], unk_id

    raise ValueError(f"Unsupported model type: {model_type!r}")


def check_hugging_face_loading(
    tokenizer_dir: Path,
    backend_vocab_size: int,
    unk_token: str | None,
    unk_id: int | None,
) -> None:
    """Load with Transformers and check important wrapper metadata."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir,
        local_files_only=True,
    )
    if len(tokenizer) != backend_vocab_size:
        raise AssertionError(
            f"Transformers vocab size {len(tokenizer)} != backend vocab size "
            f"{backend_vocab_size}"
        )
    if unk_token is not None:
        if tokenizer.unk_token != unk_token:
            raise AssertionError(
                f"Transformers unk_token {tokenizer.unk_token!r} != {unk_token!r}"
            )
        if tokenizer.unk_token_id != unk_id:
            raise AssertionError(
                f"Transformers unk_token_id {tokenizer.unk_token_id} != {unk_id}"
            )


def smoke_test(
    tokenizer_path: Path,
    samples: tuple[str, ...],
    require_roundtrip: bool,
    require_unk: bool,
    require_hf: bool,
    verbose: bool,
) -> list[str]:
    """Test one tokenizer and return non-fatal warning messages."""
    with tokenizer_path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    model_type, model_vocab_size, unk_token, unk_id = model_summary(document)

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    backend_vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    if backend_vocab_size != model_vocab_size:
        raise AssertionError(
            f"Backend vocab size {backend_vocab_size} != model vocab size "
            f"{model_vocab_size}"
        )

    warnings: list[str] = []
    if unk_token is None or unk_id is None:
        message = "tokenizer has no configured unknown token"
        if require_unk:
            raise AssertionError(message)
        warnings.append(message)
    else:
        if tokenizer.token_to_id(unk_token) != unk_id:
            raise AssertionError(
                f"Backend ID for {unk_token!r} does not equal unk_id={unk_id}"
            )
        unk_encoding = tokenizer.encode(unk_token)
        if unk_encoding.ids != [unk_id]:
            raise AssertionError(
                f"Encoding {unk_token!r} produced {unk_encoding.ids}, "
                f"expected [{unk_id}]"
            )
        if tokenizer.decode([unk_id], skip_special_tokens=False) != unk_token:
            raise AssertionError(f"Unknown token ID {unk_id} does not decode correctly")

    for index, text in enumerate(samples, start=1):
        encoding = tokenizer.encode(text)
        if any(
            not isinstance(token_id, int)
            or token_id < 0
            or token_id >= backend_vocab_size
            for token_id in encoding.ids
        ):
            raise AssertionError(f"Sample {index} produced an out-of-range token ID")

        decoded = tokenizer.decode(encoding.ids, skip_special_tokens=False)
        if require_roundtrip and decoded != text:
            raise AssertionError(
                f"Sample {index} failed round trip:\n"
                f"  input:   {text!r}\n"
                f"  decoded: {decoded!r}"
            )
        if verbose:
            token_preview = encoding.tokens[:20]
            suffix = " ..." if len(encoding.tokens) > len(token_preview) else ""
            print(
                f"    sample {index}: ids={len(encoding.ids):>3} "
                f"tokens={token_preview!r}{suffix}"
            )

    config_path = tokenizer_path.parent / "tokenizer_config.json"
    if config_path.is_file():
        check_hugging_face_loading(
            tokenizer_path.parent,
            backend_vocab_size,
            unk_token,
            unk_id,
        )
    elif require_hf:
        raise AssertionError("tokenizer_config.json is missing")
    else:
        warnings.append("Transformers loading skipped: tokenizer_config.json is missing")

    unk_description = f"{unk_token}:{unk_id}" if unk_token is not None else "none"
    print(
        f"PASS {tokenizer_path.parent} "
        f"(model={model_type}, vocab={backend_vocab_size:,}, unk={unk_description})"
    )
    return warnings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "tokenizers",
        type=Path,
        nargs="+",
        help="Tokenizer files/directories or roots containing tokenizer directories",
    )
    parser.add_argument(
        "--text",
        action="append",
        default=[],
        help="Additional text to test; may be passed more than once",
    )
    parser.add_argument(
        "--no-roundtrip",
        action="store_true",
        help="Only require encoding/decoding to run, not exact text recovery",
    )
    parser.add_argument(
        "--require-unk",
        action="store_true",
        help="Fail tokenizers that have no configured unknown token",
    )
    parser.add_argument(
        "--require-hf",
        action="store_true",
        help="Require tokenizer_config.json and successful Transformers loading",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print a token preview for every sample",
    )

    # Be forgiving of ``\ path \ --flag`` entered on one line. In Bash those
    # backslashes escape spaces rather than continue a command, leaving each
    # argument with a leading space.
    cli_args: list[str] = []
    for argument in sys.argv[1:]:
        trimmed = argument.strip()
        if (
            argument != trimmed
            and not Path(argument).exists()
            and (trimmed.startswith("-") or Path(trimmed).exists())
        ):
            print(
                f"WARN treating argument {argument!r} as {trimmed!r}",
                file=sys.stderr,
            )
            argument = trimmed
        cli_args.append(argument)
    return parser.parse_args(cli_args)


def main() -> None:
    args = parse_args()
    samples = (*DEFAULT_SAMPLES, *args.text)
    tokenizer_paths = discover_tokenizers(args.tokenizers)
    failures = 0
    warning_count = 0

    print(f"Testing {len(tokenizer_paths)} tokenizer(s) with {len(samples)} sample(s)")
    for tokenizer_path in tokenizer_paths:
        try:
            warnings = smoke_test(
                tokenizer_path=tokenizer_path,
                samples=samples,
                require_roundtrip=not args.no_roundtrip,
                require_unk=args.require_unk,
                require_hf=args.require_hf,
                verbose=args.verbose,
            )
            for warning in warnings:
                warning_count += 1
                print(f"  WARN {tokenizer_path.parent}: {warning}")
        except Exception as error:
            failures += 1
            print(
                f"FAIL {tokenizer_path.parent}: "
                f"{type(error).__name__}: {error}",
                file=sys.stderr,
            )

    print(
        f"Summary: {len(tokenizer_paths) - failures} passed, "
        f"{failures} failed, {warning_count} warning(s)"
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
