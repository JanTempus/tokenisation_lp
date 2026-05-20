#!/usr/bin/env python3
"""Upload rounded tokenizer artifacts to the Hugging Face Hub.

Example:
    python upload_tokenizers_to_hf.py \
        --repo-id YOUR_USERNAME/cross-over-climbmix400b-s7-tokenizers

Authentication is handled by huggingface_hub. Set HF_TOKEN in the environment,
or run `huggingface-cli login` before using this script.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


DEFAULT_TOKENIZER_DIR = (
    Path(__file__).resolve().parent
    / "rounded_tokenizers"
    / "cross_over_climbmix400b_s7"
)
REQUIRED_FILES = ("tokenizer.json", "tokenizer_config.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a directory of tokenizer artifacts to Hugging Face."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_TOKENIZER_DIR,
        help=f"Tokenizer root directory to upload. Default: {DEFAULT_TOKENIZER_DIR}",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Destination Hub repo, for example 'username/cross-over-climbmix400b-s7-tokenizers'.",
    )
    parser.add_argument(
        "--repo-type",
        default="model",
        choices=("model", "dataset", "space"),
        help="Hub repository type. Tokenizers usually belong in a model repo.",
    )
    parser.add_argument(
        "--path-in-repo",
        default=".",
        help="Destination path inside the Hub repo. Default uploads at repo root.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Branch or revision to upload to. Defaults to the repo default branch.",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload rounded cross_over_climbmix400b_s7 tokenizers",
        help="Hub commit message.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the Hub repo as private if it does not already exist.",
    )
    parser.add_argument(
        "--no-readme",
        action="store_true",
        help="Skip generating a README.md in the destination repo.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print discovered tokenizers and exit without uploading.",
    )
    return parser.parse_args()


def find_tokenizer_dirs(input_dir: Path) -> list[Path]:
    tokenizer_dirs = []
    for tokenizer_json in sorted(input_dir.glob("vocab_*/lp_*/tokenizer.json")):
        candidate = tokenizer_json.parent
        if all((candidate / filename).is_file() for filename in REQUIRED_FILES):
            tokenizer_dirs.append(candidate)
    return tokenizer_dirs


def validate_tokenizer_dir(tokenizer_dir: Path) -> None:
    for filename in REQUIRED_FILES:
        path = tokenizer_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing required file: {path}")

        try:
            json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def build_readme(input_dir: Path, tokenizer_dirs: list[Path]) -> str:
    rows = []
    for tokenizer_dir in tokenizer_dirs:
        rel = tokenizer_dir.relative_to(input_dir)
        rows.append(f"- `{rel.as_posix()}`")

    tokenizer_list = "\n".join(rows)
    return f"""---
license: other
tags:
- tokenizer
- tokenizers
- lp-tokenizer
- climbmix
---

# Rounded LP Tokenizers

This repository contains rounded tokenizer artifacts from `{input_dir.name}`.

Each tokenizer directory contains:

- `tokenizer.json`
- `tokenizer_config.json`

## Included Tokenizers

{tokenizer_list}
"""


def upload_readme(api, args: argparse.Namespace, readme: str) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "README.md"
        path.write_text(readme, encoding="utf-8")
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=args.revision,
            commit_message="Add tokenizer README",
        )


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Tokenizer directory does not exist: {input_dir}")

    tokenizer_dirs = find_tokenizer_dirs(input_dir)
    if not tokenizer_dirs:
        raise ValueError(f"No tokenizer directories found under {input_dir}")

    for tokenizer_dir in tokenizer_dirs:
        validate_tokenizer_dir(tokenizer_dir)

    print(f"Found {len(tokenizer_dirs)} tokenizers under {input_dir}:")
    for tokenizer_dir in tokenizer_dirs:
        print(f"  - {tokenizer_dir.relative_to(input_dir)}")

    if args.dry_run:
        print("Dry run complete; no files were uploaded.")
        return

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: huggingface_hub. Install it with "
            "`pip install huggingface_hub`."
        ) from exc

    if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGINGFACE_HUB_TOKEN"):
        print(
            "HF_TOKEN is not set. Continuing; this will work if you are already "
            "logged in with `huggingface-cli login`."
        )

    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        private=args.private,
        exist_ok=True,
    )

    api.upload_folder(
        folder_path=str(input_dir),
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        path_in_repo=args.path_in_repo,
        revision=args.revision,
        commit_message=args.commit_message,
        allow_patterns=["vocab_*/lp_*/*.json"],
    )

    if not args.no_readme:
        upload_readme(api, args, build_readme(input_dir, tokenizer_dirs))

    print(f"Uploaded tokenizers to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()

