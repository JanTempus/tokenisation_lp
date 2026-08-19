#!/usr/bin/env python3
"""Create a small, deterministic Hugging Face dataset for tokenizer testing."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset


SUBJECTS = (
    "tokenization",
    "language modeling",
    "information retrieval",
    "machine translation",
    "speech recognition",
    "document ranking",
    "data compression",
)

LANGUAGE_SAMPLES = (
    "English text with punctuation, contractions, and numbers.",
    "Deutsch: Grüße aus Zürich; größere Wörter werden geprüft.",
    "Français: un café, une naïve façade et déjà vu.",
    "Español: el pingüino camina rápidamente por la estación.",
    "日本語の短い文章と東京123。",
    "Ελληνικά: ένα μικρό δείγμα κειμένου.",
)


def build_text(row_number: int) -> str:
    subject = SUBJECTS[row_number % len(SUBJECTS)]
    language_sample = LANGUAGE_SAMPLES[row_number % len(LANGUAGE_SAMPLES)]
    experiment = row_number % 37
    score = (row_number * 17) % 101
    return (
        f"Document {row_number:04d}: experiment_{experiment} studies {subject}. "
        f"The measured score is {score}.{row_number % 10:02d}; "
        f"flags=[baseline, local-test, shard-{row_number % 5}]. "
        f"Code: result_{experiment} = tokenize(sample_{row_number % 23}). "
        f"{language_sample}\n"
        f"Second line {row_number % 11}: tabs\tspaces  repeated   whitespace!"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_path", type=Path)
    parser.add_argument(
        "--rows",
        type=int,
        default=256,
        help="Number of dataset rows to create (default: 256)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rows < 1:
        raise ValueError("--rows must be at least 1")
    if args.output_path.exists():
        raise FileExistsError(
            f"Output path already exists: {args.output_path}. "
            "Choose a new path or remove it explicitly."
        )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = Dataset.from_dict(
        {
            "text": [build_text(row_number) for row_number in range(args.rows)],
            "document_id": list(range(args.rows)),
        }
    )
    dataset.save_to_disk(args.output_path)
    print(dataset)
    print(f"Saved test dataset to: {args.output_path}")


if __name__ == "__main__":
    main()
