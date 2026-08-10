import os
from pathlib import Path
import sys

from datasets import load_from_disk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hf_baseline_tokenizers import train_tokenizer


def train_requested_tokenizers(dataset, vocab_sizes, save_dir, tokenizer_type="bpe"):
    if "text" not in dataset.column_names:
        raise ValueError(
            f"Training dataset must contain a 'text' column, got {dataset.column_names}"
        )
    if not vocab_sizes:
        raise ValueError("At least one vocabulary size is required")

    for vocab_size in vocab_sizes:
        tokenizer_json = Path(save_dir) / f"{tokenizer_type}_{vocab_size}" / "tokenizer.json"
        if tokenizer_json.is_file():
            print(
                f"Skipping {tokenizer_type.upper()} vocab_size={vocab_size}; "
                f"completed tokenizer exists at {tokenizer_json}"
            )
            continue

        print(f"Training {tokenizer_type.upper()} vocab_size={vocab_size}...")
        train_tokenizer(vocab_size, dataset, save_dir, tokenizer_type)
        if not tokenizer_json.is_file():
            raise RuntimeError(
                f"{tokenizer_type.upper()} training did not create expected output: "
                f"{tokenizer_json}"
            )
        print(f"Done vocab_size={vocab_size}")


def main():
    tokenizer_type = os.environ.get("TOKENIZER_TYPE", "bpe").strip().lower()
    train_dataset_path = os.environ["TRAIN_DATASET_PATH"]
    vocab_sizes = [
        int(value)
        for value in os.environ["VOCAB_SIZES"].split(",")
        if value.strip()
    ]
    save_dir = os.environ["SAVE_DIR"]

    dataset = load_from_disk(train_dataset_path)
    print(f"Loaded {len(dataset):,} rows from {train_dataset_path}")
    train_requested_tokenizers(dataset, vocab_sizes, save_dir, tokenizer_type)


if __name__ == "__main__":
    main()
