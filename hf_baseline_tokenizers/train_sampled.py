import os
import sys

from datasets import load_from_disk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hf_baseline_tokenizers import train_tokenizer


def train_requested_tokenizers(dataset, vocab_sizes, save_dir, tokenizer_type="bpe"):
    for vocab_size in vocab_sizes:
        print(f"Training {tokenizer_type.upper()} vocab_size={vocab_size}...")
        train_tokenizer(vocab_size, dataset, save_dir, tokenizer_type)
        print(f"Done vocab_size={vocab_size}")


def main():
    tokenizer_type = os.environ["TOKENIZER_TYPE"]
    train_dataset_path = os.environ["TRAIN_DATASET_PATH"]
    vocab_sizes = [
        int(value)
        for value in os.environ["VOCAB_SIZES"].split(",")
    ]
    save_dir = os.environ["SAVE_DIR"]

    dataset = load_from_disk(train_dataset_path)
    print(f"Loaded {len(dataset):,} rows from {train_dataset_path}")
    train_requested_tokenizers(dataset, vocab_sizes, save_dir, tokenizer_type)


if __name__ == "__main__":
    main()
