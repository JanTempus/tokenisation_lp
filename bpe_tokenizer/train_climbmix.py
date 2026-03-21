import os
import sys

from datasets import load_dataset, Dataset
from huggingface_hub import list_repo_files

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bpe_tokenizer import train_bpe_tokenizer

BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
DATASET_ID = "karpathy/climbmix-400b-shuffle"

NUM_SHARDS = int(os.getenv("NUM_SHARDS", "8"))
MAX_CHARS = int(os.getenv("MAX_CHARS", "2000000000"))
VOCAB_SIZE = int(os.getenv("VOCAB_SIZE", "32768"))
SAVE_DIR = os.getenv("SAVE_DIR", "bpe_tokenizers_climbmix")


def main():
    # List parquet shards from the Hub and take the first NUM_SHARDS
    print(f"Listing parquet shards for {DATASET_ID}...")
    all_shards = sorted(
        f for f in list_repo_files(DATASET_ID, repo_type="dataset")
        if f.endswith(".parquet")
    )
    selected = all_shards[:NUM_SHARDS]
    print(f"Using {len(selected)} shards: {selected}")

    shard_urls = [f"{BASE_URL}/{f}" for f in selected]

    # Load shards as a non-streaming dataset
    print("Loading shards...")
    raw_dataset = load_dataset("parquet", data_files={"train": shard_urls}, split="train")
    print(f"Loaded {len(raw_dataset):,} rows from {len(shard_urls)} shards")

    # Apply MAX_CHARS cap
    print(f"Collecting up to {MAX_CHARS:,} characters...")
    texts = []
    total_chars = 0
    for item in raw_dataset:
        text = item["text"]
        texts.append(text)
        total_chars += len(text)
        if total_chars >= MAX_CHARS:
            break

    print(f"Collected {len(texts):,} documents, {total_chars:,} characters")

    dataset = Dataset.from_dict({"text": texts})

    print(f"Training BPE tokenizer (vocab_size={VOCAB_SIZE})...")
    train_bpe_tokenizer(VOCAB_SIZE, dataset, SAVE_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
