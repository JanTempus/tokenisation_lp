import os
import sys

from datasets import load_dataset, Dataset
from huggingface_hub import list_repo_files


BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
DATASET_ID = "karpathy/climbmix-400b-shuffle"

NUM_SHARDS = int(os.getenv("NUM_SHARDS", "8"))
MAX_CHARS = int(os.getenv("MAX_CHARS", "2000000000"))
VOCAB_SIZES = [int(v) for v in os.environ["VOCAB_SIZES"].split(",") if v.strip()]
SAVE_DIR = os.environ["SAVE_DIR"]
NANOCHAT_REPO = os.environ.get(
    "NANOCHAT_REPO", "/home/jantempus/Desktop/Projects/NLP/nanochat"
)

sys.path.insert(0, NANOCHAT_REPO)
from nanochat.tokenizer import HuggingFaceTokenizer  # noqa: E402


SAMPLE_TEXT = (
    "Hello world! naive cafe 中文 Ελληνικα. "
    "def f(x):\n    return x**2  # square\n"
)


def text_iterator(dataset):
    for row in dataset:
        yield row["text"]


def main():
    print(f"Listing parquet shards for {DATASET_ID}...")
    all_shards = sorted(
        f for f in list_repo_files(DATASET_ID, repo_type="dataset")
        if f.endswith(".parquet")
    )
    selected = all_shards[:NUM_SHARDS]
    print(f"Using {len(selected)} shards: {selected}")

    shard_urls = [f"{BASE_URL}/{f}" for f in selected]

    print("Loading shards...")
    raw_dataset = load_dataset("parquet", data_files={"train": shard_urls}, split="train")
    print(f"Loaded {len(raw_dataset):,} rows from {len(shard_urls)} shards")

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
    print(f"Using nanochat repo at {NANOCHAT_REPO}")

    for vocab_size in VOCAB_SIZES:
        print(f"\nTraining nano_bpe vocab_size={vocab_size}")
        tok = HuggingFaceTokenizer.train_from_iterator(
            text_iterator(dataset), vocab_size
        )

        save_path = os.path.join(SAVE_DIR, "nano_bpe_climb_mix", str(vocab_size))
        os.makedirs(save_path, exist_ok=True)
        tok.save(save_path)

        ids = tok.encode(SAMPLE_TEXT)
        decoded = tok.decode(ids)
        print(
            f"[SANITY] vocab_size={tok.get_vocab_size()} "
            f"sample_ids={len(ids)} round_trip={decoded == SAMPLE_TEXT}"
        )


if __name__ == "__main__":
    main()
