import os
import sys

from datasets import load_from_disk


TRAIN_DATASET_PATH = os.environ["TRAIN_DATASET_PATH"]
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
    dataset = load_from_disk(TRAIN_DATASET_PATH)
    print(f"Loaded dataset from {TRAIN_DATASET_PATH} ({len(dataset)} rows)")
    print(f"Using nanochat repo at {NANOCHAT_REPO}")

    for vocab_size in VOCAB_SIZES:
        print(f"\nTraining nano_bpe vocab_size={vocab_size}")
        tok = HuggingFaceTokenizer.train_from_iterator(
            text_iterator(dataset), vocab_size
        )

        save_path = os.path.join(SAVE_DIR, "nano_bpe", str(vocab_size))
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
