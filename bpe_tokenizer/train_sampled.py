import os
import sys

from datasets import load_from_disk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bpe_tokenizer import train_bpe_tokenizer

TRAIN_DATASET_PATH = os.environ["TRAIN_DATASET_PATH"]
VOCAB_SIZES        = [int(v) for v in os.environ["VOCAB_SIZES"].split(",") if v.strip()]
SAVE_DIR           = os.environ["SAVE_DIR"]

dataset = load_from_disk(TRAIN_DATASET_PATH)
print(f"Loaded {len(dataset):,} rows from {TRAIN_DATASET_PATH}")

for vs in VOCAB_SIZES:
    print(f"Training BPE vocab_size={vs}...")
    train_bpe_tokenizer(vs, dataset, SAVE_DIR)
    print(f"Done vocab_size={vs}")
