import os
from tqdm import tqdm
import numpy as np
import json
from datasets import load_from_disk, Dataset
from lp_tokenizer import Tokenizer
import csv

# ------------------------------
# Config
# ------------------------------
dataset_size = 65536
vocab_size = 1024
chunk_size = 20000  # how many examples to merge together

dataset_path = "finewebedu_data"
vocab_file = f"vocabs/vocab_finewebedu_data_0_{vocab_size}.json"
output_file = "dataset_stats.csv"

# ------------------------------
# Load vocab
# ------------------------------
with open(vocab_file, "r", encoding="utf-8") as f:
    vocab = json.load(f)

tokenizer = Tokenizer(vocab_size=vocab_size, vocab=vocab, unk_token="[UNK]")

# ------------------------------
# Load dataset
# ------------------------------
dataset = load_from_disk(dataset_path)["train"].select(range(dataset_size))

# ------------------------------
# Merge texts into chunks
# ------------------------------
merged_texts = []
for i in tqdm(range(0, len(dataset), chunk_size), desc="Merging texts"):
    chunk = [dataset[j]["text"] for j in range(i, min(i + chunk_size, len(dataset)))]
    merged_text = "<|endoftext|>".join(chunk)
    merged_texts.append(merged_text)

# ------------------------------
# Tokenize merged texts
# ------------------------------
all_ids = []
all_lens = []

for text in merged_texts:
    ids = tokenizer.encode(text, vocab)
    all_ids.append(ids)
    all_lens.append(len(ids))

# ------------------------------
# Prepare Dataset (optional)
# ------------------------------
tokenized_dataset = Dataset.from_dict({
    "ids": all_ids,
    "len": all_lens
})

# ------------------------------
# Compute stats
# ------------------------------
total_tokens = int(np.sum(all_lens))
stats = {
    "total_tokens": total_tokens,
    "vocab_size": vocab_size,
    "dataset_size": dataset_size,
}

print(stats)