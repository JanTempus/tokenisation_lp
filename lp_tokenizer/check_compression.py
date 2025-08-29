import os
from tqdm import tqdm
import numpy as np
import json
from datasets import load_from_disk, Dataset
from transformers import PreTrainedTokenizerFast
import csv

# Configuration
num_proc = 8
dataset_size = 65536
start=dataset_size
end=start+dataset_size
dataset_path = "finewebedu_data"
tokenizers_dir = "tokenizers_lp"  # directory containing all tokenizers
output_file = "dataset_stats_out_of_distribuition.csv"

# Load dataset slice
dataset = load_from_disk(dataset_path)['train'].select(range(start, end))

def process(batch, tokenizer):
    """Tokenize batch with Hugging Face tokenizer and split on EOS token ID."""
    texts = [t + "endoftextbehere" for t in batch["text"]]
    merged_text = "".join(texts)
    merged_ids = tokenizer.encode(merged_text)
    
    # Split on token ID 1
    split_ids = []
    current = []
    for token in merged_ids:
        if token == 1:
            split_ids.append(current)
            current = []
        else:
            current.append(token)
    
    return {
        "ids": split_ids,
        "len": [len(x) for x in split_ids],
    }

# Loop over all tokenizers
for tokenizer_name in os.listdir(tokenizers_dir):
    tokenizer_path = os.path.join(tokenizers_dir, tokenizer_name)
    if not os.path.isdir(tokenizer_path):
        continue  # skip files, only directories
    
    print(f"\nProcessing tokenizer: {tokenizer_name}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    
    # Tokenize dataset
    tokenized = dataset.map(
        lambda batch: process(batch, tokenizer),
        remove_columns=["text"],
        desc=f"Tokenizing with {tokenizer_name}",
        batched=True,
        batch_size=100,
        num_proc=num_proc,
    )
    
    # Compute stats
    total_tokens = int(np.sum(tokenized["len"]))
    stats = {
        "tokenizer": tokenizer_name,
        "total_tokens": total_tokens,
        "vocab_size": len(tokenizer),
        "dataset_size": dataset_size,
    }

    # Append to CSV
    file_exists = os.path.exists(output_file)
    with open(output_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=stats.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(stats)
    
    print(f"Stats written/appended: {stats}")
