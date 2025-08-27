import os
import csv
from datasets import load_from_disk, Dataset
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm
import numpy as np

dataset_path="/local/home/jtempus/token_lp/tokenisation_lp/lp_tokenizer/finewebedu_data"
tokenizers_dir = "bpe_tokenizers"
output_file = "dataset_stats_val.csv"


dataset_size=65536
dataset = load_from_disk(dataset_path)['train'].select(range(dataset_size, 2*dataset_size))
dataset_size=len(dataset)

# Number of parallel processes for map
num_proc = 8
batch_size = 1000  # size of batch passed to map

# CSV setup
file_exists = os.path.exists(output_file)
csv_fieldnames = ["tokenizer_name", "total_tokens", "vocab_size", "dataset_size"]
if not file_exists:
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
        writer.writeheader()

# Iterate over each tokenizer folder
tokenizer_folders = [os.path.join(tokenizers_dir, d) for d in os.listdir(tokenizers_dir)
                     if os.path.isdir(os.path.join(tokenizers_dir, d))]

for folder in tokenizer_folders:
    tokenizer_name = os.path.basename(folder)
    print(f"\nProcessing with tokenizer: {tokenizer_name}")

    tokenizer = PreTrainedTokenizerFast.from_pretrained(folder)

    def tokenize_batch(batch):
        # batch is a dict with key "text" -> list of strings
        encoded = tokenizer(batch["text"], add_special_tokens=False)["input_ids"]
        # Count tokens per row
        lengths = [len(x) for x in encoded]
        return {"len": lengths}

    # Run in parallel using map
    tokenized = dataset.map(
        tokenize_batch,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc=f"{tokenizer_name} - tokenizing",
    )

    total_tokens = int(sum(tokenized["len"]))

    # Write stats
    stats = {
        "tokenizer_name": tokenizer_name,
        "total_tokens": total_tokens,
        "vocab_size": tokenizer.vocab_size,
        "dataset_size": dataset_size,
    }

    with open(output_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
        writer.writerow(stats)

    print(f"Stats for {tokenizer_name} written/appended: {stats}")
