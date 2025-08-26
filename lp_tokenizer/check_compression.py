import os
from tqdm import tqdm
import numpy as np
import json
from datasets import load_dataset,load_from_disk, Dataset # huggingface datasets
from lp_tokenizer import Tokenizer
import csv
# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 12

dataset_size=65536
vocab_size=32768

dataset = load_from_disk("finewebedu_data")['train'].select(range(65536))

file_path=f"new_vocab/vocab_finewebedu_data_0_{vocab_size}.json"

with open(file_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

tokenizer=Tokenizer(vocab_size=vocab_size,vocab=vocab,unk_token="[UNK]")

def merge_every_n_rows(dataset, n: int):
    """
    Merge every n rows of a Hugging Face Dataset into one row,
    concatenating the 'text' fields into a single string per merged row.
    Returns a Hugging Face Dataset compatible with .map().
    """
    merged_rows = []

    for i in tqdm(range(0, len(dataset), n), total=(len(dataset) + n - 1) // n, desc="Merging texts"):
        # Select the next n rows and convert to list of dicts
        chunk = dataset.select(range(i, min(i+n, len(dataset)))).to_dict()["text"]
        # Concatenate the 'text' values with a separator
        merged_text = "<|endoftext|>".join(chunk)
        # Append as a dict
        merged_rows.append({"text": merged_text})

    return Dataset.from_list(merged_rows)

def process(example):
    ids = tokenizer.encode_matrix(example['text'],vocab) # encode_ordinary ignores any special tokens
    out = {'ids': ids, 'len': len(ids)}
    return out


# tokenize the dataset
tokenized = dataset.map(
    process,
    remove_columns=['text'],
    desc="tokenizing the splits",
    num_proc=num_proc,
)



total_tokens = int(np.sum(tokenized["len"]))

output_file = "dataset_stats.csv"
stats = {
        "total_tokens": total_tokens,
        "vocab_size": vocab_size,
        "dataset_size": dataset_size,
    }


file_exists = os.path.exists(output_file)

with open(output_file, "a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=stats.keys())
    if not file_exists:
        writer.writeheader()
    writer.writerow(stats)

print(f"Stats written/appended to {output_file}: {stats}")