import os
from tqdm import tqdm
import numpy as np
import json
from datasets import load_dataset,load_from_disk, Dataset # huggingface datasets
from lp_tokenizer import Tokenizer
import csv
# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 8

dataset_size=65536
vocab_size=1024


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

file_path=f"vocabs/vocab_finewebedu_data_0_{vocab_size}.json"

with open(file_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

tokenizer=Tokenizer(vocab_size=vocab_size,vocab=vocab,unk_token="[UNK]")

dataset = load_from_disk("finewebedu_data")['train'].select(range(65536))

def merge_into_chunks(dataset, t: int,):
        merged_texts = []
        # Go through dataset in steps of t
        for i in tqdm(range(0, len(dataset), t),desc="Making into larger chunks"):
            chunk = dataset[i : i + t]  # list of texts
            merged_text = " ".join(chunk)
            merged_texts.append(merged_text)

        # Create new dataset
        dataset_merged = Dataset.from_dict({'text': merged_texts})
        return dataset_merged

def process(example):
    ids = tokenizer.encode(example['text'],vocab) # encode_ordinary ignores any special tokens
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