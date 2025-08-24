import os
from tqdm import tqdm
import numpy as np
import json
from datasets import load_from_disk, Dataset
from lp_tokenizer import Tokenizer

file_path = "vocab_finewebedu_data_32768.json"
with open(file_path, 'r', encoding='utf-8') as f:
    vocab = json.load(f)

tokenizer = Tokenizer(vocab_size=32768, vocab=vocab, unk_token="[UNK]")


def merge_into_chunks(dataset, t: int):
    merged_texts = []
    for i in tqdm(range(0, len(dataset), t), desc="Making into larger chunks"):
        chunk = dataset[i: i + t]["text"]  # fetch slice of dataset
        merged_text = " ".join(chunk)
        merged_texts.append(merged_text)
    dataset_merged = Dataset.from_dict({'text': merged_texts})
    return dataset_merged


if __name__ == "__main__":
    dataset = load_from_disk("finewebedu_data")['train']

    # midpoint = len(dataset) // 2 + 5000000
    # second_half = dataset.select(range(midpoint, len(dataset)))

    # dataset_merged = merge_into_chunks(second_half, 2000)

    # sequential tokenization without map
    tokenized_dict = {"ids": [], "len": []}
    dataset_small=dataset.select(range(1))
    for example in tqdm(dataset_small, desc="tokenizing sequentially"):
        ids = tokenizer.encode(example['text'], vocab)
        ids.append(1)  # add eot
        out={"ids": ids, "len": len(ids)}
        tokenized_dict["ids"].append(out["ids"])
        tokenized_dict["len"].append(out["len"])
    tokenized = Dataset.from_dict(tokenized_dict)

    # write to binary
    arr_len = np.sum(tokenized["len"], dtype=np.uint64)
    filename = os.path.join(os.path.dirname(__file__), "train.bin")
    dtype = np.uint16
    arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))

    idx = 0
    for batch_idx in tqdm(range(len(tokenized)), desc=f"writing {filename}"):
        arr_batch = np.array(tokenized["ids"][batch_idx], dtype=dtype)
        arr[idx: idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)
    arr.flush()
