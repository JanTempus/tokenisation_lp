from lp_tokenizer import Tokenizer
from datasets import  load_from_disk,load_dataset, Dataset
import os
import csv
from tqdm import tqdm
import numpy as np
from functools import partial
from tokenizers import Tokenizer as tokTokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from transformers import PreTrainedTokenizerFast, AutoTokenizer
import json
from pathlib import Path
from datasets import  load_from_disk
import pickle

datasetname="finewebedu"
dataset_url="pietrolesci/finewebedu-20B"
dataset_path="finewebedu_data"

dataset_size=65536
vocab_size=32768

file_path=f"new_vocab/vocab_finewebedu_data_0_{vocab_size}.json"

with open(file_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

tokenizer=Tokenizer(vocab_size=vocab_size,vocab=vocab,unk_token="[UNK]")


pretokenizer=AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped",
                              revision="step3000",
                              cache_dir="./pythia-70m-deduped/step3000",
                                            )
dataset=load_from_disk(dataset_path)['train']

dataset_0=dataset[0]
dataset_1=dataset[1]
chunk=dataset.select(range(2))
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


merged=merge_every_n_rows(chunk,2)

tokens_b=tokenizer.encode(dataset_0['text'],vocab)

print(tokens_b)
