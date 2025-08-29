from lp_tokenizer import Tokenizer
from transformers import AutoTokenizer
from datasets import  load_from_disk,load_dataset, Dataset
import os
import csv
from tqdm import tqdm
import numpy as np
from functools import partial
import itertools

datasetname="finewebedu"
dataset_url="pietrolesci/finewebedu-20B"
dataset_path="finewebedu_data"

num_proc=8
vocab_size=8

tokenizer=Tokenizer(saved_dataset_path=dataset_path)


pretokenizer=AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped",
                              revision="step3000",
                              cache_dir="./pythia-70m-deduped/step3000",
                                            )

dataset_size=65536 #1048576
vocab_size=32768



dataset_raw=load_from_disk(dataset_path)



unique_chars = np.load("unique_charsfinewebedu_data20200000.npy", allow_pickle=True)
unique_chars = unique_chars.tolist()  # convert to Python list if needed

seeds=[977,837,081,430,946,234,100]

token_sets=[]
token_set_length=[]

for seed in seeds:
    dataset=dataset_raw['train'].shuffle(seed=seed).select(range(vocab_size))

    tokenizer = Tokenizer(saved_dataset_path=dataset_path, vocab_size=vocab_size)
    input_strings, input_strings_frequencies = tokenizer.pretokenize_direct(dataset_size, dataset_raw, save=False)

    raw_tokens=tokenizer.generate_vocab_nonzero(
        input_strings=input_strings,
        input_strings_frequencies=input_strings_frequencies,
        unique_chars=unique_chars
    )
    token_set_length.append(len(raw_tokens))
    token_sets.append(raw_tokens)



n = len(token_sets)
dist_matrix = np.zeros((n, n))


def jaccard_distance(a, b):
    inter = len(a & b)
    union = len(a | b)
    return 1 - inter / union if union > 0 else 0.0

for i in range(n):
    for j in range(i+1, n):
        d = jaccard_distance(token_sets[i], token_sets[j])
        dist_matrix[i, j] = dist_matrix[j, i] = d

print(dist_matrix)
print(token_set_length)
