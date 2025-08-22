from lp_tokenizer import Tokenizer
from transformers import AutoTokenizer
from datasets import  load_from_disk,load_dataset, Dataset
import os
import csv
from tqdm import tqdm
import numpy as np
from functools import partial

def collect_pretokenized_words(corpus, pretokenizer):
    all_words = []

    for i, text in tqdm(enumerate(corpus), total=len(corpus), desc="Pretokenizing"):
        words_with_offsets = pretokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        new_words = [word for word, offset in words_with_offsets]
        all_words.extend(new_words)  # collect all words

    print(len(all_words))


datasetname="finewebedu"
dataset_url="pietrolesci/finewebedu-20B"
dataset_path="finewebedu_data"

pretokenizer=AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped",
                              revision="step3000",
                              cache_dir="./pythia-70m-deduped/step3000",
                                            )

dataset_size=8192 #1048576



if dataset_url is None and dataset_path is None:
    raise ValueError("Must include either dataset_url or dataset_path")


dataset=load_from_disk(dataset_path)

corpus=[]

for i in tqdm(range(dataset_size),desc="Appending text to the corpus"):
    corpus.append(dataset['train'][i]['text'])


collect_pretokenized_words(corpus, pretokenizer)


   
