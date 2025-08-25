from lp_tokenizer import Tokenizer
from transformers import AutoTokenizer
from datasets import  load_from_disk,load_dataset, Dataset
import os
import csv
from tqdm import tqdm
import numpy as np
from functools import partial


def merge_into_chunks(dataset, t: int,):


    merged_texts = []
    # Go through dataset in steps of t
    for i in range(0, len(dataset), t):
        chunk = dataset[i : i + t]  # list of texts
        merged_text = " ".join(chunk)
        merged_texts.append(merged_text)

    # Create new dataset
    dataset_merged = Dataset.from_dict({'text': merged_texts})
    return dataset_merged


def process(example, vocab, tokenizer):
    ids = tokenizer.encode(example['text'], vocab)
    return {'ids': ids, 'len': len(ids)}

intristics_path="intrinstics.csv"

datasetname="finewebedu"
dataset_url="pietrolesci/finewebedu-20B"
dataset_path="finewebedu_data"

num_proc=12

# datasetname="tinystories"
# dataset_url="roneneldan/TinyStories"
# dataset_path="tinystories_data"

tokenizer=Tokenizer(saved_dataset_path=dataset_path)


pretokenizer=AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped",
                              revision="step3000",
                              cache_dir="./pythia-70m-deduped/step3000",
                                            )

dataset_size=16384 #1048576
vocab_sizes=[4096,8192,16384,32768]

if dataset_url is None and dataset_path is None:
    raise ValueError("Must include either dataset_url or dataset_path")

if os.path.exists(dataset_path):
    dataset_raw=load_from_disk(dataset_path)
else:
    dataset_raw=load_dataset(dataset_url)
    dataset_raw.save_to_disk(dataset_path)

dataset=dataset_raw['train']


#merged_dataset=merge_into_chunks(dataset,1000)

true_dataset_size=len(dataset)
unique_chars = tokenizer.get_unique_chars_parallel(dataset_raw,true_dataset_size,pretokenizer,num_proc=num_proc)
unique_chars_size=len(unique_chars)

for vocab_size in vocab_sizes:
    tokenizer=Tokenizer(saved_dataset_path=dataset_path, vocab_size=vocab_size)
    input_strings,  input_strings_frequencies = tokenizer.pretokenize_and_prepare_dataset(dataset_size,dataset_raw,save=False)

    tokenizer.make_vocab(input_strings=input_strings,
                            input_strings_frequencies=input_strings_frequencies, 
                            unique_chars=unique_chars )
    
   


    