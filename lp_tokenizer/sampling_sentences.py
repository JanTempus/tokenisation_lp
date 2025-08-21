from lp_tokenizer import Tokenizer
from transformers import AutoTokenizer
from datasets import  load_from_disk,load_dataset, Dataset
import os
import csv
from tqdm import tqdm
import numpy as np
from functools import partial

intristics_path="intrinstics.csv"

datasetname="finewebedu"
dataset_url="pietrolesci/finewebedu-20B"
dataset_path="finewebedu_data"

tokenizer=Tokenizer(saved_dataset_path=dataset_path)

pretokenizer=AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped",
                              revision="step3000",
                              cache_dir="./pythia-70m-deduped/step3000",
                                            )

dataset_size=8192 #1048576

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
unique_chars = tokenizer.get_unique_chars(dataset_raw,true_dataset_size)
unique_chars_size=len(unique_chars)


tokenizer=Tokenizer(saved_dataset_path=dataset_path, vocab_size=8126)
input_strings,  input_strings_frequencies = tokenizer.pretokenize_and_prepare_dataset(dataset_size,dataset_raw,save=False)
tokenizer.check_number_edges(input_strings)

print(f"Number of input strings {len(input_strings)}")
   
