from lp_tokenizer import Tokenizer
from transformers import AutoTokenizer
from datasets import  load_from_disk,load_dataset, Dataset
import os
import csv
from tqdm import tqdm
import numpy as np
from functools import partial
import pynvml

datasetname="finewebedu"
dataset_url="pietrolesci/finewebedu-20B"
dataset_path="finewebedu_data"

num_proc=8

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU 0



tokenizer=Tokenizer(saved_dataset_path=dataset_path)


pretokenizer=AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped",
                              revision="step3000",
                              cache_dir="./pythia-70m-deduped/step3000",
                                            )

dataset_size=65536 #1048576
vocab_sizes=[1024,2048,4096,8192,16384,32768,65536,131072]

if dataset_url is None and dataset_path is None:
    raise ValueError("Must include either dataset_url or dataset_path")

if os.path.exists(dataset_path):
    dataset_raw=load_from_disk(dataset_path)
else:
    dataset_raw=load_dataset(dataset_url)
    dataset_raw.save_to_disk(dataset_path)

dataset=dataset_raw['train']


true_dataset_size=len(dataset)
unique_chars = tokenizer.get_unique_chars_parallel(dataset_raw,true_dataset_size,pretokenizer,num_proc=num_proc)
unique_chars_size=len(unique_chars)

output_file="computation_time.csv"
file_exists = os.path.exists(output_file)

with open(output_file, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([f"Dataset size {dataset_size}"])


pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU 0

for vocab_size in vocab_sizes:
    tokenizer = Tokenizer(saved_dataset_path=dataset_path, vocab_size=vocab_size)
    input_strings, input_strings_frequencies = tokenizer.pretokenize_and_prepare_dataset(dataset_size, dataset_raw, save=False)

    tokenizer.make_vocab(
        input_strings=input_strings,
        input_strings_frequencies=input_strings_frequencies,
        unique_chars=unique_chars,
        save_vocab=False
    )

    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)

    print(f"Vocab size: {vocab_size}, GPU memory used: {mem_info.used / 1024**2:.2f} MB, GPU utilization: {util.gpu}%")

    with open(output_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([vocab_size, mem_info.used, util.gpu])

pynvml.nvmlShutdown()
