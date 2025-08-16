from lp_tokenizer import Tokenizer
from transformers import AutoTokenizer
from datasets import  load_from_disk,load_dataset, Dataset
import os
import csv
from tqdm import tqdm
import numpy as np
from functools import partial

def save_data(csv_path: str, num1: float, num2: float, num3:int):
    """
    Opens (or creates) a CSV file and appends a row with two numbers.

    Args:
        csv_path (str): Path to the CSV file.
        num1 (float): First number.
        num2 (float): Second number.
    """
    # Check if file exists
    file_exists = os.path.isfile(csv_path)

    # Open in append mode
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)

        # If file didn't exist, write a header first (optional)
        if not file_exists:
            writer.writerow(["Data Set size", "Vocab Size", "Compression"])

        # Write the numbers as a new row
        writer.writerow([num1, num2, num3])


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

# datasetname="finewebedu"
# dataset_url="pietrolesci/finewebedu-20B"
# dataset_path="finewebedu_data"


datasetname="tinystories"
dataset_url="roneneldan/TinyStories"
dataset_path="tinystories_data"

tokenizer=Tokenizer(saved_dataset_path=dataset_path)

num_proc = 12
pretokenizer=AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped",
                              revision="step3000",
                              cache_dir="./pythia-70m-deduped/step3000",
                                            )
vocab_size_max=32000
dataset_size_max=1500000
dataset_size=1

if dataset_url is None and dataset_path is None:
    raise ValueError("Must include either dataset_url or dataset_path")

if os.path.exists(dataset_path):
    dataset_raw=load_from_disk(dataset_path)
else:
    dataset_raw=load_dataset(dataset_url)
    dataset_raw.save_to_disk(dataset_path)

dataset=dataset_raw['train']


merged_dataset=merge_into_chunks(dataset,1000)

true_dataset_size=len(dataset)
unique_chars = tokenizer.get_unique_chars(dataset_raw,true_dataset_size)
unique_chars_size=len(unique_chars)


# corpus=[]

# for i in tqdm(range(dataset_size),desc="Appending text to the corpus"):
#     corpus.append(dataset['train'][i]['text'])

while dataset_size<dataset_size_max:
    
    vocab_size_dif=20
    vocab_size=unique_chars_size+vocab_size_dif
    while vocab_size<vocab_size_max:
        print(f"Curr vocab size {vocab_size}, Curr dataset size {dataset_size}")

        tokenizer=Tokenizer(saved_dataset_path=dataset_path, vocab_size=vocab_size)
        input_strings,  input_strings_frequencies = tokenizer.pretokenize_and_prepare_dataset(dataset_size,dataset_raw,save=False)

        tokenizer.make_vocab(input_strings=input_strings,
                             input_strings_frequencies=input_strings_frequencies, 
                             unique_chars=unique_chars )
        
        #vocab=tokenizer.get_vocab()
        # process_fn = partial(process, vocab=vocab, tokenizer=tokenizer)

        # # tokenize the merged_dataset
            
        # tokenized = merged_dataset.map(
        #     process_fn,
        #     remove_columns=['text'],
        #     desc="tokenizing the splits",
        #     num_proc=num_proc
        # )



        # # Sum all lengths to get total number of token IDs
        # compression = np.sum(tokenized['len'], dtype=np.uint64)

        # print(f"dataset_size {dataset_size } vocab size {vocab_size} compression {compression}  ")
        # save_data(intristics_path,dataset_size,vocab_size,compression)
        vocab_size=vocab_size*2

    dataset_size=dataset_size*2
    
    