from lp_tokenizer import Tokenizer
from transformers import AutoTokenizer
from datasets import  load_from_disk,load_dataset, Dataset
import os
import csv
from tqdm import tqdm
import numpy as np

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


def merge_into_chunks(dataset, t: int, text_column: str = "text"):
    """
    Load dataset from disk, merge every t examples into one example, 
    and return a new dataset with multiple merged examples.

    Args:
        dataset_path (str): Path to dataset saved with `save_to_disk`.
        t (int): Number of examples per chunk.
        text_column (str): Column containing text data (default: "text").

    Returns:
        Dataset: A new dataset where each row is a merged chunk.
    """

    merged_texts = []
    # Go through dataset in steps of t
    for i in range(0, len(dataset), t):
        chunk = dataset[i : i + t][text_column]  # list of texts
        merged_text = " ".join(chunk)
        merged_texts.append(merged_text)

    # Create new dataset
    dataset_merged = Dataset.from_dict({text_column: merged_texts})
    return dataset_merged

intristics_path="intrinstics.csv"

# datasetname="finewebedu"
# dataset_url="pietrolesci/finewebedu-20B"
# dataset_path="finewebedu_data"


datasetname="tinystories"
dataset_url="roneneldan/TinyStories"
dataset_path="tinystories_data"

tokenizer=Tokenizer()

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
    dataset=load_from_disk(dataset_path)
else:
    dataset=load_dataset(dataset_url)
    dataset.save_to_disk(dataset_path)


merged_dataset=merge_into_chunks(dataset,1000,"text")

true_dataset_size=len(merged_dataset)
unique_chars = tokenizer.get_unique_chars(merged_dataset,true_dataset_size)
unique_chars_size=len(unique_chars)





while dataset_size<dataset_size_max:
    
    vocab_size_dif=20
    vocab_size=unique_chars_size+vocab_size_dif
    while vocab_size<vocab_size_max:
        tokenizer=Tokenizer(saved_dataset_path=dataset_path, vocab_size=vocab_size)
        input_strings,  input_strings_frequencies = tokenizer.pretokenize_and_prepare_dataset(dataset_size,merged_dataset,save=False)

        tokenizer.make_vocab(input_strings=input_strings,
                             input_strings_frequencies=input_strings_frequencies, 
                             unique_chars=unique_chars )
        
        vocab=tokenizer.get_vocab()

        def process(example):
            tokenizer=Tokenizer(saved_dataset_path=dataset_path, vocab_size=vocab_size,unk_token="[UNK]",pretokenizer=pretokenizer)
            ids = tokenizer.encode(example['text'],vocab) 
            out = {'ids': ids, 'len': len(ids)}
            return out

        # tokenize the merged_dataset
        tokenized = merged_dataset.map(
            process,
            remove_columns=['text'],
            desc="tokenizing the splits",
            num_proc=num_proc,
            batched=True
        )

        # concatenate all the ids in each merged_dataset into one large file we can use for training
        for split, dset in tokenized.items():
            arr_len = np.sum(dset['len'], dtype=np.uint64)
            filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
            dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
            arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
            total_batches = min(1024, len(dset))

            total_ids = 0
            for split, dset in tqdm(tokenized.items(), desc="Summing token IDs"):
                total_ids += np.sum(dset['len'], dtype=np.uint64)
            arr.flush()

        # Sum all lengths to get total number of token IDs
        compression = np.sum(tokenized['len'], dtype=np.uint64)

        print(f"dataset_size {dataset_size } vocab size {vocab_size} compression {compression}  ")
        save_data(intristics_path,dataset_size,vocab_size,compression)
        vocab_size_dif=vocab_size_dif*2

    dataset_size=dataset_size*2
    
    