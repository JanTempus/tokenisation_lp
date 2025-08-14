from lp_tokenizer import Tokenizer

from datasets import  load_from_disk,load_dataset
import pickle
import os
import csv


# datasetname="finewebedu"
# dataset_url="pietrolesci/finewebedu-20B"
# dataset_path="finewebedu_data"


datasetname="tinystories"
dataset_url="roneneldan/TinyStories"
dataset_path="tinystories_data"

tokenizer=Tokenizer()



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

true_dataset_size=len(dataset)
unique_chars = tokenizer.get_unique_chars(dataset,true_dataset_size)
unique_chars_size=len(unique_chars)





def write_two_numbers(csv_path: str, num1: float, num2: float, num3:int):
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



while dataset_size<dataset_size_max:
    
    vocab_size_dif=20
    vocab_size=unique_chars_size+vocab_size_dif
    while vocab_size<vocab_size_max:
        tokenizer=Tokenizer(vocab_size=vocab_size)
        input_strings,  input_strings_frequencies = tokenizer.pretokenize_and_prepare_dataset(dataset_size,dataset,save=False)

        tokenizer.make_vocab(input_strings=input_strings,
                             input_strings_frequencies=input_strings_frequencies, 
                             unique_chars=unique_chars )
        
        vocab=tokenizer.get_vocab()

        vocab_size_dif=vocab_size_dif*2

    dataset_size=dataset_size*2
    
    