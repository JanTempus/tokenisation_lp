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

def bytelevel_vocab_to_tokenizer(vocab, vocab_size:int, raw_dataset_path, save_dir: str):
    """
    Load a vocab.json (byte-level tokens -> IDs) into a Hugging Face compatible tokenizer.

    Args:
        vocab_path (str): Path to the vocab.json file.

    Returns:
        PreTrainedTokenizerFast: Hugging Face compatible tokenizer.
    """

    pretokenizer=AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped",
                              revision="step3000",
                              cache_dir="./pythia-70m-deduped/step3000",
                                            )
    # Create WordLevel model (with unk token)
    tokenizer = tokTokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))

    # Set ByteLevel pre-tokenizer and decoder
    tokenizer.pre_tokenizer = pretokenizer.backend_tokenizer.pre_tokenizer
    tokenizer.decoder = ByteLevelDecoder()

    # Wrap with Hugging Face interface
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        eos_token="<|endoftext|>"
    )

    save_path = os.path.join(save_dir, f"lp_{vocab_size}_{raw_dataset_path}")
    os.makedirs(save_path, exist_ok=True)
    hf_tokenizer.save_pretrained(save_path)
    print(f"Saved Hugging Face–compatible tokenizer at {save_path}")

datasetname="finewebedu"
dataset_url="pietrolesci/finewebedu-20B"
dataset_path="finewebedu_data"


# datasetname="tinystories"
# dataset_url="roneneldan/TinyStories"
# dataset_path="tinystories_data"

tokenizer=Tokenizer(saved_dataset_path=dataset_path)

num_proc = 12
pretokenizer=AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped",
                              revision="step3000",
                              cache_dir="./pythia-70m-deduped/step3000",
                                            )
#vocab_size_max=32000
#dataset_size_max=1500000

#~/token_lp/tokenisation_lp/lp_tokenizer/vocabs/vocab_tinystories_data_0_32768.json
dataset_sizes=[65536] #1048576
vocab_sizes=[1024,2048,8192,16384,32768,65536,131072]

if dataset_url is None and dataset_path is None:
    raise ValueError("Must include either dataset_url or dataset_path")

if os.path.exists(dataset_path):
    dataset_raw=load_from_disk(dataset_path)
else:
    dataset_raw=load_dataset(dataset_url)
    dataset_raw.save_to_disk(dataset_path)

dataset=dataset_raw['train']

save_dir = "tokenizers"
#merged_dataset=merge_into_chunks(dataset,1000)

true_dataset_size=len(dataset)
unique_chars = tokenizer.get_unique_chars(dataset_raw,true_dataset_size)
unique_chars_size=len(unique_chars)
for dataset_size in dataset_sizes:
    for vocab_size in vocab_sizes:
        tokenizer=Tokenizer(saved_dataset_path=dataset_path, vocab_size=vocab_size)
        input_strings,  input_strings_frequencies = tokenizer.pretokenize_and_prepare_dataset(dataset_size,dataset_raw,save=False)

        tokenizer.make_vocab(input_strings=input_strings,
                                input_strings_frequencies=input_strings_frequencies, 
                                unique_chars=unique_chars )
        
        vocab=tokenizer.get_vocab()

        bytelevel_vocab_to_tokenizer(vocab,vocab_size,dataset_path,save_dir)
        
  