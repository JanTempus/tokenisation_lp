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

file_path=f"vocab_finewebedu_data_{vocab_size}.json"

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
from transformers import PreTrainedTokenizerFast

# Path to the folder where you saved it
tokenizer_path = "tokenizers_lp/lp_1024_finewebedu_data"

# Load the tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

# Example usage
#text = "Hello world, this is a test."
tokens = tokenizer.encode(dataset_0)
decoded = tokenizer.decode(tokens)

print("Original text:", dataset_0)
print("Token IDs:", tokens)
print("Decoded text:", decoded)



# if tokens_a == tokens_b:
#     print("✅ Outputs are identical")
# else:
#     print("❌ Outputs differ!")
#     # show where they differ
#     for i, (a, b) in enumerate(zip(tokens_a, tokens_b)):
#         if a != b:
#             print(f"Mismatch at position {i}: {a} vs {b}")
#             break