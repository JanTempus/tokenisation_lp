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
from transformers import PreTrainedTokenizerFast

datasetname="finewebedu"
dataset_url="pietrolesci/finewebedu-20B"
dataset_path="finewebedu_data"

dataset_size=65536
vocab_size=1024

file_path=f"new_vocab/vocab_finewebedu_data_0_{vocab_size}.json"

with open(file_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)



#lp_tokenizer=Tokenizer(vocab_size=vocab_size,vocab=vocab,unk_token="[UNK]")


pretokenizer=AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped",
                              revision="step3000",
                              cache_dir="./pythia-70m-deduped/step3000",
                                            ).backend_tokenizer.pre_tokenizer
dataset=load_from_disk(dataset_path)['train']

dataset_0=dataset[200]['text']




# Path to the folder where you saved it
tokenizer_path = "tokenizers_lp/lp_1024_finewebedu_data"

# Load the tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

# Example usage
text = "Hello world, this is a test. endoftextbehere And now we see what happens"
tokens = tokenizer.encode(text)
decoded = tokenizer.decode(tokens)

# Check the EOS token ID
eos_id = tokenizer.eos_token_id
print("EOS token ID:", eos_id)

# Convert the ID back to the string token
eos_token = tokenizer.convert_ids_to_tokens(eos_id)
print("EOS token string:", eos_token)

# Or, decode it directly to see it as text
print("EOS token decoded:", tokenizer.decode([eos_id]))

#lp_tokens=lp_tokenizer.encode(dataset_0,vocab=vocab)

def flatten_strings(strings, separator=" "):
    """
    Takes a list of strings and flattens them into one large string.
    
    Args:
        strings (list[str]): List of strings to flatten.
        separator (str): String to insert between elements (default: space).
    
    Returns:
        str: The flattened string.
    """
    return separator.join(strings)


#tokens_a=flatten_strings(lp_tokens)
tokens_b=decoded
# print(flatten_strings(lp_tokens))
# print("--------")
# #print("Original text:", dataset_0)
# #print("Token IDs:", tokens)
# print("Decoded text:", decoded)

#print(tokens_a)
print(tokens_b)

# if tokens_a == tokens_b:
#     print("✅ Outputs are identical")
# else:
#     print("❌ Outputs differ!")
#     # show where they differ
#     for i, (a, b) in enumerate(zip(tokens_a, tokens_b)):
#         if a != b:
#             print(f"Mismatch at position {i}: {a} vs {b}")
#             break