from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
from setup_lp import create_instance
import numpy as np
import os
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
import pickle
from collections import defaultdict


data = np.load("strings_with_frequency.npz")
inputStrings=data["inputStrings" ]
inputStringsfrequencies=data["inputStringsfrequencies"]

unique_chars = np.load("unique_characters.npz")["unique_chars"]

tokens=create_instance(inputStrings,inputStringsfrequencies,35000)

all_tokens=tokens+unique_chars

# Build vocab dictionary
vocab = {token: idx for idx, token in enumerate(all_tokens)}

# Create tokenizer
tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

# Save to disk
tokenizer.save("my_tokenizer.json")

