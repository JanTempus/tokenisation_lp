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

# Load .npz and extract correctly
unique_chars_npz = np.load("unique_characters.npz", allow_pickle=True)
unique_chars = unique_chars_npz["unique_chars"].item()  # .item() to extract the original Python object (e.g., set or list)

# Now ensure it's a list for concatenation

print(type(unique_chars))

# tokens=create_instance(inputStrings,inputStringsfrequencies,35000)

# all_tokens = tokens + list(unique_chars)

# # Build vocab dictionary
# vocab = {token: idx for idx, token in enumerate(all_tokens)}

# # Create tokenizer
# tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
# tokenizer.pre_tokenizer = Whitespace()

# # Save to disk
# tokenizer.save("my_tokenizer.json")

