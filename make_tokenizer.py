from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFKC
from collections import OrderedDict
from lp_functions import create_instance
import numpy as np
import os
import pickle
import json

data = np.load("strings_with_frequency.npz")
inputStrings=data["inputStrings" ]
inputStringsfrequencies=data["inputStringsfrequencies"]

# Load .npz and extract correctly
unique_chars_npz = np.load("unique_characters.npz", allow_pickle=True)
unique_chars = list(unique_chars_npz["unique_chars"].item())  # .item() to extract the original Python object (e.g., set or list)

# Now ensure it's a list for concatenation


tokens=create_instance(inputStrings,inputStringsfrequencies,35000)

all_tokens = tokens + unique_chars

output_path="lp_tokenizer.json"
vocab = OrderedDict((token, idx) for idx, token in enumerate(all_tokens))

# Ensure there's an unknown token
if "[UNK]" not in vocab:
    vocab["[UNK]"] = len(vocab)

# Build the tokenizer with WordLevel model
tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))

# Add byte-level behavior
tokenizer.normalizer = NFKC()
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
tokenizer.decoder = ByteLevelDecoder()
tokenizer.post_processor = ByteLevelProcessor(trim_offsets=True)

# Save to single HuggingFace-style JSON
tokenizer.save(output_path)
print(f"Tokenizer with custom merge logic saved to {output_path}")