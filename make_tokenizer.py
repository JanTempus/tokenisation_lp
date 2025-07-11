from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
from setup_lp import create_instance
import numpy as np
import os
import pickle

data = np.load("strings_with_frequency.npz")
inputStrings=data["inputStrings" ]
inputStringsfrequencies=data["inputStringsfrequencies"]

# Load .npz and extract correctly
unique_chars_npz = np.load("unique_characters.npz", allow_pickle=True)
unique_chars = list(unique_chars_npz["unique_chars"].item())  # .item() to extract the original Python object (e.g., set or list)

# Now ensure it's a list for concatenation


tokens=create_instance(inputStrings,inputStringsfrequencies,35000)

all_tokens = tokens + unique_chars

# Build vocab dictionary
vocab = {token: idx for idx, token in enumerate(all_tokens)}

# Create tokenizer
tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

# Wrap in a HuggingFace-compatible wrapper
hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)

# Add special tokens
hf_tokenizer.add_special_tokens({
    "unk_token": "[UNK]",
    "pad_token": "[PAD]",
    "cls_token": "[CLS]",
    "sep_token": "[SEP]",
    "mask_token": "[MASK]"
})

# Save the tokenizer (HuggingFace-compatible format)
hf_tokenizer.save_pretrained("lp_tokenizer")
