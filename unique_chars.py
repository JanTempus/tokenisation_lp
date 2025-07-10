from transformers import AutoTokenizer
from collections import defaultdict
import numpy as np
from datasets import load_dataset, load_from_disk
import os
import matplotlib.pyplot as plt
import pickle

dataset_path = "tinystories_data"

tokenizer=AutoTokenizer.from_pretrained(
  "EleutherAI/pythia-70m-deduped",
  revision="step3000",
  cache_dir="./pythia-70m-deduped/step3000",
)

corpus=[]


dataSetSize = 2119719

file_name= "word_freqs_"+str(dataSetSize)+".pkl"


if os.path.exists(file_name):
    print("Loading word frequencies from pickle...")
    with open(file_name, "rb") as f:
        word_freqs = pickle.load(f)

else:
    print("Started working on pre tokenization")
    word_freqs = defaultdict(int)
    
    for i, text in enumerate(corpus):
        words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        new_words = [word for word, offset in words_with_offsets]
        for word in new_words:
            word_freqs[word] += 1
    print("Finished working on pre tokenization")
    with open(file_name, "wb") as f:
        pickle.dump(word_freqs, f)
    print("Saved word frequencies to disk.")

unique_chars = set()
for word in word_freqs:
    unique_chars.update(word)

# Convert to NumPy array (dtype=object since it's strings)
unique_chars_array = np.array(unique_chars, dtype=object)

# Save to .npz file
np.savez_compressed("unique_characters.npz", unique_chars=unique_chars_array)

print(f"Saved {len(unique_chars)} unique characters to 'unique_characters.npz'")
