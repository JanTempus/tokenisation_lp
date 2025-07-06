from datasets import load_dataset
from transformers import AutoTokenizer
from collections import defaultdict
import numpy as np
from datasets import load_dataset, load_from_disk
import os
import matplotlib.pyplot as plt
import pickle



def count_total_and_single_occurrences(word_freqs, num_count):
    """
    Args:
        word_freqs (dict or defaultdict): A dictionary mapping items (e.g. words) to integer counts.
    
    Returns:
        tuple: (total_number_of_elements, number_with_count_1)
    """

    single_occurrence_count = sum(1 for count in word_freqs.values() if count == num_count)
    return single_occurrence_count

dataset_path = "tinystories_data"

if os.path.exists(dataset_path):
    print("Loading dataset from disk...")
    TinyStories = load_from_disk(dataset_path)
else:
    print("Downloading dataset...")
    TinyStories = load_dataset("roneneldan/TinyStories")
    TinyStories.save_to_disk(dataset_path)


tokenizer=AutoTokenizer.from_pretrained(
  "EleutherAI/pythia-70m-deduped",
  revision="step3000",
  cache_dir="./pythia-70m-deduped/step3000",
)

corpus=[]


dataSetSize=len(TinyStories['train'])

dataSetSize = int(dataSetSize/64)

for i in range(dataSetSize):
    corpus.append(TinyStories['train'][i]['text'])

word_freqs = defaultdict(int)

print("created the corpus")

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

word_freqs={k: v for k, v in word_freqs.items() if len(k) > 1}


inputStrings=list(word_freqs.keys())
inputStringsfrequencies=list(word_freqs.values())

np.savez("strings_with_frequency.npz", inputStrings=np.array(inputStrings), inputStringsfrequencies=np.array(inputStringsfrequencies))