from datasets import load_dataset
from transformers import AutoTokenizer
from collections import defaultdict
import numpy as np
from datasets import load_dataset, load_from_disk
import os
import matplotlib.pyplot as plt
import pickle

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

dataSetSize = int(dataSetSize)

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
    word_freqs = defaultdict(int)
    print("Created the corpus")

    for i, text in enumerate(corpus):
        print(i, " out of ", dataSetSize)
        words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        new_words = [word for word, offset in words_with_offsets]
        for word in new_words:
            word_freqs[word] += 1

    with open(file_name, "wb") as f:
        pickle.dump(word_freqs, f)
    print("Saved word frequencies to disk.")

word_freqs={k: v for k, v in word_freqs.items() if len(k) > 1}

print("removed all items which have only a length of 1")
frequencies = list(word_freqs.values())

print("we have sorted things now")
# Step 2: Define bin size and bins
bin_size = 5
max_freq = max(frequencies)
bins = np.arange(0, max_freq + bin_size, bin_size)  # e.g., [0, 5, 10, 15, ...]

# Step 3: Count how many terms fall into each bin
hist, bin_edges = np.histogram(frequencies, bins=bins)

# Step 4: Plot
plt.figure(figsize=(10, 6))
plt.bar(
    [f"{int(start)}–{int(end)}" for start, end in zip(bin_edges[:-1], bin_edges[1:])],
    hist,
    color="mediumseagreen",
    edgecolor="black"
)
plt.xlabel("Frequency Range (# of Occurrences)")
plt.ylabel("Number of Unique Terms")
plt.title("Distribution of Token Frequencies")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("token_frequency_distribution.png", dpi=300, bbox_inches='tight')

# Optional: close the plot to free memory if in a loop or notebook
plt.close()

#sorted_word_freq=sorted(word, key=lambda t: t.lpValue, reverse=True)


# inputStrings=list(word_freqs.keys())
# inputStringsfrequencies=list(word_freqs.values())

# np.savez("strings_with_frequency.npz", inputStrings=np.array(inputStrings), inputStringsfrequencies=np.array(inputStringsfrequencies))