from datasets import load_dataset
from transformers import AutoTokenizer
from collections import defaultdict
import numpy as np

TinyStories = load_dataset("roneneldan/TinyStories")

tokenizer=AutoTokenizer.from_pretrained(
  "EleutherAI/pythia-70m-deduped",
  revision="step3000",
  cache_dir="./pythia-70m-deduped/step3000",
)

print("Downloaded the data set")
corpus=[]


dataSetSize=len(TinyStories['train'])

for i in range(dataSetSize):
    corpus.append(TinyStories['train'][i]['text'])

word_freqs = defaultdict(int)

print("created the corpus")

i=0
for text in corpus:
    print(i, " out of ", dataSetSize)
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    new_words = [word for word, offset in words_with_offsets]
    for word in new_words:
        word_freqs[word] += 1
    i=i+1

word_freqs={k: v for k, v in word_freqs.items() if len(k) > 1}

inputStrings=list(word_freqs.keys())
inputStringsfrequencies=list(word_freqs.values())

np.savez("strings_with_frequency.npz", inputStrings=np.array(inputStrings), inputStringsfrequencies=np.array(inputStringsfrequencies))