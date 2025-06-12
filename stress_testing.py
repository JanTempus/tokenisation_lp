from datasets import load_dataset
import multi_strings_flow as msf
from collections import defaultdict
import time
import numpy as np

from transformers import AutoTokenizer
TinyStories = load_dataset("roneneldan/TinyStories")
corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]
tokenizer=AutoTokenizer.from_pretrained(
  "EleutherAI/pythia-70m-deduped",
  revision="step3000",
  cache_dir="./pythia-70m-deduped/step3000",
)
# corpus = [
#     "This is the Hugging Face Course.",
#     "This chapter is about tokenization.",
#     "This section shows several tokenizer algorithms.",
#     "Hopefully, you will be able to understand how they are trained and generate tokens.",
# ]




corpus=[]
for i in range(len(TinyStories)):
    corpus.append(TinyStories['train'][i]['text'])

word_freqs = defaultdict(int)

for text in corpus:
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    new_words = [word for word, offset in words_with_offsets]
    for word in new_words:
        word_freqs[word] += 1

word_freqs={k: v for k, v in word_freqs.items() if len(k) > 1}

inputStrings=list(word_freqs.keys())
inputStringsfrequencies=list(word_freqs.values())


msf.CreateInstanceAndSolve(inputStrings,inputStringsfrequencies,5,5)
