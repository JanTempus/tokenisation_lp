from datasets import load_dataset
import multi_strings_flow as msf
from collections import defaultdict
import time
import numpy as np
import argparse

from transformers import AutoTokenizer
TinyStories = load_dataset("roneneldan/TinyStories")

tokenizer=AutoTokenizer.from_pretrained(
  "EleutherAI/pythia-70m-deduped",
  revision="step3000",
  cache_dir="./pythia-70m-deduped/step3000",
)

print("Downloaded the data set")
corpus=[]


for i in range(len(TinyStories['train'])):
    corpus.append(TinyStories['train'][i]['text'])

word_freqs = defaultdict(int)

print("created the corpus")
for text in corpus:
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    new_words = [word for word, offset in words_with_offsets]
    for word in new_words:
        word_freqs[word] += 1

word_freqs={k: v for k, v in word_freqs.items() if len(k) > 1}

inputStrings=list(word_freqs.keys())
inputStringsfrequencies=list(word_freqs.values())

print("Prepared the input strings")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi_strings_flow with variable arguments.")
    parser.add_argument("arg1", type=int, help="First integer argument for CreateInstanceAndSolve")
    parser.add_argument("arg2", type=int, help="Second integer argument for CreateInstanceAndSolve")
    args = parser.parse_args()

msf.CreateInstanceAndSolve(inputStrings,inputStringsfrequencies,args.arg1,args.arg2)