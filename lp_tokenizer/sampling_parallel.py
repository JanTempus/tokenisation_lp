from transformers import AutoTokenizer
from datasets import  load_from_disk,load_dataset, Dataset
from collections import OrderedDict,defaultdict
from tqdm import tqdm
import numpy as np
import itertools


def sample_hf_dataset(dataset: Dataset, t: int, seed: int | None = None) -> Dataset:
    """
    Sample exactly t documents without replacement from a Hugging Face dataset.

    Args:
        dataset: Hugging Face Dataset to sample from.
        t: Number of documents to sample (0 <= t <= len(dataset)).
        seed: Optional random seed for reproducibility.

    Returns:
        Hugging Face Dataset with t rows.
    """
    m = len(dataset)
    if t > m:
        raise ValueError(f"Cannot sample {t} documents from dataset of size {m}")

    rng = np.random.default_rng(seed)
    indices = rng.choice(m, size=t, replace=False)
    return dataset.select(indices.tolist())



datasetname="finewebedu"
dataset_url="pietrolesci/finewebedu-20B"
dataset_path="finewebedu_data"

pretokenizer=AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped",
                              revision="step3000",
                              cache_dir="./pythia-70m-deduped/step3000",
                                            )

dataset_size=10000000

dataset=load_from_disk(dataset_path)

all_words = collect_pretokenized_words(dataset, pretokenizer, dataset_size, num_proc=8)

num_words=8025163 #number of words to sample
   

seeds=[101,212,323,434,545]  

for seed in seeds:
    word_freqs = defaultdict(int)
    sampled_words=sample_words_large(all_words, num_words, seed)
    
    desc=f"Counting frequencies for seed {seed} "
    for word in tqdm(sampled_words, desc=desc):
        word_freqs[word] += 1

    input_strings=list(word_freqs.keys())
    input_strings_frequencies=list(word_freqs.values())
    base_name = f"words_{datasetname}_{num_words}_{seed}"
    strings_file = base_name + "_strings.npy"
    freqs_file = base_name + "_freqs.npy"

    np.save(strings_file, np.array(input_strings, dtype=object),allow_pickle=True)
    np.save(freqs_file, np.array(input_strings_frequencies, dtype=np.int64))