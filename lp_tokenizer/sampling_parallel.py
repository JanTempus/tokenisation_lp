from transformers import AutoTokenizer
from datasets import  load_from_disk,load_dataset, Dataset
from collections import OrderedDict,defaultdict
from tqdm import tqdm
import numpy as np
import itertools

def collect_pretokenized_words(dataset, pretokenizer, t: int, num_proc: int):
    """
    Pretokenize the first t examples of a Hugging Face dataset in parallel
    and return all words as a single list.
    """

    # restrict dataset to the first t examples
    dataset = dataset['train'].select(range(t))

    # define a per-example processing function
    def process(example):
        words_with_offsets = pretokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(example['text'])
        new_words = [word for word, offset in words_with_offsets]
        return {"words": new_words}

    # run map in parallel
    processed = dataset.map(
        process,
        remove_columns=dataset.column_names,
        desc="Pretokenizing",
        num_proc=num_proc,
    )

    all_words = list(itertools.chain.from_iterable(
        tqdm(processed["words"], desc="Flattening words", total=len(processed))
        ))

    return all_words

def sample_words_large(all_words: list[str], m: int, seed: int | None = None) -> list[str]:
    """
    Efficiently sample `m` words from `all_words` without replacement.
    Works well for very large lists by operating on indices only.
    """
    if m > len(all_words):
        raise ValueError("Cannot sample more elements than available without replacement")
    
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(all_words), size=m, replace=False)
    return [all_words[i] for i in indices]




datasetname="finewebedu"
dataset_url="pietrolesci/finewebedu-20B"
dataset_path="finewebedu_data"

pretokenizer=AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped",
                              revision="step3000",
                              cache_dir="./pythia-70m-deduped/step3000",
                                            )

dataset_size=10000000

dataset=load_from_disk(dataset_path)

all_words = collect_pretokenized_words(dataset, pretokenizer, dataset_size, num_proc=4)

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