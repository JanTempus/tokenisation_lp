from lp_tokenizer import Tokenizer
from transformers import AutoTokenizer
from datasets import  load_from_disk,load_dataset, Dataset
import os
import csv
from tqdm import tqdm

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

datasetname="finewebedu"
dataset_url="pietrolesci/finewebedu-20B"
dataset_path="finewebedu_data"

pretokenizer=AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped",
                              revision="step3000",
                              cache_dir="./pythia-70m-deduped/step3000",
                                            )

dataset_size=65536 #1048576

dataset=load_from_disk(dataset_path)

all_words = collect_pretokenized_words(dataset, pretokenizer, dataset_size, num_proc=4)
print(len(all_words), "words collected")
print(all_words[:50])

   
