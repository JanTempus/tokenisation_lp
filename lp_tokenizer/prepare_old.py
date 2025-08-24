# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
from tqdm import tqdm
import numpy as np
import json
from datasets import load_dataset,load_from_disk, Dataset # huggingface datasets
from lp_tokenizer import Tokenizer
# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 8

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc



file_path="vocab_finewebedu_data_32768.json"
with open(file_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

tokenizer=Tokenizer(vocab_size=32768,vocab=vocab,unk_token="[UNK]")

if __name__ == '__main__':
   
    dataset = load_from_disk("finewebedu_data")['train']
    def merge_into_chunks(dataset, t: int):
        merged_texts = []
        for i in tqdm(range(0, len(dataset), t), desc="Making into larger chunks"):
            chunk = [x['text'] for x in dataset[i:i+t]]  # extract text
            merged_text = " ".join(chunk)
            merged_texts.append(merged_text)
        return Dataset.from_dict({'text': merged_texts})

    dataset_merged=merge_into_chunks(dataset,2000)

    split_dataset = dataset_merged.train_test_split(test_size=0.005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')
    

    
    def process(example):
        ids = tokenizer.encode(example['text'],vocab) # encode_ordinary ignores any special tokens
        ids.append(1) # add the end of text token, 3199 for 
        # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
        out = {'ids': ids, 'len': len(ids)}
        return out

    # tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        
        idx = 0
        for example in tqdm(dset, desc=f'writing {filename}'):
            arr_batch = np.array(example['ids'], dtype=dtype)
            arr[idx:idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()