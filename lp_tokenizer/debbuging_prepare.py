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
    from datasets import Dataset

    def slice_dataset_by_chunks(dataset, start_chunk: int, end_chunk: int, chunk_size: int):
        """
        Return a Dataset slice corresponding to merged chunk indices.

        Args:
            dataset: Original Hugging Face Dataset
            start_chunk: starting chunk index (inclusive)
            end_chunk: ending chunk index (exclusive)
            chunk_size: number of examples per chunk (t)

        Returns:
            Dataset slice containing all examples in the selected chunks.
        """
        start_idx = start_chunk * chunk_size
        end_idx = min(end_chunk * chunk_size, len(dataset))
        subset = dataset.select(range(start_idx, end_idx))
        return subset

    # Example usage: isolate chunk containing index 7992
    chunk_size = 2000
    problem_index = 7992
    start_chunk = problem_index // chunk_size
    end_chunk = start_chunk + 1  # just that chunk

    subset_dataset = slice_dataset_by_chunks(dataset, start_chunk, end_chunk, chunk_size)
    print(f"Selected dataset indices: {start_chunk*chunk_size} to {min(end_chunk*chunk_size, len(dataset))}")


    corpus=[]
    
    for i in tqdm(range(len(subset_dataset)),desc="Appending text to the corpus"):
                corpus.append(dataset['train'][i]['text'])
    tokenizer.encode(corpus,vocab)

    # dataset_merged_into_chunks=merge_into_chunks(dataset,2000)



    
    # def process(example):
    #     ids = tokenizer.encode(example['text'], vocab)

    #     # normalize into a list
    #     if isinstance(ids, (int, np.integer)):
    #         ids = [int(ids)]
    #     elif isinstance(ids, (list, np.ndarray)):
    #         ids = list(ids)
    #     else:
    #         # catch weird cases (None, str, etc.)
    #         print("⚠️ Unexpected type:", type(ids), "value:", ids, "text:", example['text'][:100])
    #         ids = [] if ids is None else [ids]

    #     ids.append(1)  # EOT token

    #     return {
    #         "ids": ids,
    #         "len": len(ids),
    #     }

    # # tokenize the dataset
    # tokenized = dataset_merged.map(
    #     process,
    #     remove_columns=['text'],
    #     desc="tokenizing the splits",
    #     num_proc=num_proc,
    # )

    # # concatenate all the ids in each dataset into one large file we can use for training
    # for split, dset in tokenized.items():
    #     arr_len = np.sum(dset['len'], dtype=np.uint64)
    #     filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
    #     dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
    #     arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
    #     total_batches = min(1024, len(dset))

    #     idx = 0
    #     for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
    #         # Batch together samples for faster write
    #         batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
    #         arr_batch = np.concatenate(batch['ids'])
    #         # Write into mmap
    #         arr[idx : idx + len(arr_batch)] = arr_batch
    #         idx += len(arr_batch)
    #     arr.flush()