import os
import json
import shutil
import tempfile
from tqdm import tqdm
from datasets import load_from_disk
from lp_tokenizer import Tokenizer
from multiprocessing import Pool, cpu_count

# Config
dataset_path = "finewebedu_data"
vocab_size = 32768
num_proc = 16      # processes inside Dataset.map()
batch_size = 100
shard_size = 100000 # examples per shard
out_dir = "tokenized_shards"
num_workers = 6

# Load dataset
dataset = load_from_disk(dataset_path)['train']

# Load vocab
file_path = f"new_vocab/vocab_finewebedu_data_0_{vocab_size}.json"
with open(file_path, 'r', encoding='utf-8') as f:
    vocab = json.load(f)

tokenizer = Tokenizer(vocab_size=vocab_size, vocab=vocab, unk_token="[UNK]")

# --- Processing function ---
def process(batch):
    # Append <|endoftext|> to every string
    texts = [t + "endoftextbehere" for t in batch["text"]]
    
    # Merge
    merged_text = "".join(texts)
    
    # Tokenize
    merged_ids = tokenizer.encode(merged_text, vocab)

    
    # Split on token 1
    split_ids = []
    current = []
    for token in merged_ids:
        if token == 1:
            split_ids.append(current)
            current = []
        else:
            current.append(token)
    
    # Return
    return {
        "ids": split_ids,
        "len": [len(x) for x in split_ids],
    }


# --- Tokenize a single shard ---
def tokenize_shard(args):
    shard_idx, shard_dataset = args
    shard_folder = os.path.join(out_dir, f"shard_{shard_idx}")
    lock_folder = shard_folder + ".lock"

    if os.path.exists(shard_folder):
        print(f"[Shard {shard_idx}] Already done, skipping.")
        return shard_folder

    try:
        os.makedirs(lock_folder)
    except FileExistsError:
        print(f"[Shard {shard_idx}] Locked by another process, skipping.")
        return None

    print(f"[Shard {shard_idx}] Tokenizing...")
    with tempfile.TemporaryDirectory(dir=out_dir) as tmpdir:
        tokenized_shard = shard_dataset.map(
            process,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            remove_columns=["text"],
            desc=f"Tokenizing shard {shard_idx}",
            writer_batch_size=1000
        )
        tokenized_shard.save_to_disk(tmpdir)
        shutil.move(tmpdir, shard_folder)

    shutil.rmtree(lock_folder)
    print(f"[Shard {shard_idx}] Done!")
    return shard_folder

# --- Select and process specific shards ---
def process_specific_shards(dataset, shard_size, shard_indices, k):
    os.makedirs(out_dir, exist_ok=True)
    shard_args = []

    for i in shard_indices:
        start = i * shard_size
        end = min((i + 1) * shard_size, len(dataset))
        shard_dataset = dataset.select(range(start, end))
        shard_args.append((i, shard_dataset))

    shard_paths = []
    with Pool(min(k, len(shard_args))) as pool:
        batch_paths = list(tqdm(pool.imap(tokenize_shard, shard_args), total=len(shard_args)))
        shard_paths.extend([p for p in batch_paths if p])

    return shard_paths

# --- Main ---
if __name__ == "__main__":
    shards_to_run = [112,113,114,115,116,117,118,119,120,121,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138] 
    shard_paths = process_specific_shards(dataset, shard_size, shards_to_run, num_workers)

    print("Selected shards written to disk:")
    for p in shard_paths:
        print("  ", p)

