import os
import json
import shutil
import tempfile
from tqdm import tqdm
from datasets import load_from_disk, Dataset
from lp_tokenizer import Tokenizer
from multiprocessing import Pool, cpu_count

# Config
dataset_path = "finewebedu_data"
vocab_size = 32768
num_proc = 8        # processes inside Dataset.map()
batch_size = 100
shard_size = 100000 # examples per shard
out_dir = "tokenized_shards"
num_workers = min(cpu_count(), 12)  # parallel shard workers (k=12)

# Load dataset
dataset = load_from_disk(dataset_path)['train']

# Load vocab
file_path = f"new_vocab/vocab_finewebedu_data_0_{vocab_size}.json"
with open(file_path, 'r', encoding='utf-8') as f:
    vocab = json.load(f)

tokenizer = Tokenizer(vocab_size=vocab_size, vocab=vocab, unk_token="[UNK]")

# --- Processing function ---
def process(batch):
    merged_text = "endoftextbehere".join(batch["text"])
    merged_ids = tokenizer.encode(merged_text, vocab)

    split_ids = []
    current = []
    for token in merged_ids:
        if token == 1:  # <|endoftext|>
            if current:
                split_ids.append(current)
                current = []
        else:
            current.append(token)
    if current:
        split_ids.append(current)

    return {"ids": split_ids, "len": [len(x) for x in split_ids]}

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
            load_from_cache_file=False,   # don’t reuse in-RAM cache
            writer_batch_size=1000        # flush results frequently
        )
        tokenized_shard.save_to_disk(tmpdir)
        shutil.move(tmpdir, shard_folder)

    shutil.rmtree(lock_folder)
    print(f"[Shard {shard_idx}] Done!")
    return shard_folder

# --- Parallel sharding (k shards at a time) ---
def shard_and_tokenize_parallel(dataset, shard_size, k):
    os.makedirs(out_dir, exist_ok=True)
    num_shards = (len(dataset) + shard_size - 1) // shard_size

    shard_args = []
    for i in range(num_shards):
        start = i * shard_size
        end = min((i + 1) * shard_size, len(dataset))
        shard_dataset = dataset.select(range(start, end))
        shard_args.append((i, shard_dataset))

    shard_paths = []
    for i in range(0, len(shard_args), k):   # process k shards at a time
        batch = shard_args[i:i+k]
        with Pool(k) as pool:
            batch_paths = list(tqdm(pool.imap(tokenize_shard, batch), total=len(batch)))
        shard_paths.extend([p for p in batch_paths if p])

    # Save manifest (list of shard paths)
    manifest_file = os.path.join(out_dir, "manifest.json")
    with open(manifest_file, "w") as f:
        json.dump(shard_paths, f, indent=2)

    return shard_paths

# --- Main ---
if __name__ == "__main__":
    shard_paths = shard_and_tokenize_parallel(dataset, shard_size, num_workers)
    print("All shards written to disk.")
    print("Manifest saved at:", os.path.join(out_dir, "manifest.json"))
