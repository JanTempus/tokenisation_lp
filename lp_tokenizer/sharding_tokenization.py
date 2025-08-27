import os
import json
import shutil
import tempfile
from tqdm import tqdm
from datasets import load_from_disk, Dataset
from lp_tokenizer import Tokenizer
from multiprocessing import Pool, cpu_count

# ---------------- Configuration ----------------
dataset_path = "finewebedu_data"
vocab_size = 32768
num_proc = 8        # processes inside Dataset.map()
batch_size = 100
shard_size = 400000  # NEW shard size
out_dir = "tokenized_shards"
num_workers = min(cpu_count(), 12)  # parallel shard workers (k=12)

# ---------------- Load dataset ----------------
dataset = load_from_disk(dataset_path)['train']

# ---------------- Load vocab ----------------
file_path = f"new_vocab/vocab_finewebedu_data_0_{vocab_size}.json"
with open(file_path, 'r', encoding='utf-8') as f:
    vocab = json.load(f)

tokenizer = Tokenizer(vocab_size=vocab_size, vocab=vocab, unk_token="[UNK]")

# ---------------- Tokenization function ----------------
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

# ---------------- Shard tokenization ----------------
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
            load_from_cache_file=False,
            writer_batch_size=1000
        )
        tokenized_shard.save_to_disk(tmpdir)
        shutil.move(tmpdir, shard_folder)

    shutil.rmtree(lock_folder)
    print(f"[Shard {shard_idx}] Done!")
    return shard_folder

# ---------------- Existing shards detection ----------------
def get_existing_shard_indices(out_dir):
    indices = []
    if os.path.exists(out_dir):
        for folder in os.listdir(out_dir):
            if folder.startswith("shard_") and os.path.isdir(os.path.join(out_dir, folder)):
                idx = int(folder.split("_")[1])
                indices.append(idx)
    return set(indices)

def shard_index_to_range(shard_idx, shard_size):
    start = shard_idx * shard_size
    end = start + shard_size
    return start, end

# ---------------- Shard and tokenize with resume ----------------
def shard_and_tokenize_parallel(dataset, shard_size, k, old_shard_size=None):
    os.makedirs(out_dir, exist_ok=True)
    num_shards = (len(dataset) + shard_size - 1) // shard_size

    existing_shards = get_existing_shard_indices(out_dir)
    shard_args = []

    for i in range(num_shards):
        start = i * shard_size
        end = min((i + 1) * shard_size, len(dataset))

        # Check if this range overlaps with any existing shard
        overlaps = False
        if old_shard_size is not None:
            for old_idx in existing_shards:
                old_start, old_end = shard_index_to_range(old_idx, old_shard_size)
                if not (end <= old_start or start >= old_end):
                    overlaps = True
                    break
        elif i in existing_shards:
            overlaps = True

        if not overlaps:
            shard_dataset = dataset.select(range(start, end))
            shard_args.append((i, shard_dataset))

    shard_paths = []
    for i in range(0, len(shard_args), k):
        batch = shard_args[i:i+k]
        with Pool(k) as pool:
            batch_paths = list(tqdm(pool.imap(tokenize_shard, batch), total=len(batch)))
        shard_paths.extend([p for p in batch_paths if p])

    # Merge with old manifest if exists
    manifest_file = os.path.join(out_dir, "manifest.json")
    old_manifest = []
    if os.path.exists(manifest_file):
        with open(manifest_file, "r") as f:
            old_manifest = json.load(f)

    new_manifest = old_manifest + [p for p in shard_paths if p not in old_manifest]

    with open(manifest_file, "w") as f:
        json.dump(new_manifest, f, indent=2)

    return new_manifest

# ---------------- Main ----------------
if __name__ == "__main__":
    shard_paths = shard_and_tokenize_parallel(dataset, shard_size, num_workers, old_shard_size=100000)
    print("All shards written to disk.")
    print("Manifest saved at:", os.path.join(out_dir, "manifest.json"))
