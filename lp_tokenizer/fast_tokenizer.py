import os
from datasets import load_from_disk, DatasetDict, concatenate_datasets
from transformers import PreTrainedTokenizerFast
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import shutil

# --- Config ---
dataset_path = "finewebedu_data"
out_dir = "tokenized_dataset"
tokenizer_path = "tokenizers_lp/lp_1024_finewebedu_data"
batch_size = 1000
num_proc = 16            # parallel workers for Dataset.map
shard_size = 100000      # examples per shard to avoid memory issues
val_frac = 0.1
test_frac = 0.1
eos_token = "endoftextbehere"

# --- Load dataset ---
dataset = load_from_disk(dataset_path)
if "train" in dataset:  # flatten DatasetDict if necessary
    dataset = dataset["train"]

# --- Train/val/test split ---
all_indices = list(range(len(dataset)))
train_val_ids, test_ids = train_test_split(all_indices, test_size=test_frac, random_state=42)
train_ids, val_ids = train_test_split(train_val_ids, test_size=val_frac/(1-test_frac), random_state=42)

splits = {
    "train": dataset.select(train_ids),
    "validation": dataset.select(val_ids),
    "test": dataset.select(test_ids)
}

# --- Load tokenizer ---
tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

# --- Tokenization function ---
def process(batch):
    merged_text = eos_token.join(batch["text"])
    merged_ids = tokenizer.encode(merged_text)

    split_ids = []
    current = []
    for token in merged_ids:
        if token == tokenizer.eos_token_id:
            if current:
                split_ids.append(current)
                current = []
        else:
            current.append(token)
    if current:
        split_ids.append(current)

    return {"ids": split_ids, "len": [len(x) for x in split_ids]}

# --- Memory-efficient parallel shard processing ---
def tokenize_and_save_split(split_name, split_dataset):
    split_out = os.path.join(out_dir, split_name)
    os.makedirs(split_out, exist_ok=True)
    num_shards = (len(split_dataset) + shard_size - 1) // shard_size

    for shard_idx in tqdm(range(num_shards), desc=f"Tokenizing {split_name}"):
        start = shard_idx * shard_size
        end = min((shard_idx + 1) * shard_size, len(split_dataset))
        shard_dataset = split_dataset.select(range(start, end))
        tokenized_shard = shard_dataset.map(
            process,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            remove_columns=["text"]
        )
        shard_folder = os.path.join(split_out, f"shard_{shard_idx}")
        if os.path.exists(shard_folder):
            shutil.rmtree(shard_folder)
        tokenized_shard.save_to_disk(shard_folder)

# --- Run tokenization for all splits ---
for split_name, split_dataset in splits.items():
    tokenize_and_save_split(split_name, split_dataset)

print(f"Tokenized dataset saved in shards at: {out_dir}")
