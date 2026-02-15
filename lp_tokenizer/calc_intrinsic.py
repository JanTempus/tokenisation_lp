import os
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
import csv

# ---------- Config ----------
num_proc = 8
dataset_size = 65536
dataset_url="pietrolesci/finewebedu-20B"
tokenizers_dir = "/local/home/jtempus/tokenisation_lp/rounded_tokenizers"
output_file = "dataset_stats_lp_2.csv"
append_eot = "[EOS]"  # keep consistent with your previous runs
batch_size_map = 10000
batch_size_vocab = 1024  # batching for vocab utilization pass

# ---------- Data ----------
dataset = load_dataset(dataset_url)["train"].select(range(dataset_size, 2 * dataset_size))

# ---------- Map pass: length & fertility ----------
def process_stats(batch, tokenizer):
    lengths = []
    fertilities = []
    for text in batch["text"]:
        text2 = text + append_eot
        n_chars = len(text2)
        ids = tokenizer.encode(text2)
        n_tokens = len(ids)
        lengths.append(n_tokens)
        fertilities.append((n_tokens / n_chars) if n_chars > 0 else 0.0)
    return {"length": lengths, "fertility": fertilities}

# ---------- Single-process pass: vocab utilization ----------
def compute_vocab_utilization(ds, tokenizer, step=batch_size_vocab):
    all_tokens = set()
    for i in tqdm(range(0, len(ds), step), desc="Collecting unique tokens (single-process)"):
        texts = ds[i : min(i + step, len(ds))]["text"]
        texts = [t + append_eot for t in texts]
        # Use fast batch encode
        out = tokenizer(texts, add_special_tokens=True, return_attention_mask=False, return_token_type_ids=False)
        for ids in out["input_ids"]:
            all_tokens.update(ids)
    vocab_size_total = len(tokenizer)  # includes added tokens
    return (len(all_tokens) / vocab_size_total) if vocab_size_total > 0 else 0.0

# ---------- Main loop over tokenizers ----------
for tokenizer_name in os.listdir(tokenizers_dir):
    tokenizer_path = os.path.join(tokenizers_dir, tokenizer_name)

    # only keep directories ending in "det" or "bias" (case-insensitive)
    if not (
        os.path.isdir(tokenizer_path)
        and tokenizer_name.lower().endswith(("det", "bias"))
    ):
        continue

    print(f"\nProcessing tokenizer: {tokenizer_name}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

    # 1) Parallel map for per-example stats
    tokenized = dataset.map(
        lambda batch: process_stats(batch, tokenizer),
        remove_columns=["text"],
        desc=f"Tokenizing with {tokenizer_name}",
        batched=True,
        batch_size=batch_size_map,
        num_proc=num_proc,
    )

    avg_length = float(np.mean(tokenized["length"])) if len(tokenized) else 0.0
    avg_fertility = float(np.mean(tokenized["fertility"])) if len(tokenized) else 0.0

    # 2) Single-process pass for vocab utilization (no shared-state issues)
    vocab_utilization = compute_vocab_utilization(dataset, tokenizer, step=batch_size_vocab)

    stats = {
        "tokenizer": tokenizer_name,
        "avg_compression": avg_length,
        "avg_fertility": avg_fertility,
        "vocab_utilization": vocab_utilization,
        "vocab_size": len(tokenizer),
        "dataset_size": dataset_size,
    }

    # Write/append CSV
    file_exists = os.path.exists(output_file)
    with open(output_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=stats.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(stats)

    print(f"Stats written/appended: {stats}")
