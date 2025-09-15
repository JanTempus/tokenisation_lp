from datasets import load_from_disk, DatasetDict, load_dataset,Sequence,Value,Features
from transformers import AutoTokenizer
import numpy as np


# --- Config ---
dataset_path = "pietrolesci/finewebedu-20B"  # let HF do the caching for you
out_dir = "tokenized_dataset"
tokenizer_path = "tokenizers_lp/lp_32768_finewebedu_data"
batch_size = 1000
num_proc = 8            # parallel workers for Dataset.map
val_frac = 0.1



# --- Tokenization function ---
# def process(batch: dict) -> dict:
#     input_ids = tokenizer(batch["text"])
#     return {"input_ids": input_ids, "len": [len(x) for x in input_ids]}
def process(batch: dict) -> dict:
    tokens = tokenizer(batch["text"])["input_ids"]
    return {"input_ids": tokens,
        "len": [len(x) for x in tokens]
    }



# NOTE: (best practice) use this when multiproc functions are called
if __name__ == "__main__":

    # --- Load dataset ---
    dataset = load_dataset(dataset_path)  # let HF do the caching for you
    if isinstance(dataset, DatasetDict):
        dataset = dataset["train"]  # flatten

    dataset = dataset.select_columns(["id", "text"])  # keep only the required columns


    # --- Train/val/test split ---
    # Read the docs: https://huggingface.co/docs/datasets/en/process#split
    # NOTE: here we want a DatasetDict because it's more convenient
    ds_dict = dataset.train_test_split(test_size=2_000_000, seed=42)

    # NOTE: However, we want to rename "test" to "validation"
    ds_dict["validation"] = ds_dict.pop("test")


    # --- Load tokenizer ---
    # NOTE: use the AutoTokenizer instead of PreTrainedTokeniserFast
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)


    # --- Tokenize ---
    # NOTE: read the docs https://huggingface.co/docs/datasets/v4.0.0/en/package_reference/main_classes#datasets.DatasetDict.map
    ds_dict = ds_dict.map(
        process,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        remove_columns=["text"],
        desc="Tokenizing"
    )

    # --- Save ---
    # NOTE: read the docs https://huggingface.co/docs/datasets/v4.0.0/en/package_reference/main_classes#datasets.DatasetDict.map
    ds_dict.save_to_disk(out_dir, max_shard_size="3GB")

    # optionally, you can save on HF
    # NOTE: read the docs https://huggingface.co/docs/datasets/v4.0.0/en/package_reference/main_classes#datasets.IterableDataset.push_to_hub
    # ds_dict.push_to_hub(repo_id, tokenizer_path)
    # This create a repo under username/repo_id/tokenizer_path. Note that it's cool to pass tokenizer_path because it creates a subfolder so you can in principle have all tokenized version of this dataset under one repo_id