#!/usr/bin/env python3

import os
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files


DATASET_ID = "karpathy/climbmix-400b-shuffle"
NUM_SHARDS = 7
NUM_PROC = 64


def main():
    output_dir = Path(os.environ["TRAIN_DATASET_PATH"])
    if output_dir.exists():
        print(f"Using existing dataset: {output_dir}")
        return

    shard_dir = Path(f"{output_dir}_parquet")
    shard_dir.mkdir(parents=True, exist_ok=True)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    shards = sorted(
        filename
        for filename in list_repo_files(DATASET_ID, repo_type="dataset")
        if filename.endswith(".parquet")
    )[:NUM_SHARDS]

    local_shards = [
        hf_hub_download(
            repo_id=DATASET_ID,
            filename=filename,
            repo_type="dataset",
            local_dir=shard_dir,
        )
        for filename in shards
    ]

    dataset = load_dataset(
        "parquet",
        data_files=local_shards,
        split="train",
        num_proc=NUM_SHARDS,
    ).select_columns(["text"])
    dataset.save_to_disk(output_dir, num_proc=NUM_PROC)
    print(f"Saved {len(dataset):,} rows to {output_dir}")


if __name__ == "__main__":
    main()
