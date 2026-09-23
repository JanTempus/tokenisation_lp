#!/usr/bin/env python3

import os
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files


DATASET_ID = os.environ.get("DATASET_ID", "karpathy/climbmix-400b-shuffle")
NUM_SHARDS = int(os.environ.get("NUM_SHARDS", "7"))
NUM_PROC = int(os.environ.get("NUM_PROC", "64"))
ROW_FRACTION = float(os.environ.get("CLIMBMIX_ROW_FRACTION", "1"))


def main():
    if not 0 < ROW_FRACTION <= 1:
        raise ValueError("CLIMBMIX_ROW_FRACTION must be greater than 0 and at most 1")

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
    if ROW_FRACTION < 1:
        selected_rows = max(1, int(len(dataset) * ROW_FRACTION))
        dataset = dataset.select(range(selected_rows))
        print(
            f"Selected the first {selected_rows:,} rows "
            f"({ROW_FRACTION:.1%}) of the downloaded data"
        )
    dataset.save_to_disk(output_dir, num_proc=NUM_PROC)
    print(f"Saved {len(dataset):,} rows to {output_dir}")


if __name__ == "__main__":
    main()
