#!/usr/bin/env python3

import os
from pathlib import Path

from datasets import DatasetDict, load_from_disk


def main():
    output_path = Path(os.environ["PICKY_CORPUS_PATH"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = load_from_disk(os.environ["TRAIN_DATASET_PATH"])
    if isinstance(dataset, DatasetDict):
        if "train" not in dataset:
            raise ValueError(
                "The dataset has no 'train' split. "
                f"Available splits: {list(dataset.keys())}"
            )
        dataset = dataset["train"]

    requested_num_proc = int(os.environ.get("NUM_PROC", "1"))
    if requested_num_proc < 1:
        raise ValueError("NUM_PROC must be at least 1")
    num_proc = min(requested_num_proc, max(1, len(dataset)))
    print(
        f"Writing PickyBPE corpus with {num_proc} worker(s) "
        f"from {len(dataset):,} rows"
    )
    dataset.select_columns(["text"]).to_json(
        output_path,
        num_proc=num_proc,
        orient="records",
        lines=True,
        force_ascii=False,
    )
    print(f"Saved {len(dataset):,} rows to {output_path}")


if __name__ == "__main__":
    main()
