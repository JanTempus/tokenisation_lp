#!/usr/bin/env python3

import os
from pathlib import Path

from datasets import load_from_disk


def main():
    output_path = Path(os.environ["PICKY_CORPUS_PATH"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = load_from_disk(os.environ["TRAIN_DATASET_PATH"])
    dataset.select_columns(["text"]).to_json(
        output_path,
        num_proc=64,
        orient="records",
        lines=True,
        force_ascii=False,
    )
    print(f"Saved {len(dataset):,} rows to {output_path}")


if __name__ == "__main__":
    main()
