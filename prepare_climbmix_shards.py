import os

from datasets import load_dataset
from huggingface_hub import list_repo_files


DATASET_ID = os.environ.get("DATASET_ID", "karpathy/climbmix-400b-shuffle")
BASE_URL = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/main"
NUM_SHARDS = int(os.environ.get("NUM_SHARDS", "8"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR")


def main():
    if not OUTPUT_DIR:
        raise SystemExit("OUTPUT_DIR env var is required")

    print(f"Listing parquet shards for {DATASET_ID}...")
    all_shards = sorted(
        f for f in list_repo_files(DATASET_ID, repo_type="dataset")
        if f.endswith(".parquet")
    )
    if not all_shards:
        raise SystemExit(f"No parquet shards found in {DATASET_ID}")

    selected = all_shards[:NUM_SHARDS]
    print(f"Using {len(selected)}/{len(all_shards)} shards:")
    for shard in selected:
        print(f"  - {shard}")

    shard_urls = [f"{BASE_URL}/{f}" for f in selected]

    print("Loading shards...")
    dataset = load_dataset("parquet", data_files={"train": shard_urls}, split="train")
    print(f"Loaded {len(dataset):,} rows; columns={dataset.column_names}")

    if "text" not in dataset.column_names:
        raise SystemExit(
            f"Expected a 'text' column in climbmix shards, got {dataset.column_names}"
        )
    dataset = dataset.select_columns(["text"])

    print(f"Saving dataset to {OUTPUT_DIR}")
    dataset.save_to_disk(OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
