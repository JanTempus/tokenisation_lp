from pathlib import Path

from datasets import load_dataset, load_from_disk


BASE_PATH = "/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/tokenizer_training_dataset"


if __name__ == "__main__":
    base = Path(BASE_PATH)
    if not base.exists():
        print("BASE_PATH_NOT_FOUND")
        print(BASE_PATH)
        raise SystemExit(1)

    subdirs = sorted(path for path in base.iterdir() if path.is_dir())
    if not subdirs:
        print("NO_SUBDIRECTORIES_FOUND")
        print(BASE_PATH)
        raise SystemExit(1)

    for subdir in subdirs:
        print(f"\n=== {subdir.name} ===")

        try:
            dataset_obj = load_from_disk(str(subdir))
            if hasattr(dataset_obj, "keys"):
                split_name = "train" if "train" in dataset_obj else list(dataset_obj.keys())[0]
                row_count = len(dataset_obj[split_name])
                print(f"LOAD_OK_FROM_DISK split={split_name} rows={row_count}")
            else:
                print(f"LOAD_OK_FROM_DISK rows={len(dataset_obj)}")
            continue
        except Exception as from_disk_error:
            print(f"LOAD_FROM_DISK_FAILED {type(from_disk_error).__name__}: {from_disk_error}")

        parquet_files = sorted(str(path) for path in subdir.rglob("*.parquet"))
        if not parquet_files:
            print("NO_PARQUET_FILES_FOUND")
            continue

        try:
            dataset = load_dataset("parquet", data_files=parquet_files, split="train")
            print(f"LOAD_OK_PARQUET files={len(parquet_files)} rows={len(dataset)}")
        except Exception as parquet_error:
            print(f"LOAD_PARQUET_FAILED {type(parquet_error).__name__}: {parquet_error}")
