import json
import os
from pathlib import Path

from datasets import Value, concatenate_datasets, load_dataset


SOURCE_DIRS = [
    "fineweb2",
    "fineweb",
    "megamath",
    "infimath",
    "finemath",
    "starcoder",
]
SOURCE_TEXT_COLUMNS = {
    "fineweb2": "text",
    "fineweb": "text",
    "megamath": "text",
    "infimath": "text",
    "finemath": "text",
    "starcoder": "content",
}


def discover_parquet_files(base_dir, source_dirs):
    source_to_files = {}
    base_path = Path(base_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Base directory does not exist: {base_dir}")

    for source in source_dirs:
        source_path = base_path / source
        if not source_path.exists():
            print(f"[WARN] Missing source directory: {source_path}, skipping.")
            continue

        files = sorted(str(path) for path in source_path.rglob("*.parquet"))
        if not files:
            print(f"[WARN] No parquet files found in: {source_path}, skipping.")
            continue
        source_to_files[source] = files

    if not source_to_files:
        raise FileNotFoundError(f"No parquet files found under any source in: {base_dir}")

    return source_to_files


def normalize_to_text_column(dataset, source_name=None):
    text_column = SOURCE_TEXT_COLUMNS.get(source_name) if source_name else None
    if text_column is None or text_column not in dataset.column_names:
        preferred = ("text", "content", "code")
        for col in preferred:
            if col in dataset.column_names:
                text_column = col
                break
        else:
            text_column = dataset.column_names[0]

    if text_column != "text":
        dataset = dataset.rename_column(text_column, "text")
    columns_to_remove = [col for col in dataset.column_names if col != "text"]
    if columns_to_remove:
        dataset = dataset.remove_columns(columns_to_remove)
    text_dtype = getattr(dataset.features.get("text"), "dtype", None)
    if text_dtype != "string":
        dataset = dataset.cast_column("text", Value("string"))
    return dataset


def load_full_dataset(base_dir, source_dirs):
    source_to_files = discover_parquet_files(base_dir, source_dirs)
    source_datasets = []
    for source, files in source_to_files.items():
        print(f"Loading {len(files)} parquet files from source '{source}'...")
        chunks = []
        for i, f in enumerate(files, 1):
            ds = load_dataset("parquet", data_files=f, split="train")
            ds = normalize_to_text_column(ds, source_name=source)
            chunks.append(ds)
            if i % 100 == 0 or i == len(files):
                print(f"  {source}: {i}/{len(files)} files loaded")
        source_ds = concatenate_datasets(chunks) if len(chunks) > 1 else chunks[0]
        print(f"  {source}: {len(source_ds)} rows total")
        source_datasets.append(source_ds)

    full_dataset = concatenate_datasets(source_datasets)
    print(f"Full dataset: {len(full_dataset)} rows across {len(source_datasets)} sources")
    return full_dataset


def split_into_chunks(dataset, chunk_size, output_dir):
    total_rows = len(dataset)
    num_chunks = (total_rows + chunk_size - 1) // chunk_size
    print(f"Splitting {total_rows} rows into {num_chunks} chunks of up to {chunk_size} rows each")

    os.makedirs(output_dir, exist_ok=True)
    chunk_paths = []

    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, total_rows)
        chunk = dataset.select(range(start, end))
        chunk_path = os.path.join(output_dir, f"chunk_{i:04d}")
        chunk.save_to_disk(chunk_path)
        chunk_paths.append(chunk_path)
        print(f"  Saved chunk {i:04d}: rows {start}-{end-1} -> {chunk_path}")

    return chunk_paths, num_chunks


if __name__ == "__main__":
    DATASET_BASE_DIR = os.environ.get(
        "TOKENIZER_DATASET_BASE",
        "/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/tokenizer_training_dataset",
    )
    CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "50000"))
    CHUNK_OUTPUT_DIR = os.environ.get("CHUNK_OUTPUT_DIR", "dataset_chunks")

    full_dataset = load_full_dataset(DATASET_BASE_DIR, SOURCE_DIRS)
    chunk_paths, num_chunks = split_into_chunks(full_dataset, CHUNK_SIZE, CHUNK_OUTPUT_DIR)

    manifest = {
        "chunk_size": CHUNK_SIZE,
        "num_chunks": num_chunks,
        "total_rows": len(full_dataset),
        "chunks": chunk_paths,
    }
    manifest_path = os.path.join(CHUNK_OUTPUT_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest to {manifest_path}")
