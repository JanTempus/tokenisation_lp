from collections import Counter
import json
import os
import random
from pathlib import Path

from datasets import concatenate_datasets, load_dataset


def discover_parquet_files(base_dir, source_dirs):
    source_to_files = {}
    base_path = Path(base_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Base directory does not exist: {base_dir}")

    for source in source_dirs:
        source_path = base_path / source
        if not source_path.exists():
            raise FileNotFoundError(f"Missing source directory: {source_path}")

        files = sorted(str(path) for path in source_path.rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files found in: {source_path}")
        source_to_files[source] = files

    return source_to_files


def infer_text_column(dataset):
    preferred_columns = ("text", "content", "code")
    for column in preferred_columns:
        if column in dataset.column_names:
            return column

    for name, feature in dataset.features.items():
        dtype = getattr(feature, "dtype", None)
        if dtype in {"string", "large_string"}:
            return name

    if dataset.column_names:
        return dataset.column_names[0]
    raise ValueError("Dataset has no columns")


def normalize_to_text_column(dataset, source_name=None, source_text_columns=None):
    text_column = None
    if source_name and source_text_columns and source_name in source_text_columns:
        candidate = source_text_columns[source_name]
        if candidate in dataset.column_names:
            text_column = candidate
        else:
            raise ValueError(
                f"Configured text column '{candidate}' not found for source '{source_name}'. "
                f"Columns are: {dataset.column_names}"
            )
    else:
        text_column = infer_text_column(dataset)

    if text_column != "text":
        dataset = dataset.rename_column(text_column, "text")
    columns_to_remove = [column for column in dataset.column_names if column != "text"]
    if columns_to_remove:
        dataset = dataset.remove_columns(columns_to_remove)
    return dataset


def draw_source_counts(source_names, total_rows, seed):
    rng = random.Random(seed)
    source_counts = Counter({source: 0 for source in source_names})
    for _ in range(total_rows):
        source_counts[rng.choice(source_names)] += 1
    return dict(source_counts)


def sample_dataset(source_to_files, target_rows, seed, source_text_columns):
    source_names = list(source_to_files.keys())
    source_counts = draw_source_counts(source_names, target_rows, seed)
    rng = random.Random(seed)

    sampled_chunks = []
    sampling_manifest = []

    for source_index, source in enumerate(source_names):
        rows_required = source_counts[source]
        if rows_required == 0:
            continue

        rows_collected = 0
        draw_index = 0
        files = source_to_files[source]

        while rows_collected < rows_required:
            file_path = rng.choice(files)
            file_dataset = load_dataset("parquet", data_files=file_path, split="train")
            file_dataset = normalize_to_text_column(
                file_dataset,
                source_name=source,
                source_text_columns=source_text_columns,
            )
            available_rows = len(file_dataset)

            if available_rows == 0:
                draw_index += 1
                continue

            rows_missing = rows_required - rows_collected
            rows_to_take = min(rows_missing, available_rows)
            shuffle_seed = seed + source_index * 100_000 + draw_index
            sampled_file_rows = file_dataset.shuffle(seed=shuffle_seed).select(range(rows_to_take))

            sampled_chunks.append(sampled_file_rows)
            sampling_manifest.append(
                {
                    "source": source,
                    "file": file_path,
                    "rows": rows_to_take,
                }
            )
            rows_collected += rows_to_take
            draw_index += 1

        print(f"Sampled {rows_collected} rows from {source} (target: {rows_required})")

    if not sampled_chunks:
        raise ValueError("No data sampled from parquet sources")

    sampled_dataset = concatenate_datasets(sampled_chunks).shuffle(seed=seed)
    return sampled_dataset, source_counts, sampling_manifest


def save_sampling_outputs(output_dir, dataset, source_counts, sampling_manifest, target_rows, seed):
    os.makedirs(output_dir, exist_ok=True)
    dataset.save_to_disk(output_dir)
    print(f"Saved sampled dataset to {output_dir} with {len(dataset)} rows")

    manifest_path = os.path.join(output_dir, "sampling_manifest.json")
    payload = {
        "target_rows": target_rows,
        "seed": seed,
        "source_counts": source_counts,
        "draws": sampling_manifest,
    }
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    print(f"Wrote sampling manifest to {manifest_path}")


if __name__ == "__main__":
    DATASET_BASE_DIR = os.environ.get(
        "TOKENIZER_DATASET_BASE",
        "/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/tokenizer_training_dataset",
    )
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
    TARGET_ROWS = int(os.environ.get("TARGET_ROWS", "120000"))
    SEED = int(os.environ.get("SEED", "42"))
    OUTPUT_DATASET_DIR = os.environ.get("OUTPUT_DATASET_DIR", "sampled_tokenizer_data")

    source_to_files = discover_parquet_files(DATASET_BASE_DIR, SOURCE_DIRS)
    dataset, source_counts, sampling_manifest = sample_dataset(
        source_to_files,
        TARGET_ROWS,
        SEED,
        SOURCE_TEXT_COLUMNS,
    )
    save_sampling_outputs(
        OUTPUT_DATASET_DIR,
        dataset,
        source_counts,
        sampling_manifest,
        TARGET_ROWS,
        SEED,
    )
