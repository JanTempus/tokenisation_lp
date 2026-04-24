import argparse
from collections import Counter
import json
import os
import random
from pathlib import Path

from datasets import Value, concatenate_datasets, load_dataset


_CLIMBMIX_SHARD_LIST_CACHE = {}


def _list_climbmix_shards(dataset_id):
    if dataset_id not in _CLIMBMIX_SHARD_LIST_CACHE:
        from huggingface_hub import list_repo_files
        shards = sorted(
            f for f in list_repo_files(dataset_id, repo_type="dataset")
            if f.endswith(".parquet")
        )
        if not shards:
            raise ValueError(f"No parquet shards found in {dataset_id}")
        _CLIMBMIX_SHARD_LIST_CACHE[dataset_id] = shards
    return _CLIMBMIX_SHARD_LIST_CACHE[dataset_id]


def sample_climbmix(dataset_id, num_shards, target_rows, seed, shard_tmp_dir):
    """Two-tier sampling against a HF parquet dataset (e.g. climbmix).

    Tier 1: randomly pick `num_shards` parquet shards from `dataset_id` using
            `seed`, and download only those shards into `shard_tmp_dir`.
    Tier 2: shuffle the concatenated shard rows with `seed` and select
            `target_rows`.

    Returns (dataset, source_counts, manifest, shard_tmp_dir). The caller is
    responsible for deleting `shard_tmp_dir` after saving the dataset.
    """
    from huggingface_hub import hf_hub_download

    all_shards = _list_climbmix_shards(dataset_id)
    if num_shards > len(all_shards):
        raise ValueError(
            f"Requested num_shards={num_shards} > available shards={len(all_shards)} "
            f"for {dataset_id}"
        )

    rng = random.Random(seed)
    selected = rng.sample(all_shards, num_shards)
    print(f"climbmix tier-1: seed={seed} picked shards: {selected}")

    os.makedirs(shard_tmp_dir, exist_ok=True)
    local_paths = []
    for shard in selected:
        print(f"  downloading {shard} -> {shard_tmp_dir}")
        local_path = hf_hub_download(
            repo_id=dataset_id,
            filename=shard,
            repo_type="dataset",
            local_dir=shard_tmp_dir,
        )
        local_paths.append(local_path)

    dataset = load_dataset("parquet", data_files=local_paths, split="train")
    if "text" not in dataset.column_names:
        raise ValueError(
            f"Expected a 'text' column in {dataset_id} shards, got {dataset.column_names}"
        )
    dataset = dataset.select_columns(["text"])

    if target_rows > len(dataset):
        raise ValueError(
            f"target_rows={target_rows} exceeds rows available in "
            f"{num_shards} shards ({len(dataset)}). Increase NUM_SHARDS."
        )

    dataset = dataset.shuffle(seed=seed).select(range(target_rows))

    source_counts = {"climbmix": target_rows}
    manifest = [
        {"source": "climbmix", "dataset_id": dataset_id, "shard": shard, "local_path": local_path}
        for shard, local_path in zip(selected, local_paths)
    ]
    return dataset, source_counts, manifest, shard_tmp_dir


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
    text_dtype = getattr(dataset.features.get("text"), "dtype", None)
    if text_dtype != "string":
        dataset = dataset.cast_column("text", Value("string"))
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


def sample_finewebedu(target_rows, seed):
    print(f"Loading pietrolesci/finewebedu-20B (split=train)")
    dataset = load_dataset("pietrolesci/finewebedu-20B", split="train")
    print(f"Loaded {len(dataset)} rows, shuffling and selecting {target_rows}")
    dataset = dataset.shuffle(seed=seed).select(range(target_rows))
    dataset = dataset.select_columns(["text"])
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=os.environ.get("NAME"))
    parser.add_argument("--source", default=os.environ.get("SOURCE", "parquet"))
    parser.add_argument(
        "--target-rows",
        type=int,
        default=int(os.environ.get("TARGET_ROWS", "120000")),
    )
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42")))
    parser.add_argument(
        "--output-dataset-dir",
        default=os.environ.get("OUTPUT_DATASET_DIR"),
    )
    args = parser.parse_args()

    source = args.source.strip().lower()

    if source == "finewebedu":
        if args.name is None:
            parser.error("--name is required when --source finewebedu")
        output_dir = args.output_dataset_dir or (
            f"{args.name}_finewebedu20B_n{args.target_rows}_seed{args.seed}"
        )
        dataset = sample_finewebedu(args.target_rows, args.seed)
        save_sampling_outputs(
            output_dir,
            dataset,
            source_counts={"finewebedu20B": args.target_rows},
            sampling_manifest=[{"source": "finewebedu20B", "rows": args.target_rows}],
            target_rows=args.target_rows,
            seed=args.seed,
        )
    elif source == "parquet":
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
        output_dir = args.output_dataset_dir or "sampled_tokenizer_data"

        source_to_files = discover_parquet_files(DATASET_BASE_DIR, SOURCE_DIRS)
        dataset, source_counts, sampling_manifest = sample_dataset(
            source_to_files,
            args.target_rows,
            args.seed,
            SOURCE_TEXT_COLUMNS,
        )
        save_sampling_outputs(
            output_dir,
            dataset,
            source_counts,
            sampling_manifest,
            args.target_rows,
            args.seed,
        )
    else:
        parser.error(f"Unknown --source '{source}'. Expected: parquet, finewebedu")
