import argparse
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
import json
import os
import random
from pathlib import Path

from datasets import Value, concatenate_datasets, load_dataset


_CLIMBMIX_SHARD_LIST_CACHE = {}
FINEWEB2_DATASET_ID = "HuggingFaceFW/fineweb-2"
FINEWEB_DATASET_ID = "HuggingFaceFW/fineweb"
FINEWEB2_LANGUAGE_CONFIGS = (
    "cmn_Hani",
    "fra_Latn",
    "arb_Arab",
    "rus_Cyrl",
    "tha_Thai",
    "hin_Deva",
    "tur_Latn",
    "swh_Latn",
    "tel_Telu",
)
FINEWEB_ENGLISH_SOURCE = "eng_Latn"
FINEWEB_ENGLISH_PREFIX = "sample/10BT/"


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


def allocate_equal_quotas(source_names, target_rows):
    """Divide target_rows as evenly as possible in declared source order."""
    source_names = list(source_names)
    if not source_names:
        raise ValueError("At least one source is required")
    if len(source_names) != len(set(source_names)):
        raise ValueError(f"Source names must be unique: {source_names}")
    if target_rows <= 0:
        raise ValueError(f"target_rows must be positive, got {target_rows}")

    rows_per_source, remainder = divmod(target_rows, len(source_names))
    return {
        source: rows_per_source + (1 if index < remainder else 0)
        for index, source in enumerate(source_names)
    }


def build_fineweb_prefix_sources(
    fineweb2_dataset_id=FINEWEB2_DATASET_ID,
    fineweb_dataset_id=FINEWEB_DATASET_ID,
):
    sources = [
        {
            "source": config,
            "dataset_id": fineweb2_dataset_id,
            "parquet_prefix": f"data/{config}/train/",
        }
        for config in FINEWEB2_LANGUAGE_CONFIGS
    ]
    sources.append(
        {
            "source": FINEWEB_ENGLISH_SOURCE,
            "dataset_id": fineweb_dataset_id,
            "parquet_prefix": FINEWEB_ENGLISH_PREFIX,
        }
    )
    return sources


def list_ordered_parquet_shards(source, list_repo_tree_fn=None):
    if list_repo_tree_fn is None:
        from huggingface_hub import list_repo_tree

        list_repo_tree_fn = list_repo_tree

    repo_entries = list_repo_tree_fn(
        source["dataset_id"],
        path_in_repo=source["parquet_prefix"].rstrip("/"),
        recursive=False,
        repo_type="dataset",
    )
    shards = sorted(
        entry.path
        for entry in repo_entries
        if getattr(entry, "path", "").endswith(".parquet")
    )
    if not shards:
        raise FileNotFoundError(
            "No parquet shards found for "
            f"source={source['source']} dataset={source['dataset_id']} "
            f"prefix={source['parquet_prefix']}"
        )
    return shards


def _first_valid_text_indices(dataset, limit):
    indices = []
    for index in range(len(dataset)):
        text = dataset[index]["text"]
        if isinstance(text, str) and text.strip():
            indices.append(index)
            if len(indices) == limit:
                break
    return indices


def materialize_fineweb_prefix_dataset(
    target_rows,
    sources=None,
    list_repo_tree_fn=None,
    hf_hub_download_fn=None,
    load_dataset_fn=None,
    concatenate_datasets_fn=None,
):
    """Download ordered parquet shards and retain a balanced prefix of valid texts."""
    if sources is None:
        sources = build_fineweb_prefix_sources()
    sources = list(sources)
    source_names = [source["source"] for source in sources]
    source_quotas = allocate_equal_quotas(source_names, target_rows)

    if hf_hub_download_fn is None:
        from huggingface_hub import hf_hub_download

        hf_hub_download_fn = hf_hub_download
    if load_dataset_fn is None:
        load_dataset_fn = load_dataset
    if concatenate_datasets_fn is None:
        concatenate_datasets_fn = concatenate_datasets

    selected_chunks = []
    shard_manifest = []
    source_counts = {}

    for source in sources:
        source_name = source["source"]
        quota = source_quotas[source_name]
        retained = 0
        shards = list_ordered_parquet_shards(source, list_repo_tree_fn)
        print(
            f"[{source_name}] selecting first {quota} valid documents from "
            f"{source['dataset_id']}"
        )

        for shard in shards:
            if retained >= quota:
                break

            print(f"[{source_name}] downloading {shard}")
            local_path = hf_hub_download_fn(
                repo_id=source["dataset_id"],
                filename=shard,
                repo_type="dataset",
            )
            shard_dataset = load_dataset_fn(
                "parquet",
                data_files=local_path,
                split="train",
            )
            shard_dataset = normalize_to_text_column(
                shard_dataset,
                source_name=source_name,
                source_text_columns={source_name: "text"},
            )

            rows_needed = quota - retained
            valid_indices = _first_valid_text_indices(shard_dataset, rows_needed)
            if valid_indices:
                selected_chunks.append(shard_dataset.select(valid_indices))
                retained += len(valid_indices)

            shard_manifest.append(
                {
                    "source": source_name,
                    "dataset_id": source["dataset_id"],
                    "repository_path": shard,
                    "cache_path": str(local_path),
                    "downloaded_rows": len(shard_dataset),
                    "retained_rows": len(valid_indices),
                }
            )
            print(
                f"[{source_name}] retained {retained}/{quota} documents "
                f"after {shard}"
            )

        if retained != quota:
            raise RuntimeError(
                f"Could not fill quota for {source_name}: retained {retained}/{quota} "
                f"after examining {len(shards)} parquet shards"
            )
        source_counts[source_name] = retained

    if not selected_chunks:
        raise RuntimeError("FineWeb/FineWeb2 prefix selection produced no data")

    dataset = concatenate_datasets_fn(selected_chunks)
    dataset = normalize_to_text_column(dataset)
    if dataset.column_names != ["text"]:
        raise RuntimeError(
            f"Expected a text-only dataset, got columns={dataset.column_names}"
        )
    if len(dataset) != target_rows:
        raise RuntimeError(
            f"Expected {target_rows} rows after concatenation, got {len(dataset)}"
        )
    return dataset, source_counts, shard_manifest


def get_library_versions():
    versions = {}
    for package_name in ("datasets", "huggingface_hub", "tokenizers"):
        try:
            versions[package_name] = version(package_name)
        except PackageNotFoundError:
            versions[package_name] = None
    return versions


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


def save_sampling_outputs(
    output_dir,
    dataset,
    source_counts,
    sampling_manifest,
    target_rows,
    seed=None,
    selection_strategy=None,
    manifest_metadata=None,
):
    os.makedirs(output_dir, exist_ok=True)
    dataset.save_to_disk(output_dir)
    print(f"Saved sampled dataset to {output_dir} with {len(dataset)} rows")

    manifest_path = os.path.join(output_dir, "sampling_manifest.json")
    payload = {
        "target_rows": target_rows,
        "source_counts": source_counts,
        "draws": sampling_manifest,
    }
    if seed is not None:
        payload["seed"] = seed
    if selection_strategy is not None:
        payload["selection_strategy"] = selection_strategy
    if manifest_metadata:
        payload.update(manifest_metadata)
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
        default=None,
    )
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42")))
    parser.add_argument(
        "--output-dataset-dir",
        default=os.environ.get("OUTPUT_DATASET_DIR"),
    )
    args = parser.parse_args()

    source = args.source.strip().lower()
    if args.target_rows is None:
        default_target_rows = "60000" if source == "fineweb2" else "120000"
        args.target_rows = int(os.environ.get("TARGET_ROWS", default_target_rows))

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
    elif source == "fineweb2":
        if args.name is None:
            parser.error("--name is required when --source fineweb2")

        fineweb2_dataset_id = os.environ.get(
            "FINEWEB2_DATASET_ID", FINEWEB2_DATASET_ID
        )
        fineweb_dataset_id = os.environ.get(
            "FINEWEB_DATASET_ID", FINEWEB_DATASET_ID
        )
        sources = build_fineweb_prefix_sources(
            fineweb2_dataset_id=fineweb2_dataset_id,
            fineweb_dataset_id=fineweb_dataset_id,
        )
        output_dir = args.output_dataset_dir or (
            f"{args.name}_fineweb2_10lang_n{args.target_rows}"
        )
        dataset, source_counts, shard_manifest = materialize_fineweb_prefix_dataset(
            args.target_rows,
            sources=sources,
        )
        save_sampling_outputs(
            output_dir,
            dataset,
            source_counts,
            shard_manifest,
            args.target_rows,
            selection_strategy="ordered_parquet_prefix",
            manifest_metadata={
                "sources": sources,
                "library_versions": get_library_versions(),
            },
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
        parser.error(
            f"Unknown --source '{source}'. Expected: parquet, finewebedu, fineweb2"
        )
