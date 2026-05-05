from pathlib import Path
from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files
from transformers import PreTrainedTokenizerFast


DATASET_ID = "karpathy/climbmix-400b-shuffle"
NUM_SHARDS = 7
DOWNLOAD_DIR = "climbmix_eval_shards"
BATCH_SIZE = 1000
NUM_TOKENIZER_WORKERS = 8
ADD_SPECIAL_TOKENS = False
TOKENIZER_ROOTS = [
    "/home/jantempus/Desktop/Projects/NLP/tokenisation_lp/rounded_tokenizers/cross_over_climbmix400b_s7",
    #"/home/jantempus/Desktop/Projects/NLP/tokenisation_lp/bpe_tokenizers_climbmix/nano_bpe_climb_mix",
]
TOKEN_COUNT_COLUMN = "token_count"
_TOKENIZER_CACHE = {}


def list_first_parquet_shards(dataset_id, num_shards):
    shards = sorted(
        file_name
        for file_name in list_repo_files(dataset_id, repo_type="dataset")
        if file_name.endswith(".parquet")
    )
    if not shards:
        raise RuntimeError(f"No parquet shards found in {dataset_id}")
    if num_shards > len(shards):
        raise RuntimeError(
            f"Requested NUM_SHARDS={num_shards}, but only found {len(shards)} parquet shards"
        )
    return shards[:num_shards]


def download_shards(dataset_id, shard_names, download_dir):
    download_root = Path(download_dir)
    download_root.mkdir(parents=True, exist_ok=True)

    local_paths = []
    for shard_name in shard_names:
        print(f"Downloading shard: {shard_name}")
        local_path = hf_hub_download(
            repo_id=dataset_id,
            filename=shard_name,
            repo_type="dataset",
            local_dir=str(download_root),
        )
        local_paths.append(local_path)
    return local_paths


def load_climbmix_dataset(local_paths):
    dataset = load_dataset("parquet", data_files=local_paths, split="train")
    if "text" not in dataset.column_names:
        raise RuntimeError(f"Expected a 'text' column, got columns={dataset.column_names}")
    columns_to_remove = [column for column in dataset.column_names if column != "text"]
    if columns_to_remove:
        dataset = dataset.remove_columns(columns_to_remove)
    return dataset


def discover_tokenizer_dirs(tokenizer_roots):
    tokenizer_dirs = set()
    for tokenizer_root in tokenizer_roots:
        root = Path(tokenizer_root).expanduser()
        if not root.exists():
            raise FileNotFoundError(f"Tokenizer root not found: {root}")

        if (root / "tokenizer.json").is_file():
            tokenizer_dirs.add(root.resolve())

        for tokenizer_json in root.rglob("tokenizer.json"):
            tokenizer_dirs.add(tokenizer_json.parent.resolve())

    if not tokenizer_dirs:
        raise RuntimeError(f"No tokenizer.json files found under: {tokenizer_roots}")
    return sorted(tokenizer_dirs)


def get_cached_tokenizer(tokenizer_json_path):
    tokenizer = _TOKENIZER_CACHE.get(tokenizer_json_path)
    if tokenizer is None:
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_json_path)
        _TOKENIZER_CACHE[tokenizer_json_path] = tokenizer
    return tokenizer


def count_token_lengths(batch, tokenizer_json_path, add_special_tokens):
    tokenizer = get_cached_tokenizer(tokenizer_json_path)
    texts = [text if isinstance(text, str) else "" for text in batch["text"]]
    encoded = tokenizer(
        texts,
        add_special_tokens=add_special_tokens,
    )
    return {TOKEN_COUNT_COLUMN: [len(input_ids) for input_ids in encoded["input_ids"]]}


def count_tokens_for_tokenizer(tokenizer_dir, dataset):
    tokenizer_json = tokenizer_dir / "tokenizer.json"
    tokenizer_json_path = str(tokenizer_json)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_json_path)
    counted_dataset = dataset.map(
        count_token_lengths,
        batched=True,
        batch_size=BATCH_SIZE,
        num_proc=NUM_TOKENIZER_WORKERS,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
        desc=f"Tokenizing {tokenizer_dir.name}",
        fn_kwargs={
            "tokenizer_json_path": tokenizer_json_path,
            "add_special_tokens": ADD_SPECIAL_TOKENS,
        },
    )

    return {
        "path": str(tokenizer_dir),
        "vocab_size": len(tokenizer),
        "rows": len(counted_dataset),
        "tokens": sum(counted_dataset[TOKEN_COUNT_COLUMN]),
    }


def print_summary(results):
    print("")
    print("Final token counts")
    print("=" * 120)
    print(f"{'tokens':>18}  {'vocab':>8}  {'rows':>10}  tokenizer")
    print("-" * 120)
    for result in sorted(results, key=lambda item: item["path"]):
        print(
            f"{result['tokens']:>18,}  "
            f"{result['vocab_size']:>8,}  "
            f"{result['rows']:>10,}  "
            f"{result['path']}"
        )


def main():
    print(f"Listing parquet shards for {DATASET_ID}")
    selected_shards = list_first_parquet_shards(DATASET_ID, NUM_SHARDS)
    print(f"Using first {len(selected_shards)} parquet shards:")
    for shard_name in selected_shards:
        print(f"  - {shard_name}")

    local_paths = download_shards(DATASET_ID, selected_shards, DOWNLOAD_DIR)
    dataset = load_climbmix_dataset(local_paths)
    print(f"Loaded {len(dataset):,} rows")

    tokenizer_dirs = discover_tokenizer_dirs(TOKENIZER_ROOTS)
    print(f"Discovered {len(tokenizer_dirs)} tokenizers:")
    for tokenizer_dir in tokenizer_dirs:
        print(f"  - {tokenizer_dir}")

    results = []
    for tokenizer_dir in tokenizer_dirs:
        print(f"Counting tokens for: {tokenizer_dir}")
        results.append(count_tokens_for_tokenizer(tokenizer_dir, dataset))

    print_summary(results)


if __name__ == "__main__":
    main()
