from lp_tokenizer.lp_tokenizer import Tokenizer
from transformers import AutoTokenizer
from datasets import concatenate_datasets, load_dataset
from collections import Counter
import pickle
import os
import json
import random
from datetime import datetime
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr


num_proc = int(os.environ.get("NUM_PROC", "16"))
batch_size = int(os.environ.get("BATCH_SIZE", "10000"))
pretokenizer = AutoTokenizer.from_pretrained(
    "EleutherAI/pythia-70m-deduped",
    revision="step3000",
)


def get_unique_chars_batch(batch):
    unique_chars = set()

    for text in batch["text"]:
        if not isinstance(text, str) or not text:
            continue
        words_with_offsets = pretokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        for token, _ in words_with_offsets:
            unique_chars.update(token)

    return {"unique_chars": [sorted(unique_chars)]}


def train_lp_tokenizer(dataset, unique_chars, vocab_size, save_dir):
    corpus_all = [text for text in dataset["text"] if isinstance(text, str) and text]

    tokenizer = Tokenizer(
        corpus=corpus_all,
        vocab_size=vocab_size,
        unique_chars=unique_chars,
        unk_token="[UNK]",
        eos_token="[EOS]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    tokens = tokenizer.make_vocab()
    file_name = os.path.join(save_dir, f"lp_tokens_{vocab_size}.pkl")
    os.makedirs(save_dir, exist_ok=True)
    with open(file_name, "wb") as f:
        pickle.dump(tokens, f)


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


def save_sampling_manifest(save_dir, source_counts, sampling_manifest, target_rows, seed):
    os.makedirs(save_dir, exist_ok=True)
    manifest_path = os.path.join(save_dir, f"sampling_manifest_seed_{seed}_rows_{target_rows}.json")
    payload = {
        "target_rows": target_rows,
        "seed": seed,
        "source_counts": source_counts,
        "draws": sampling_manifest,
    }
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    print(f"Wrote sampling manifest to {manifest_path}")


def train_with_optional_log(
    dataset,
    unique_chars,
    vocab_size,
    save_dir,
    write_log_to_file,
    log_dir,
    run_context,
):
    if not write_log_to_file:
        train_lp_tokenizer(dataset, unique_chars, vocab_size, save_dir)
        return

    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"train_lp_tokens_{vocab_size}_{timestamp}.log")

    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write("# LP Tokenizer Training Log\n")
        log_file.write(f"- timestamp: {datetime.now().isoformat()}\n")
        log_file.write(f"- vocab_size: {vocab_size}\n")
        log_file.write(f"- dataset_rows: {len(dataset)}\n")
        log_file.write(f"- unique_chars: {len(unique_chars)}\n")
        for key, value in run_context.items():
            log_file.write(f"- {key}: {value}\n")
        log_file.write("\n## Captured stdout/stderr\n\n")
        log_file.flush()

        with redirect_stdout(log_file), redirect_stderr(log_file):
            train_lp_tokenizer(dataset, unique_chars, vocab_size, save_dir)

    print(f"Wrote training log to {log_path}")


if __name__ == "__main__":
    DATASET_BASE_DIR = os.environ.get(
        "TOKENIZER_DATASET_BASE",
        "/capstor/store/cscs/swissai/a139/datasets/tokenizer_training",
    )
    SOURCE_DIRS = [
        "fineweb2_sample",
        "fineweb_sample",
        "megamath_sample",
        "infimath_sample",
        "finemath_sample",
        "starcoder_sample",
    ]
    SOURCE_TEXT_COLUMNS = {
        "fineweb2_sample": "text",
        "fineweb_sample": "text",
        "megamath_sample": "text",
        "infimath_sample": "text",
        "finemath_sample": "text",
        "starcoder_sample": "content",
    }
    TARGET_ROWS = int(os.environ.get("TARGET_ROWS", "60000"))
    SEED = int(os.environ.get("SEED", "42"))
    WRITE_TRAIN_LOG_TO_FILE = os.environ.get("WRITE_TRAIN_LOG_TO_FILE", "0") == "1"
    vocab_size = [131072]
    save_dir = "rounding_vocabs_apertus_2/"
    TRAIN_LOG_DIR = os.environ.get("TRAIN_LOG_DIR", os.path.join(save_dir, "train_logs"))

    source_to_files = discover_parquet_files(DATASET_BASE_DIR, SOURCE_DIRS)
    dataset, source_counts, sampling_manifest = sample_dataset(
        source_to_files,
        TARGET_ROWS,
        SEED,
        SOURCE_TEXT_COLUMNS,
    )
    save_sampling_manifest(save_dir, source_counts, sampling_manifest, TARGET_ROWS, SEED)

    char_chunks = dataset.map(
        get_unique_chars_batch,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc="Finding unique characters",
    )

    unique_chars = sorted(set().union(*map(set, char_chunks["unique_chars"])))

    run_context = {
        "seed": SEED,
        "target_rows": TARGET_ROWS,
        "num_proc": num_proc,
        "batch_size": batch_size,
        "save_dir": save_dir,
        "dataset_base_dir": DATASET_BASE_DIR,
    }

    for vs in vocab_size:
        train_with_optional_log(
            dataset=dataset,
            unique_chars=unique_chars,
            vocab_size=vs,
            save_dir=save_dir,
            write_log_to_file=WRITE_TRAIN_LOG_TO_FILE,
            log_dir=TRAIN_LOG_DIR,
            run_context=run_context,
        )
