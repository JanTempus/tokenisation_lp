from lp_tokenizer.lp_tokenizer import Tokenizer
from transformers import AutoTokenizer
from datasets import concatenate_datasets, load_dataset, load_from_disk
from tokenizers import Regex
from tokenizers.pre_tokenizers import ByteLevel, Sequence, Split
import pickle
import os
from pathlib import Path


num_proc = int(os.environ.get("NUM_PROC", "16"))
batch_size = int(os.environ.get("BATCH_SIZE", "10000"))
PRETOKENIZER_MODE = os.environ.get("PRETOKENIZER_MODE", "pythia").strip().lower()
_CUSTOM_SPLIT_PATTERN = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n/]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


def build_pretokenizer(mode):
    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-70m-deduped",
        revision="step3000",
    )

    if mode == "pythia":
        return tokenizer

    if mode in {"split_bytelevel", "custom"}:
        tokenizer.backend_tokenizer.pre_tokenizer = Sequence(
            [
                Split(
                    pattern=Regex(_CUSTOM_SPLIT_PATTERN),
                    behavior="isolated",
                    invert=False,
                ),
                ByteLevel(
                    add_prefix_space=False,
                    trim_offsets=True,
                    use_regex=False,
                ),
            ]
        )
        return tokenizer

    raise ValueError(
        f"Unsupported PRETOKENIZER_MODE='{mode}'. Expected one of: pythia, split_bytelevel, custom"
    )


pretokenizer = build_pretokenizer(PRETOKENIZER_MODE)


def get_unique_chars_batch(batch):
    unique_chars = set()

    for text in batch["text"]:
        if not isinstance(text, str) or not text:
            continue
        words_with_offsets = pretokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        for token, _ in words_with_offsets:
            unique_chars.update(token)

    return {"unique_chars": [sorted(unique_chars)]}


def train_lp_tokenizer(dataset, unique_chars, vocab_size, save_dir, pretokenizer_obj):
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
        pretokenizer=pretokenizer_obj,
    )
    tokens = tokenizer.make_vocab()
    file_name = os.path.join(save_dir, f"lp_tokens_{vocab_size}.pkl")
    os.makedirs(save_dir, exist_ok=True)
    with open(file_name, "wb") as f:
        pickle.dump(tokens, f)


def infer_text_column(dataset):
    preferred_columns = ("text", "content", "code")
    for column in preferred_columns:
        if column in dataset.column_names:
            return column

    for name, feature in dataset.features.items():
        dtype = getattr(feature, "dtype", None)
        if dtype in {"string", "large_string"}:
            return name

    raise ValueError(f"Could not infer text column from columns: {dataset.column_names}")


def normalize_to_text_column(dataset):
    text_column = infer_text_column(dataset)
    if text_column != "text":
        dataset = dataset.rename_column(text_column, "text")
    columns_to_remove = [column for column in dataset.column_names if column != "text"]
    if columns_to_remove:
        dataset = dataset.remove_columns(columns_to_remove)
    return dataset


def load_training_dataset(path):
    try:
        dataset_obj = load_from_disk(path)
        if hasattr(dataset_obj, "keys"):
            if "train" in dataset_obj:
                dataset = dataset_obj["train"]
            else:
                raise ValueError(
                    f"DatasetDict at {path} does not contain a 'train' split. "
                    f"Available splits: {list(dataset_obj.keys())}"
                )
        else:
            dataset = dataset_obj

        print("Loaded dataset using load_from_disk")
        return normalize_to_text_column(dataset)
    except Exception as load_from_disk_error:
        base_path = Path(path)
        if not base_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {path}") from load_from_disk_error

        source_dirs = sorted(entry for entry in base_path.iterdir() if entry.is_dir())
        source_datasets = []

        for source_dir in source_dirs:
            parquet_files = sorted(str(parquet_path) for parquet_path in source_dir.rglob("*.parquet"))
            if not parquet_files:
                continue

            source_dataset = load_dataset("parquet", data_files=parquet_files, split="train")
            source_dataset = normalize_to_text_column(source_dataset)
            source_datasets.append(source_dataset)
            print(
                f"Loaded source '{source_dir.name}' via parquet "
                f"({len(parquet_files)} files, {len(source_dataset)} rows)"
            )

        if source_datasets:
            return concatenate_datasets(source_datasets)

        parquet_files = sorted(str(parquet_path) for parquet_path in base_path.rglob("*.parquet"))
        if not parquet_files:
            raise RuntimeError(
                f"Failed to load as Dataset/DatasetDict and found no parquet files under: {path}"
            ) from load_from_disk_error

        print(f"load_from_disk failed; falling back to parquet load ({len(parquet_files)} files)")
        dataset = load_dataset("parquet", data_files=parquet_files, split="train")
        return normalize_to_text_column(dataset)


if __name__ == "__main__":
    TRAIN_DATASET_PATH = os.environ.get(
        "TRAIN_DATASET_PATH",
        "/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/tokenizer_training_dataset",
    )
    vocab_size = [int(size) for size in os.environ.get("VOCAB_SIZES", "131072").split(",") if size.strip()]
    save_dir = os.environ.get("RAW_VOCAB_PATH", "rounding_vocabs_apertus_2/")
    print(f"Using PRETOKENIZER_MODE={PRETOKENIZER_MODE}")
    print(f"Loading training dataset from {TRAIN_DATASET_PATH}")

    dataset = load_training_dataset(TRAIN_DATASET_PATH)
    print(f"Loaded {len(dataset)} rows")

    char_chunks = dataset.map(
        get_unique_chars_batch,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc="Finding unique characters",
    )

    unique_chars = sorted(set().union(*map(set, char_chunks["unique_chars"])))

    for vs in vocab_size:
        train_lp_tokenizer(dataset, unique_chars, vs, save_dir, pretokenizer)
