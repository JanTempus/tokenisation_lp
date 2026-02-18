from lp_tokenizer.lp_tokenizer import Tokenizer
from transformers import AutoTokenizer
from datasets import load_from_disk
from tokenizers import Regex
from tokenizers.pre_tokenizers import ByteLevel, Sequence, Split
import pickle
import os


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


if __name__ == "__main__":
    TRAIN_DATASET_PATH = os.environ.get(
        "TRAIN_DATASET_PATH",
        "/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/tokenizer_training_dataset",
    )
    vocab_size = [int(size) for size in os.environ.get("VOCAB_SIZES", "131072").split(",") if size.strip()]
    save_dir = os.environ.get("RAW_VOCAB_PATH", "rounding_vocabs_apertus_2/")
    print(f"Using PRETOKENIZER_MODE={PRETOKENIZER_MODE}")
    print(f"Loading training dataset from {TRAIN_DATASET_PATH}")

    dataset_obj = load_from_disk(TRAIN_DATASET_PATH)
    if hasattr(dataset_obj, "keys"):
        if "train" in dataset_obj:
            dataset = dataset_obj["train"]
        else:
            raise ValueError(
                f"DatasetDict at {TRAIN_DATASET_PATH} does not contain a 'train' split. "
                f"Available splits: {list(dataset_obj.keys())}"
            )
    else:
        dataset = dataset_obj

    if "text" not in dataset.column_names:
        raise ValueError(
            f"Dataset at {TRAIN_DATASET_PATH} must include a 'text' column. "
            f"Columns are: {dataset.column_names}"
        )
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
