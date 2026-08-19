#!/usr/bin/env python3
"""Train equal-score Unigram tokenizers from the most frequent pretokens.

Configuration is supplied through the same environment variables as the other
tokenizer runners:

    TRAIN_DATASET_PATH=/path/to/dataset \
    VOCAB_SIZES=8192,16384,32768 \
    SAVE_DIR=/path/to/output \
    python -u hf_baseline_tokenizers/train_frequent_pretokens.py

``NUM_PROC`` controls the Hugging Face dataset worker count and ``BATCH_SIZE``
controls how many documents each worker processes at a time.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Sequence as DatasetSequence, Value, load_from_disk
from tokenizers import Regex
from tokenizers import Tokenizer as BackendTokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import Unigram
from tokenizers.pre_tokenizers import ByteLevel, Sequence as PretokenizerSequence, Split
from transformers import PreTrainedTokenizerFast


BYTE_LEVEL_ALPHABET = sorted(ByteLevel.alphabet())
NANOCHAT_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


NANOCHAT_SPECIAL_TOKENS = [
    "<|bos|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|output_start|>",
    "<|output_end|>",
    "<|unk|>",
]

TOKENIZER_KWARGS = {
    "bos_token": "<|bos|>",
    "unk_token": "<|unk|>",
    "additional_special_tokens": NANOCHAT_SPECIAL_TOKENS[1:-1],
}


def rank_pretokens(
    pretokens: Sequence[str], frequencies: Sequence[int]
) -> list[tuple[str, int]]:
    """Rank pretokens by descending frequency, then token text."""
    return sorted(zip(pretokens, frequencies), key=lambda item: (-item[1], item[0]))


def build_nanochat_components():
    """Build the NanoChat pretokenizer and matching byte-level decoder."""
    backend_pretokenizer = PretokenizerSequence(
        [
            Split(
                pattern=Regex(NANOCHAT_SPLIT_PATTERN),
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
    return backend_pretokenizer, ByteLevelDecoder()


def _pretokenize_batch(batch, backend_pretokenizer):
    word_freqs = defaultdict(int)
    for text in batch["text"]:
        if not isinstance(text, str) or not text:
            continue
        for word, _ in backend_pretokenizer.pre_tokenize_str(text):
            if word:
                word_freqs[word] += 1
    return {
        "tokens": [list(word_freqs)],
        "frequencies": [list(word_freqs.values())],
    }


def collect_ranked_pretokens(
    dataset: Dataset,
    backend_pretokenizer,
) -> list[tuple[str, int]]:
    """Pretokenize in worker processes, merge counts, and rank by frequency."""
    batch_size = int(os.environ.get("BATCH_SIZE", "1000"))
    requested_num_proc = int(os.environ.get("NUM_PROC", "1"))
    if batch_size < 1:
        raise ValueError("BATCH_SIZE must be at least 1")
    if requested_num_proc < 1:
        raise ValueError("NUM_PROC must be at least 1")
    num_proc = min(requested_num_proc, max(1, len(dataset)))
    load_from_cache_file = os.environ.get("HF_MAP_LOAD_FROM_CACHE", "0") == "1"

    print(
        f"[pretokenize] Starting worker map: rows={len(dataset):,}, "
        f"num_proc={num_proc}, batch_size={batch_size:,}, "
        f"load_from_cache_file={load_from_cache_file}"
    )
    map_start = time.perf_counter()
    aggregates = dataset.map(
        _pretokenize_batch,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        load_from_cache_file=load_from_cache_file,
        fn_kwargs={"backend_pretokenizer": backend_pretokenizer},
        remove_columns=dataset.column_names,
        features=Features(
            {
                "tokens": DatasetSequence(Value("string")),
                "frequencies": DatasetSequence(Value("int64")),
            }
        ),
        desc="Pretokenizing corpus",
    )
    print(
        f"[pretokenize] Worker map finished in "
        f"{time.perf_counter() - map_start:.1f}s"
    )

    print("[pretokenize] Merging partial frequency tables (single process)")
    word_freqs = defaultdict(int)
    for tokens, frequencies in zip(
        aggregates["tokens"], aggregates["frequencies"]
    ):
        for word, frequency in zip(tokens, frequencies):
            word_freqs[word] += frequency
    return rank_pretokens(list(word_freqs), list(word_freqs.values()))


def load_local_dataset(path: str) -> Dataset:
    """Load a Dataset or the train split of a DatasetDict from disk."""
    dataset = load_from_disk(path)
    if isinstance(dataset, DatasetDict):
        if "train" not in dataset:
            raise ValueError(
                "The dataset has no 'train' split. "
                f"Available splits: {list(dataset.keys())}"
            )
        dataset = dataset["train"]
    if "text" not in dataset.column_names:
        raise ValueError(
            f"The dataset must have a 'text' column; found: {dataset.column_names}"
        )
    return dataset.select_columns(["text"])


def select_vocabulary(
    ranked_pretokens: Sequence[tuple[str, int]],
    vocab_size: int,
    byte_alphabet: Sequence[str],
) -> list[str]:
    """Select an exact-size vocabulary with required entries and top pretokens."""
    reserved_tokens = [*NANOCHAT_SPECIAL_TOKENS, *byte_alphabet]
    reserved_set = set(reserved_tokens)
    corpus_tokens = [
        pretoken
        for pretoken, _ in ranked_pretokens
        if pretoken not in reserved_set
    ]
    return reserved_tokens + corpus_tokens[: vocab_size - len(reserved_tokens)]


def build_equal_score_tokenizer(
    vocab_tokens: Sequence[str],
    backend_pretokenizer,
    backend_decoder,
) -> PreTrainedTokenizerFast:
    """Build a Hugging Face-compatible Unigram tokenizer with score -1."""
    vocab_tokens = list(vocab_tokens)
    unk_id = vocab_tokens.index("<|unk|>")

    unigram_vocab = [(token, -1.0) for token in vocab_tokens]
    backend = BackendTokenizer(Unigram(unigram_vocab, unk_id=unk_id))
    backend.pre_tokenizer = backend_pretokenizer
    backend.decoder = backend_decoder

    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        **TOKENIZER_KWARGS,
    )


def train_frequency_baselines(
    dataset: Dataset,
    vocab_sizes: Sequence[int],
    save_dir: Path,
    backend_pretokenizer,
    backend_decoder,
) -> None:
    """Collect frequencies once, then build every requested tokenizer."""
    ranked_pretokens = collect_ranked_pretokens(dataset, backend_pretokenizer)
    print(f"Collected and ranked {len(ranked_pretokens):,} distinct pretokens")
    for token, frequency in ranked_pretokens[:10]:
        print(f"  {frequency:>12,}  {token!r}")

    save_dir.mkdir(parents=True, exist_ok=True)
    for vocab_size in vocab_sizes:
        output_dir = save_dir / f"frequent_pretoken_{vocab_size}"
        vocab_tokens = select_vocabulary(
            ranked_pretokens,
            vocab_size,
            BYTE_LEVEL_ALPHABET,
        )
        tokenizer = build_equal_score_tokenizer(
            vocab_tokens,
            backend_pretokenizer,
            backend_decoder,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(output_dir)
        print(f"Saved tokenizer: {output_dir} (len={len(tokenizer)})")


def main() -> None:
    dataset_path = os.environ["TRAIN_DATASET_PATH"]
    vocab_sizes = [int(value) for value in os.environ["VOCAB_SIZES"].split(",")]
    save_dir = Path(os.environ["SAVE_DIR"])

    print(f"Loading training dataset from {dataset_path}")
    dataset = load_local_dataset(dataset_path)
    print(f"Loaded {len(dataset):,} rows")
    backend_pretokenizer, backend_decoder = build_nanochat_components()

    train_frequency_baselines(
        dataset=dataset,
        vocab_sizes=vocab_sizes,
        save_dir=save_dir,
        backend_pretokenizer=backend_pretokenizer,
        backend_decoder=backend_decoder,
    )


if __name__ == "__main__":
    main()
