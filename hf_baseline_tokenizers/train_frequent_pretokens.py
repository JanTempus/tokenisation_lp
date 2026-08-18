#!/usr/bin/env python3
"""Train equal-score Unigram tokenizers from the most frequent pretokens.

Configuration is supplied through the same environment variables as the other
tokenizer runners:

    TRAIN_DATASET_PATH=/path/to/dataset \
    VOCAB_SIZES=8192,16384,32768 \
    SAVE_DIR=/path/to/output \
    python -u hf_baseline_tokenizers/train_frequent_pretokens.py

``NUM_PROC`` and ``BATCH_SIZE`` are consumed by
``lp_tokenizer.Tokenizer.pretokenize_and_prepare_corpus``.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

from tokenizers import Tokenizer as BackendTokenizer
from tokenizers.models import Unigram
from transformers import PreTrainedTokenizerFast
from lp_tokenizer.lp_tokenizer import BYTE_LEVEL_ALPHABET, Tokenizer as LPTokenizer


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


def collect_ranked_pretokens(lp_tokenizer) -> list[tuple[str, int]]:
    """Run the LP frequency collector exactly once and return its ranking."""
    pretoken_dataset = lp_tokenizer.pretokenize_and_prepare_corpus(lp_tokenizer.corpus)
    return rank_pretokens(
        pretoken_dataset["pretoken"],
        pretoken_dataset["frequency"],
    )


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
    dataset,
    vocab_sizes: Sequence[int],
    save_dir: Path,
    pipeline_pretokenizer,
) -> None:
    """Collect frequencies once, then build every requested tokenizer."""
    frequency_tokenizer = LPTokenizer(
        corpus=dataset,
        vocab_size=max(vocab_sizes),
        special_tokens=NANOCHAT_SPECIAL_TOKENS,
        unique_chars=BYTE_LEVEL_ALPHABET,
        pretokenizer=pipeline_pretokenizer,
    )
    ranked_pretokens = collect_ranked_pretokens(frequency_tokenizer)
    print(f"Collected and ranked {len(ranked_pretokens):,} distinct pretokens")
    for token, frequency in ranked_pretokens[:10]:
        print(f"  {frequency:>12,}  {token!r}")

    backend = pipeline_pretokenizer.backend_tokenizer
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
            backend.pre_tokenizer,
            backend.decoder,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(output_dir)
        print(f"Saved tokenizer: {output_dir} (len={len(tokenizer)})")


def main() -> None:
    dataset_path = os.environ["TRAIN_DATASET_PATH"]
    vocab_sizes = [int(value) for value in os.environ["VOCAB_SIZES"].split(",")]
    save_dir = Path(os.environ["SAVE_DIR"])

    os.environ["PRETOKENIZER_MODE"] = "nanochat"
    import train_tokenizer as training_pipeline

    print(f"Loading training dataset from {dataset_path}")
    dataset = training_pipeline.load_training_dataset(dataset_path)
    print(f"Loaded {len(dataset):,} rows")

    train_frequency_baselines(
        dataset=dataset,
        vocab_sizes=vocab_sizes,
        save_dir=save_dir,
        pipeline_pretokenizer=training_pipeline.pretokenizer,
    )


if __name__ == "__main__":
    main()
