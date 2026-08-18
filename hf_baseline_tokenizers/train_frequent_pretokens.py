#!/usr/bin/env python3
"""Train equal-score Unigram tokenizers from the most frequent pretokens.

Configuration is supplied through the same environment variables as the other
tokenizer runners:

    TRAIN_DATASET_PATH=/path/to/dataset \
    VOCAB_SIZES=8192,16384,32768 \
    SAVE_DIR=/path/to/output \
    PRETOKENIZER_MODE=nanochat \
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


APERTUS_SPECIAL_TOKENS = ["[UNK]", "[EOS]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
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

SPECIAL_TOKENS_BY_MODE = {
    "pythia": APERTUS_SPECIAL_TOKENS,
    "split_bytelevel": APERTUS_SPECIAL_TOKENS,
    "apertus": APERTUS_SPECIAL_TOKENS,
    "nanochat": NANOCHAT_SPECIAL_TOKENS,
}


def parse_vocab_sizes(raw_value: str) -> list[int]:
    """Parse, deduplicate, and sort a comma-separated vocabulary-size list."""
    return sorted({int(value) for value in raw_value.split(",")})


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
    special_tokens: Sequence[str],
    byte_alphabet: Sequence[str],
) -> list[str]:
    """Select an exact-size vocabulary with required entries and top pretokens."""
    reserved_tokens = [*special_tokens, *byte_alphabet]
    reserved_set = set(reserved_tokens)
    corpus_tokens = [
        pretoken
        for pretoken, _ in ranked_pretokens
        if pretoken not in reserved_set
    ]
    return reserved_tokens + corpus_tokens[: vocab_size - len(reserved_tokens)]


def tokenizer_kwargs_for_mode(mode: str) -> dict[str, object]:
    """Return the special-token roles used by the existing LP exporter."""
    if mode == "nanochat":
        return {
            "bos_token": "<|bos|>",
            "unk_token": "<|unk|>",
            "additional_special_tokens": NANOCHAT_SPECIAL_TOKENS[1:-1],
        }
    return {
        "unk_token": "[UNK]",
        "eos_token": "[EOS]",
        "pad_token": "[PAD]",
        "cls_token": "[CLS]",
        "sep_token": "[SEP]",
        "mask_token": "[MASK]",
    }


def build_equal_score_tokenizer(
    vocab_tokens: Sequence[str],
    backend_pretokenizer,
    backend_decoder,
    tokenizer_kwargs: dict[str, object],
) -> PreTrainedTokenizerFast:
    """Build a Hugging Face-compatible Unigram tokenizer with score -1."""
    vocab_tokens = list(vocab_tokens)
    unk_token = tokenizer_kwargs["unk_token"]
    unk_id = vocab_tokens.index(unk_token)

    unigram_vocab = [(token, -1.0) for token in vocab_tokens]
    backend = BackendTokenizer(Unigram(unigram_vocab, unk_id=unk_id))
    backend.pre_tokenizer = backend_pretokenizer
    backend.decoder = backend_decoder

    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        **tokenizer_kwargs,
    )


def train_frequency_baselines(
    dataset,
    vocab_sizes: Sequence[int],
    save_dir: Path,
    mode: str,
    pipeline_pretokenizer,
    lp_tokenizer_class,
    byte_alphabet: Sequence[str],
) -> None:
    """Collect frequencies once, then build every requested tokenizer."""
    special_tokens = SPECIAL_TOKENS_BY_MODE[mode]
    tokenizer_kwargs = tokenizer_kwargs_for_mode(mode)

    frequency_tokenizer = lp_tokenizer_class(
        corpus=dataset,
        vocab_size=max(vocab_sizes),
        special_tokens=special_tokens,
        unique_chars=byte_alphabet,
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
        if (output_dir / "tokenizer.json").is_file():
            print(f"Skipping existing tokenizer: {output_dir}")
            continue

        vocab_tokens = select_vocabulary(
            ranked_pretokens,
            vocab_size,
            special_tokens,
            byte_alphabet,
        )
        tokenizer = build_equal_score_tokenizer(
            vocab_tokens,
            backend.pre_tokenizer,
            backend.decoder,
            tokenizer_kwargs,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(output_dir)
        print(f"Saved tokenizer: {output_dir} (len={len(tokenizer)})")


def main() -> None:
    mode = os.environ.get("PRETOKENIZER_MODE", "nanochat")
    dataset_path = os.environ["TRAIN_DATASET_PATH"]
    vocab_sizes = parse_vocab_sizes(os.environ["VOCAB_SIZES"])
    save_dir = Path(os.environ["SAVE_DIR"])
    pending_sizes = [
        size
        for size in vocab_sizes
        if not (
            save_dir / f"frequent_pretoken_{size}" / "tokenizer.json"
        ).is_file()
    ]
    if not pending_sizes:
        print(f"All requested tokenizers already exist under {save_dir}")
        return

    os.environ.setdefault("PRETOKENIZER_MODE", mode)
    import train_tokenizer as training_pipeline
    from lp_tokenizer.lp_tokenizer import BYTE_LEVEL_ALPHABET, Tokenizer as LPTokenizer

    print(f"Using PRETOKENIZER_MODE={mode}")
    print(f"Loading training dataset from {dataset_path}")
    dataset = training_pipeline.load_training_dataset(dataset_path)
    print(f"Loaded {len(dataset):,} rows")

    train_frequency_baselines(
        dataset=dataset,
        vocab_sizes=pending_sizes,
        save_dir=save_dir,
        mode=mode,
        pipeline_pretokenizer=training_pipeline.pretokenizer,
        lp_tokenizer_class=LPTokenizer,
        byte_alphabet=BYTE_LEVEL_ALPHABET,
    )


if __name__ == "__main__":
    main()
