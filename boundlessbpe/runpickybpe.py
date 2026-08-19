#!/usr/bin/env python3

import os
from pathlib import Path

from boundlessbpe import PickyBPE


SPECIAL_TOKENS = [
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


def read_bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name, str(int(default))).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of: 0, 1, false, true, no, yes, off, on")


def main():
    corpus_path = os.environ["PICKY_CORPUS_PATH"]
    num_lines = int(os.environ["PICKY_NUM_LINES"])
    vocab_size = int(os.environ["VOCAB_SIZE"])
    verbose = read_bool_env("PICKY_VERBOSE")
    progress_every = int(os.environ.get("PICKY_PROGRESS_EVERY", "1000"))
    if progress_every < 0:
        raise ValueError("PICKY_PROGRESS_EVERY must be non-negative")
    save_dir = Path(os.environ["SAVE_DIR"])
    output_prefix = save_dir / f"pickybpe_{vocab_size}"
    save_dir.mkdir(parents=True, exist_ok=True)

    print("[pickybpe] Configuration")
    print(f"[pickybpe] Corpus: {corpus_path}")
    print(f"[pickybpe] Documents: {num_lines:,}")
    print(f"[pickybpe] Requested vocabulary size: {vocab_size:,}")
    print(f"[pickybpe] Per-merge verbose output: {verbose}")
    print(f"[pickybpe] Progress interval: {progress_every:,} iterations")
    print(f"[pickybpe] Output prefix: {output_prefix}")

    tokenizer = PickyBPE()
    tokenizer.train(
        corpus_path,
        num_lines,
        vocab_size - len(SPECIAL_TOKENS),
        100,
        verbose=verbose,
        progress_every=progress_every,
    )
    tokenizer.register_special_tokens(SPECIAL_TOKENS)
    tokenizer.save(str(output_prefix))
    print(f"[pickybpe] Saved model: {output_prefix}.model")


if __name__ == "__main__":
    main()
