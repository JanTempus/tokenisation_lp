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


def main():
    corpus_path = os.environ["PICKY_CORPUS_PATH"]
    num_lines = int(os.environ["PICKY_NUM_LINES"])
    vocab_size = int(os.environ["VOCAB_SIZE"])
    save_dir = Path(os.environ["SAVE_DIR"])
    output_prefix = save_dir / f"pickybpe_{vocab_size}"
    save_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = PickyBPE()
    tokenizer.train(
        corpus_path,
        num_lines,
        vocab_size - len(SPECIAL_TOKENS),
        100,
        verbose=False,
    )
    tokenizer.register_special_tokens(SPECIAL_TOKENS)
    tokenizer.save(str(output_prefix))


if __name__ == "__main__":
    main()
