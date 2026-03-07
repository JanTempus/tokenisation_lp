import os
import pickle
import re
from pathlib import Path

from lp_tokenizer.datastructures import possibleToken


def find_chunk_vocab_files(chunk_vocab_dir, vocab_size):
    chunk_vocab_dir = Path(chunk_vocab_dir)
    if not chunk_vocab_dir.exists():
        raise FileNotFoundError(f"Chunk vocab directory not found: {chunk_vocab_dir}")

    pattern = f"lp_tokens_{vocab_size}.pkl"
    files = sorted(chunk_vocab_dir.rglob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found under {chunk_vocab_dir}"
        )
    return [str(f) for f in files]


def merge_vocab_files(pkl_files, vocab_size):
    merged_tokens = {}   # token_str -> lp_value (summed)
    merged_counts = {}   # token_str -> token_instance_count (summed)
    all_unique_chars = set()
    all_special_tokens = []
    seen_special = set()

    for pkl_path in pkl_files:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        if "possible_tokens" not in data:
            raise KeyError(f"'possible_tokens' missing in {pkl_path}")
        if "unique_chars" not in data:
            raise KeyError(f"'unique_chars' missing in {pkl_path}")

        for pt in data["possible_tokens"]:
            token_str = pt.token
            merged_tokens[token_str] = merged_tokens.get(token_str, 0.0) + pt.lp_value
            merged_counts[token_str] = merged_counts.get(token_str, 0) + pt.token_instance_count

        all_unique_chars.update(data["unique_chars"])

        for st in data.get("special_tokens", []):
            if st not in seen_special:
                all_special_tokens.append(st)
                seen_special.add(st)

    # Rebuild possibleToken list, assigning sequential indices
    possible_tokens = []
    for idx, (token_str, lp_val) in enumerate(merged_tokens.items()):
        pt = possibleToken(
            token=token_str,
            lp_value=lp_val,
            instance_count=merged_counts[token_str],
            index=idx,
        )
        possible_tokens.append(pt)

    unique_chars = sorted(all_unique_chars)

    print(
        f"  vocab_size={vocab_size}: merged {len(pkl_files)} chunks, "
        f"{len(possible_tokens)} unique tokens, {len(unique_chars)} unique chars"
    )

    return {
        "possible_tokens": possible_tokens,
        "unique_chars": unique_chars,
        "special_tokens": all_special_tokens,
    }


if __name__ == "__main__":
    CHUNK_VOCAB_DIR = os.environ.get("CHUNK_VOCAB_DIR", "chunk_vocabs")
    RAW_VOCAB_PATH = os.environ.get("RAW_VOCAB_PATH", "rounding_vocabs_apertus_2")
    VOCAB_SIZES = [
        int(v.strip())
        for v in os.environ.get("VOCAB_SIZES", "131072").split(",")
        if v.strip()
    ]

    os.makedirs(RAW_VOCAB_PATH, exist_ok=True)

    for vocab_size in VOCAB_SIZES:
        print(f"Merging vocab size {vocab_size}...")
        pkl_files = find_chunk_vocab_files(CHUNK_VOCAB_DIR, vocab_size)
        print(f"  Found {len(pkl_files)} chunk pkl files")

        merged = merge_vocab_files(pkl_files, vocab_size)

        out_path = os.path.join(RAW_VOCAB_PATH, f"lp_tokens_{vocab_size}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(merged, f)
        print(f"  Saved merged vocab to {out_path}")

    print("merge_vocabs finished")
