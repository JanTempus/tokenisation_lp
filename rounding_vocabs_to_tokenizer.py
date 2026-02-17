import os
import pickle
import re
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import Unigram
from transformers import AutoTokenizer, PreTrainedTokenizerFast


SPECIAL_TOKEN_MAP = {
    "unk_token": "[UNK]",
    "eos_token": "[EOS]",
    "pad_token": "[PAD]",
    "cls_token": "[CLS]",
    "sep_token": "[SEP]",
    "mask_token": "[MASK]",
}

ROUNDING_SCHEMES = ("all_ones", "det", "bias", "prob")

PRETOKENIZER = AutoTokenizer.from_pretrained(
    "EleutherAI/pythia-70m-deduped",
    revision="step3000",
    cache_dir="./pythia-70m-deduped/step3000",
)


def dedupe_tokens(tokens):
    seen = set()
    out = []
    for token in tokens:
        if not isinstance(token, str):
            continue
        if token == "":
            continue
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def parse_vocab_size_from_path(path):
    match = re.search(r"lp_tokens_(\d+)\.pkl$", Path(path).name)
    if not match:
        raise ValueError(f"Could not infer vocab size from file name: {path}")
    return int(match.group(1))


def list_raw_vocab_files(raw_vocab_dir):
    raw_dir = Path(raw_vocab_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw vocab directory not found: {raw_vocab_dir}")

    files = sorted(raw_dir.glob("lp_tokens_*.pkl"))
    if not files:
        files = sorted(raw_dir.glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"No .pkl files found in: {raw_vocab_dir}")
    return [str(path) for path in files]


def include_special_tokens(vocab_tokens):
    special_tokens = list(SPECIAL_TOKEN_MAP.values())
    return dedupe_tokens(special_tokens + vocab_tokens)


def build_tokenizer(vocab_tokens):
    all_tokens = include_special_tokens(vocab_tokens)
    unk_token = SPECIAL_TOKEN_MAP["unk_token"]
    unk_id = all_tokens.index(unk_token)

    unigram_vocab = [(token, -1.0) for token in all_tokens]
    tokenizer = Tokenizer(Unigram(unigram_vocab, unk_id=unk_id))

    # Keep pretokenization consistent with training.
    tokenizer.pre_tokenizer = PRETOKENIZER.backend_tokenizer.pre_tokenizer
    tokenizer.decoder = PRETOKENIZER.backend_tokenizer.decoder

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        **SPECIAL_TOKEN_MAP,
    )
    return fast_tokenizer


def save_tokenizer(tokenizer, save_dir, target_vocab_size, rnd_scheme):
    save_path = os.path.join(save_dir, f"lp_{target_vocab_size}_{rnd_scheme}")
    os.makedirs(save_path, exist_ok=True)
    tokenizer.save_pretrained(save_path)
    print(f"Saved tokenizer: {save_path} (len={len(tokenizer)})")
    return save_path


def round_vocabs(raw_tokens_path, vocab_size):
    from lp_tokenizer.lp_functions import biased_rounding, deterministic_rounding, probabilistic_rounding

    with open(raw_tokens_path, "rb") as file:
        tokens = pickle.load(file)

    if "possible_tokens" not in tokens:
        raise KeyError(f"'possible_tokens' missing in {raw_tokens_path}")
    if "unique_chars" not in tokens:
        raise KeyError(f"'unique_chars' missing in {raw_tokens_path}")

    unique_chars = dedupe_tokens(tokens["unique_chars"])
    num_special_tokens = len(SPECIAL_TOKEN_MAP)
    lp_budget = vocab_size - len(unique_chars) - num_special_tokens
    if lp_budget <= 0:
        raise ValueError(
            f"Vocab size {vocab_size} too small for unique chars ({len(unique_chars)}) "
            f"+ special tokens ({num_special_tokens})"
        )

    possible_tokens = tokens["possible_tokens"]

    det_tokens = deterministic_rounding(possible_tokens, unique_chars, lp_budget)
    bias_tokens = biased_rounding(possible_tokens, unique_chars, lp_budget)
    prob_tokens = probabilistic_rounding(possible_tokens, unique_chars, lp_budget)
    tokens_ones = [token.token for token in possible_tokens if token.lp_value >= 0.99]

    return {
        "all_ones": dedupe_tokens(tokens_ones + unique_chars),
        "det": dedupe_tokens(det_tokens + unique_chars),
        "bias": dedupe_tokens(bias_tokens + unique_chars),
        "prob": dedupe_tokens(prob_tokens + unique_chars),
    }


def test_special_tokens(tokenizer):
    for field, token in SPECIAL_TOKEN_MAP.items():
        configured = getattr(tokenizer, field, None)
        if configured != token:
            return False

        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id < 0:
            return False
        if tokenizer.convert_ids_to_tokens(token_id) != token:
            return False

        ids = tokenizer(token, add_special_tokens=False)["input_ids"]
        if len(ids) != 1 or ids[0] != token_id:
            return False

    return True


def test_text_samples(tokenizer):
    samples = [
        "hello world",
        "Apertus tokenizer test.",
        "print('hello')",
        "x + y = z",
    ]
    for sample in samples:
        ids = tokenizer(sample, add_special_tokens=True)["input_ids"]
        if len(ids) == 0:
            return False
    return True


def test_single_byte_strings(tokenizer):
    success = 0
    total = 256

    for byte_value in range(total):
        char_as_string = bytes([byte_value]).decode("latin-1")
        try:
            ids = tokenizer(char_as_string, add_special_tokens=False)["input_ids"]
            if len(ids) > 0:
                success += 1
        except Exception:
            pass

    return success == total, success, total


def run_tokenizer_tests(tokenizer_name, tokenizer):
    special_ok = test_special_tokens(tokenizer)
    text_ok = test_text_samples(tokenizer)
    bytes_ok, byte_success, byte_total = test_single_byte_strings(tokenizer)

    overall_ok = special_ok and text_ok and bytes_ok
    status = "PASS" if overall_ok else "FAIL"
    print(
        f"[TEST] {tokenizer_name}: {status} | "
        f"special={special_ok} text={text_ok} byte_chars={byte_success}/{byte_total}"
    )
    return overall_ok


if __name__ == "__main__":
    raw_vocab_path = os.environ.get("RAW_VOCAB_PATH", "rounding_vocabs_apertus_2")
    save_dir = os.environ.get("SAVE_TOKENIZER_DIR", "rounded_tokenizers_apertus_2")
    run_tests = os.environ.get("RUN_TOKENIZER_TESTS", "1") == "1"

    raw_files = list_raw_vocab_files(raw_vocab_path)
    print(f"Found {len(raw_files)} raw vocab file(s) in {raw_vocab_path}")

    total_tokenizers = 0
    passed_tokenizers = 0

    for raw_file in raw_files:
        vocab_size = parse_vocab_size_from_path(raw_file)
        print(f"\nProcessing {Path(raw_file).name} (target vocab size={vocab_size})")

        vocabs = round_vocabs(raw_file, vocab_size)
        for rnd_scheme in ROUNDING_SCHEMES:
            tokenizer = build_tokenizer(vocabs[rnd_scheme])
            tokenizer_name = f"lp_{vocab_size}_{rnd_scheme}"
            save_tokenizer(tokenizer, save_dir, vocab_size, rnd_scheme)

            total_tokenizers += 1
            if run_tests:
                if run_tokenizer_tests(tokenizer_name, tokenizer):
                    passed_tokenizers += 1
            else:
                passed_tokenizers += 1

    final_ok = passed_tokenizers == total_tokenizers
    final_status = "PASS" if final_ok else "FAIL"
    print(
        f"\n[SUMMARY] {final_status}: tokenizers_passed={passed_tokenizers}/{total_tokenizers}"
    )

    if not final_ok:
        raise SystemExit(1)
