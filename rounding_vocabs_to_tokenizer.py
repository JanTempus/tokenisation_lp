import os
import pickle
import re
from datetime import datetime
from pathlib import Path

from tokenizers import Regex
from tokenizers import Tokenizer
from tokenizers.models import Unigram
from tokenizers.pre_tokenizers import ByteLevel, Sequence, Split
from transformers import AutoTokenizer, PreTrainedTokenizerFast


SPECIAL_TOKEN_MAP = {
    "unk_token": "[UNK]",
    "eos_token": "[EOS]",
    "pad_token": "[PAD]",
    "cls_token": "[CLS]",
    "sep_token": "[SEP]",
    "mask_token": "[MASK]",
}

ROUNDING_SCHEMES = ("all_ones", "all_nonzero", "det", "bias", "prob")

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
        cache_dir="./pythia-70m-deduped/step3000",
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
        f"Unsupported PRETOKENIZER_MODE='{mode}'. "
        "Expected one of: pythia, split_bytelevel, custom"
    )


PRETOKENIZER = build_pretokenizer(PRETOKENIZER_MODE)

ROUND_TRIP_SAMPLES = [
    (
        "whitespace",
        "  leading space\tand tab\nmultiple   spaces\n\ntrailing space ",
    ),
    (
        "unicode",
        "naive cafe 中文 Ελληνικα العربية 😀🚀",
    ),
    (
        "code",
        "def f(x):\n    return x**2  # square\nprint(f(7))\n",
    ),
    (
        "long_text",
        "Apertus tokenizer stress test. " * 200,
    ),
]


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
    core_vocab_size = vocab_size - num_special_tokens
    if core_vocab_size <= 0:
        raise ValueError(
            f"Vocab size {vocab_size} too small for special tokens ({num_special_tokens})"
        )

    possible_tokens = tokens["possible_tokens"]

    # Rounding helpers already account for unique_chars internally.
    det_tokens = deterministic_rounding(possible_tokens, unique_chars, core_vocab_size)
    bias_tokens = biased_rounding(possible_tokens, unique_chars, core_vocab_size)
    prob_tokens = probabilistic_rounding(possible_tokens, unique_chars, core_vocab_size)
    tokens_ones = [token.token for token in possible_tokens if token.lp_value >= 0.99]
    tokens_nonzero = [token.token for token in possible_tokens if token.lp_value > 0.0]

    return {
        "all_ones": dedupe_tokens(tokens_ones + unique_chars),
        "all_nonzero": dedupe_tokens(tokens_nonzero + unique_chars),
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


def test_round_trip_samples(tokenizer):
    success = 0
    total = len(ROUND_TRIP_SAMPLES)

    for _, sample in ROUND_TRIP_SAMPLES:
        ids = tokenizer(sample, add_special_tokens=False)["input_ids"]
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        if len(ids) > 0 and decoded == sample:
            success += 1

    return success == total, success, total


def test_single_byte_strings(tokenizer, behavior="not_all_unk"):
    encodable = 0
    exact_roundtrip = 0
    all_unk_count = 0
    exceptions = 0
    total = 256
    unk_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKEN_MAP["unk_token"])

    for byte_value in range(total):
        char_as_string = bytes([byte_value]).decode("latin-1")
        try:
            ids = tokenizer(char_as_string, add_special_tokens=False)["input_ids"]
            decoded = tokenizer.decode(ids, skip_special_tokens=False)

            if len(ids) > 0:
                encodable += 1

            if ids and unk_id is not None and all(token_id == unk_id for token_id in ids):
                all_unk_count += 1

            if decoded == char_as_string:
                exact_roundtrip += 1
        except Exception:
            exceptions += 1

    if behavior == "strict_roundtrip":
        is_ok = (
            encodable == total
            and exact_roundtrip == total
            and all_unk_count == 0
            and exceptions == 0
        )
    elif behavior == "no_unk":
        is_ok = encodable == total and all_unk_count == 0 and exceptions == 0
    elif behavior == "not_all_unk":
        is_ok = encodable == total and all_unk_count < total and exceptions == 0
    else:
        raise ValueError(
            f"Invalid BYTE_TEST_BEHAVIOR='{behavior}'. "
            "Expected one of: not_all_unk, no_unk, strict_roundtrip"
        )

    return is_ok, {
        "encodable": encodable,
        "total": total,
        "exact_roundtrip": exact_roundtrip,
        "identity_fraction": exact_roundtrip / total,
        "all_unk_count": all_unk_count,
        "exceptions": exceptions,
        "behavior": behavior,
    }


def run_tokenizer_tests(tokenizer_name, tokenizer, byte_behavior):
    special_ok = test_special_tokens(tokenizer)
    text_ok = test_text_samples(tokenizer)
    roundtrip_ok, roundtrip_success, roundtrip_total = test_round_trip_samples(tokenizer)
    bytes_ok, byte_stats = test_single_byte_strings(tokenizer, behavior=byte_behavior)

    overall_ok = special_ok and text_ok and roundtrip_ok and bytes_ok
    status = "PASS" if overall_ok else "FAIL"
    print(
        f"[TEST] {tokenizer_name}: {status} | "
        f"special={special_ok} text={text_ok} roundtrip={roundtrip_success}/{roundtrip_total} "
        f"bytes_mode={byte_stats['behavior']} bytes_enc={byte_stats['encodable']}/{byte_stats['total']} "
        f"bytes_exact={byte_stats['exact_roundtrip']}/{byte_stats['total']} "
        f"bytes_identity_frac={byte_stats['identity_fraction']:.4f} "
        f"bytes_all_unk={byte_stats['all_unk_count']} bytes_exceptions={byte_stats['exceptions']}"
    )
    return overall_ok


def assert_expected_tokenizer_len(tokenizer, tokenizer_name, target_vocab_size, rounding_scheme):
    if rounding_scheme in {"all_ones", "all_nonzero"}:
        return
    actual = len(tokenizer)
    if actual != target_vocab_size:
        raise ValueError(
            f"{tokenizer_name} has len={actual}, expected {target_vocab_size}. "
            "For det/bias/prob this should match exactly."
        )


if __name__ == "__main__":
    raw_vocab_path = os.environ.get("RAW_VOCAB_PATH", "rounding_vocabs_apertus_2")
    save_dir = os.environ.get("SAVE_TOKENIZER_DIR", "rounded_tokenizers_apertus_2")
    run_tests = os.environ.get("RUN_TOKENIZER_TESTS", "1") == "1"
    byte_test_behavior = os.environ.get("BYTE_TEST_BEHAVIOR", "not_all_unk")
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    raw_files = list_raw_vocab_files(raw_vocab_path)
    print(f"Using PRETOKENIZER_MODE={PRETOKENIZER_MODE}")
    print(f"Found {len(raw_files)} raw vocab file(s) in {raw_vocab_path}")

    total_tokenizers = 0
    passed_tokenizers = 0

    for raw_file in raw_files:
        vocab_size = parse_vocab_size_from_path(raw_file)
        print(f"\nProcessing {Path(raw_file).name} (target vocab size={vocab_size})")
        vocab_output_dir = os.path.join(save_dir, f"{run_timestamp}_vocab_{vocab_size}")
        os.makedirs(vocab_output_dir, exist_ok=True)
        print(f"Saving under: {vocab_output_dir}")

        vocabs = round_vocabs(raw_file, vocab_size)
        for rnd_scheme in ROUNDING_SCHEMES:
            tokenizer = build_tokenizer(vocabs[rnd_scheme])
            tokenizer_name = f"lp_{vocab_size}_{rnd_scheme}"
            assert_expected_tokenizer_len(tokenizer, tokenizer_name, vocab_size, rnd_scheme)
            save_tokenizer(tokenizer, vocab_output_dir, vocab_size, rnd_scheme)

            total_tokenizers += 1
            if run_tests:
                if run_tokenizer_tests(tokenizer_name, tokenizer, byte_test_behavior):
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
