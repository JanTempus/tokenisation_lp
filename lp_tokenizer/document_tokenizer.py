"""Lossless document tokenization shared by LP training and inference.

The custom format deliberately is not named tokenizer.json: a plain Unigram
backend cannot enforce these context-dependent token boundaries.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import json
from pathlib import Path
import re

from tokenizers import Tokenizer as BackendTokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel


FORMAT_VERSION = 1
TOKENIZER_FILENAME = "document_lp_tokenizer.json"
BYTE_ALPHABET = sorted(ByteLevel.alphabet())


def _byte_mapping():
    visible = [*range(33, 127), *range(161, 173), *range(174, 256)]
    mapping = {byte: chr(byte) for byte in visible}
    for byte in range(256):
        if byte not in mapping:
            mapping[byte] = chr(256 + len(mapping) - len(visible))
    return mapping


BYTE_ENCODER = _byte_mapping()
BYTE_DECODER = {symbol: byte for byte, symbol in BYTE_ENCODER.items()}


def mode_directory(path, mode):
    directory = Path(path)
    return directory if mode == "standard" or directory.name == mode else directory / mode


def encode_bytes(text):
    return "".join(BYTE_ENCODER[byte] for byte in text.encode("utf-8"))


@dataclass(frozen=True)
class TrainingOptions:
    training_mode: str = "standard"
    max_token_bytes: int | None = None
    super_base_vocab_size: int | None = None

    def __post_init__(self):
        if self.training_mode not in {"standard", "boundless", "super"}:
            raise ValueError("LP_TRAINING_MODE must be standard, boundless, or super")
        cap = self.max_token_bytes
        if cap is None and self.training_mode != "standard":
            object.__setattr__(self, "max_token_bytes", 64)
        elif cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or cap < 0):
            raise ValueError("LP_MAX_TOKEN_BYTES must be a non-negative integer (0 means unlimited)")
        base = self.super_base_vocab_size
        if base is not None:
            if self.training_mode != "super":
                raise ValueError("LP_SUPER_BASE_VOCAB_SIZE applies only to super mode")
            if isinstance(base, bool) or not isinstance(base, int) or base <= 0:
                raise ValueError("LP_SUPER_BASE_VOCAB_SIZE must be a positive integer")

    def validate_penalties(self, pently_rho, vocab_utilisation_weight):
        if self.training_mode != "standard" and (pently_rho != 0 or vocab_utilisation_weight != 0):
            raise ValueError(
                "Document LP modes currently require PENTLY_RHO=0 and "
                "VOCAB_UTILISATION_WEIGHT=0 (compression objective only)"
            )

    def base_size(self, vocab_size, reserved_size):
        size = self.super_base_vocab_size
        if size is None:
            size = reserved_size + (vocab_size - reserved_size) // 2
        if not reserved_size < size < vocab_size:
            raise ValueError(
                f"Super base vocabulary size {size} must satisfy "
                f"{reserved_size} < base_size < {vocab_size}"
            )
        return size


def serialize_pretokenizer(pretokenizer):
    """Save only pretokenization; never introduce a synthetic prefix space."""
    backend = getattr(pretokenizer, "backend_tokenizer", pretokenizer)
    component = getattr(backend, "pre_tokenizer", backend)
    state = json.loads(component.__getstate__())

    def disable_prefix(value):
        if isinstance(value, dict):
            if value.get("type") == "ByteLevel":
                value["add_prefix_space"] = False
            for child in value.values():
                disable_prefix(child)
        elif isinstance(value, list):
            for child in value:
                disable_prefix(child)

    disable_prefix(state)
    document = json.loads(BackendTokenizer(BPE()).to_str())
    document["pre_tokenizer"] = state
    serialized = json.dumps(document, ensure_ascii=False)
    BackendTokenizer.from_str(serialized)
    return serialized


class DocumentSplitter:
    def __init__(self, serialized_pretokenizer, special_tokens):
        self.pretokenizer = BackendTokenizer.from_str(serialized_pretokenizer).pre_tokenizer
        self.special_tokens = frozenset(special_tokens)
        if any(not isinstance(token, str) or not token for token in special_tokens):
            raise ValueError("Special tokens must be nonempty strings")
        self.special_pattern = (
            re.compile("(" + "|".join(re.escape(token) for token in sorted(
                self.special_tokens, key=lambda token: (-len(token), token)
            )) + ")") if self.special_tokens else None
        )

    def split(self, text):
        """Yield (is_special, encoded_text, byte_boundaries) for one document."""
        if not isinstance(text, str):
            raise TypeError("Each document must be a string")
        parts = self.special_pattern.split(text) if self.special_pattern else [text]
        for part in parts:
            if not part:
                continue
            if part in self.special_tokens:
                yield True, part, []
                continue
            pieces = [token for token, _ in self.pretokenizer.pre_tokenize_str(part) if token]
            encoded = "".join(pieces)
            if encoded != encode_bytes(part):
                raise ValueError("Document LP requires lossless ByteLevel pretokenization")
            boundaries = [0]
            for piece in pieces:
                boundaries.append(boundaries[-1] + len(piece))
            yield False, encoded, boundaries


class SpanPolicy:
    """Enumerate exactly the admissible byte endpoints, including fallback edges."""

    def __init__(self, boundaries, mode, max_token_bytes=0):
        if mode not in {"standard", "boundless", "super"}:
            raise ValueError(f"Unknown span policy: {mode}")
        self.boundaries = list(boundaries)
        if not self.boundaries or self.boundaries[0] != 0 or any(
            right <= left for left, right in zip(self.boundaries, self.boundaries[1:])
        ):
            raise ValueError("Boundaries must start at zero and increase strictly")
        self.boundary_set = frozenset(boundaries)
        self.mode = mode
        self.limit = max_token_bytes or self.boundaries[-1]

    def ends(self, start):
        index = bisect_right(self.boundaries, start)
        if index == len(self.boundaries):
            return
        limit = min(self.boundaries[-1], start + self.limit)
        if self.mode == "super":
            if start in self.boundary_set:
                for boundary_index in range(index, len(self.boundaries)):
                    end = self.boundaries[boundary_index]
                    if end > limit:
                        break
                    yield end
            return
        pretoken_end = self.boundaries[index]
        yield from range(start + 1, min(pretoken_end, limit) + 1)
        if self.mode == "boundless" and start in self.boundary_set:
            for boundary_index in range(index + 1, len(self.boundaries)):
                end = self.boundaries[boundary_index]
                if end > limit:
                    break
                yield end

    def spans(self):
        starts = self.boundaries[:-1] if self.mode == "super" else range(self.boundaries[-1])
        for start in starts:
            for end in self.ends(start):
                yield start, end


def segment(encoded, boundaries, vocabulary, mode, max_token_bytes=0):
    """Return byte spans for the minimum-token path with stable local ties."""
    policy = SpanPolicy(boundaries, mode, max_token_bytes)
    n = len(encoded)
    costs = [n + 1] * (n + 1)
    next_end = [-1] * n
    costs[n] = 0
    for start in range(n - 1, -1, -1):
        best = None
        for end in policy.ends(start):
            token = encoded[start:end]
            if token not in vocabulary or costs[end] > n:
                continue
            key = (1 + costs[end], start - end, token)
            if best is None or key < best:
                best = key
                next_end[start] = end
        if best is not None:
            costs[start] = best[0]
    if costs[0] > n:
        raise ValueError("Vocabulary cannot cover the document under its saved boundary rules")
    spans = []
    start = 0
    while start < n:
        end = next_end[start]
        spans.append((start, end))
        start = end
    return spans


def round_document_vocab(possible_tokens, fixed_tokens, target_size, scheme="det"):
    """Round only the learned budget; fixed entries keep their order and IDs."""
    fixed = list(dict.fromkeys(fixed_tokens))
    fixed_set = set(fixed)
    if len(fixed) > target_size:
        raise ValueError("Fixed vocabulary exceeds the target vocabulary size")
    candidates = {token.token: token for token in possible_tokens if token.token not in fixed_set}
    if scheme == "all_ones":
        selected = sorted(token.token for token in candidates.values() if token.lp_value >= 0.99)
    else:
        if scheme not in {"det", "bias"}:
            raise ValueError(f"Unsupported rounding scheme: {scheme}")
        ranked = sorted(candidates.values(), key=lambda token: (
            -(token.lp_value / len(token.token) if scheme == "bias" else token.lp_value), token.token
        ))
        count = target_size - len(fixed)
        if len(ranked) < count:
            raise ValueError(
                f"Insufficient candidates for vocabulary size {target_size}: "
                f"{len(fixed)} fixed + {len(ranked)} learned candidates"
            )
        selected = [token.token for token in ranked[:count]]
    return fixed + selected


class DocumentLPTokenizer:
    def __init__(self, vocabulary, metadata):
        self.metadata = json.loads(json.dumps(metadata))
        if metadata.get("format_version") != FORMAT_VERSION:
            raise ValueError("Unsupported document LP format version")
        self.mode = metadata["training_mode"]
        if self.mode not in {"boundless", "super"}:
            raise ValueError("DocumentLPTokenizer requires boundless or super mode")
        self.max_token_bytes = TrainingOptions(self.mode, metadata["max_token_bytes"]).max_token_bytes
        self.vocabulary = list(vocabulary)
        if len(set(self.vocabulary)) != len(self.vocabulary):
            raise ValueError("Vocabulary contains duplicate tokens")
        self.vocab = {token: index for index, token in enumerate(self.vocabulary)}
        self.special_tokens = list(metadata["special_tokens"])
        if not set(BYTE_ALPHABET + self.special_tokens).issubset(self.vocab):
            raise ValueError("Vocabulary must contain all 256 bytes and every configured special token")
        self.splitter = DocumentSplitter(metadata["pretokenizer"], self.special_tokens)
        self.base_vocabulary = list(metadata.get("base_vocabulary", []))
        self.base_set = frozenset(self.base_vocabulary)
        if self.mode == "super" and (
            not set(BYTE_ALPHABET + self.special_tokens).issubset(self.base_set)
            or not self.base_set.issubset(self.vocab)
            or self.vocabulary[:len(self.base_vocabulary)] != self.base_vocabulary
        ):
            raise ValueError("Super vocabulary must retain the ordered base vocabulary and its IDs")
        ordinary = set(self.vocab).difference(self.special_tokens)
        if any(not token or any(char not in BYTE_DECODER for char in token) for token in ordinary):
            raise ValueError("Ordinary tokens must be nonempty ByteLevel strings")
        if self.max_token_bytes and any(len(token) > self.max_token_bytes for token in ordinary):
            raise ValueError("Vocabulary contains a token exceeding the saved byte cap")

    def __len__(self):
        return len(self.vocabulary)

    def get_vocab(self):
        return dict(self.vocab)

    def encode(self, text):
        ids = []
        for is_special, encoded, boundaries in self.splitter.split(text):
            if is_special:
                ids.append(self.vocab[encoded])
                continue
            if self.mode == "super":
                base_spans = segment(encoded, boundaries, self.base_set, "standard", self.max_token_bytes)
                boundaries = [0] + [end for _, end in base_spans]
            for start, end in segment(encoded, boundaries, self.vocab, self.mode, self.max_token_bytes):
                ids.append(self.vocab[encoded[start:end]])
        return ids

    def encode_batch(self, documents):
        return [self.encode(document) for document in documents]

    def decode(self, ids, skip_special_tokens=False):
        parts = []
        pending = bytearray()
        specials = set(self.special_tokens)
        for index in ids:
            if not isinstance(index, int) or not 0 <= index < len(self.vocabulary):
                raise ValueError(f"Invalid token ID: {index!r}")
            token = self.vocabulary[index]
            if token in specials:
                parts.append(pending.decode("utf-8", errors="replace"))
                pending.clear()
                if not skip_special_tokens:
                    parts.append(token)
            else:
                pending.extend(BYTE_DECODER[char] for char in token)
        parts.append(pending.decode("utf-8", errors="replace"))
        return "".join(parts)

    def save_pretrained(self, path):
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / TOKENIZER_FILENAME
        destination.write_text(json.dumps({
            "format": "document_lp", "metadata": self.metadata, "vocabulary": self.vocabulary,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (directory / "README.md").write_text(
            "# Document LP tokenizer\n\n"
            "Load with `DocumentLPTokenizer.from_pretrained(path)` from "
            "`lp_tokenizer.document_tokenizer`. `encode(text)` returns token IDs; "
            "`encode_batch(documents)` preserves document boundaries.\n\n"
            "This format requires the repository's boundary-aware encoder and "
            "cannot be loaded as an ordinary Hugging Face Unigram tokenizer.\n",
            encoding="utf-8",
        )
        return str(destination)

    @classmethod
    def from_pretrained(cls, path):
        source = Path(path)
        if source.is_dir():
            source = source / TOKENIZER_FILENAME
        document = json.loads(source.read_text(encoding="utf-8"))
        if document.get("format") != "document_lp":
            raise ValueError("Not a document LP tokenizer")
        return cls(document["vocabulary"], document["metadata"])
