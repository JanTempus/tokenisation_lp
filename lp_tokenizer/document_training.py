"""Document-preserving LP preparation and the two-stage Super LP curriculum."""

from collections import Counter
import gc
import os
from pathlib import Path
import tempfile

from datasets import Dataset, Features, Sequence, Value

from lp_tokenizer.datastructures import possibleToken
from lp_tokenizer.document_tokenizer import (
    BYTE_ALPHABET, FORMAT_VERSION, DocumentSplitter, SpanPolicy,
    round_document_vocab, segment, serialize_pretokenizer,
)
from lp_tokenizer.lp_functions import (
    _build_lp_blocks_from_graph_dataset, _resolve_lp_cache_dir,
    build_cuopt_standard_form,
)


DOCUMENT_FEATURES = Features({
    "encoded": Value("string"), "boundaries": Sequence(Value("int64")),
    "frequency": Value("int64"), "document_id": Value("int64"),
})


def _prepare_batch(batch, indices, serialized_pretokenizer, special_tokens):
    splitter = DocumentSplitter(serialized_pretokenizer, special_tokens)
    output = {key: [] for key in DOCUMENT_FEATURES}
    for document_id, text in zip(indices, batch["text"]):
        if text is None:
            continue
        for is_special, encoded, boundaries in splitter.split(text):
            if is_special:
                # A special token is a fixed, constant-cost barrier. Omitting
                # that constant from the objective cannot affect selection.
                continue
            output["encoded"].append(encoded)
            output["boundaries"].append(boundaries)
            output["frequency"].append(1)
            output["document_id"].append(document_id)
    return output


def prepare_documents(corpus, serialized_pretokenizer, special_tokens, num_proc=1, batch_size=64):
    dataset = corpus if isinstance(corpus, Dataset) else Dataset.from_dict({"text": list(corpus)})
    if "text" not in dataset.column_names:
        raise ValueError("Corpus dataset must contain a 'text' column")
    if not len(dataset):
        return Dataset.from_dict({key: [] for key in DOCUMENT_FEATURES}, features=DOCUMENT_FEATURES)
    return dataset.map(
        _prepare_batch, batched=True, with_indices=True,
        batch_size=batch_size, num_proc=min(num_proc, len(dataset)),
        fn_kwargs={"serialized_pretokenizer": serialized_pretokenizer, "special_tokens": special_tokens},
        remove_columns=dataset.column_names, features=DOCUMENT_FEATURES,
        desc="Preparing independent LP documents",
    )


def _aggregate_pretokens(documents):
    counts = Counter()
    for row in documents:
        for start, end in zip(row["boundaries"], row["boundaries"][1:]):
            counts[row["encoded"][start:end]] += row["frequency"]
    words = sorted(counts)
    return Dataset.from_dict({
        "encoded": words, "boundaries": [[0, len(word)] for word in words],
        "frequency": [counts[word] for word in words], "document_id": [-1] * len(words),
    }, features=DOCUMENT_FEATURES)


def _base_boundaries_batch(batch, base_vocabulary, max_token_bytes):
    vocabulary = frozenset(base_vocabulary)
    boundaries = []
    for encoded, original in zip(batch["encoded"], batch["boundaries"]):
        spans = segment(encoded, original, vocabulary, "standard", max_token_bytes)
        boundaries.append([0] + [end for _, end in spans])
    return {"boundaries": boundaries}


def _candidate_batch(batch, mode, max_token_bytes, fixed_tokens):
    counts = Counter()
    raw_count = 0
    fixed = frozenset(fixed_tokens)
    for encoded, boundaries, frequency in zip(batch["encoded"], batch["boundaries"], batch["frequency"]):
        for start, end in SpanPolicy(boundaries, mode, max_token_bytes).spans():
            token = encoded[start:end]
            if token not in fixed:
                counts[token] += frequency
                raw_count += 1
    return {"tokens": [list(counts)], "counts": [list(counts.values())], "raw_edges": [raw_count]}


def _graph_batch(batch, indices, mode, max_token_bytes, fixed_tokens, token_index):
    fixed = frozenset(fixed_tokens)
    values = {key: [] for key in (
        "source_indices", "string_lengths", "string_frequencies", "edge_counts",
        "edge_starts", "edge_ends", "edge_token_ids", "fixed_edge_counts",
        "fixed_edge_starts", "fixed_edge_ends",
    )}
    for index, encoded, boundaries, frequency in zip(
        indices, batch["encoded"], batch["boundaries"], batch["frequency"]
    ):
        count = fixed_count = 0
        for start, end in SpanPolicy(boundaries, mode, max_token_bytes).spans():
            token = encoded[start:end]
            if token in fixed:
                values["fixed_edge_starts"].append(start)
                values["fixed_edge_ends"].append(end)
                fixed_count += 1
            elif token in token_index:
                values["edge_starts"].append(start)
                values["edge_ends"].append(end)
                values["edge_token_ids"].append(token_index[token])
                count += 1
        values["source_indices"].append(index)
        values["string_lengths"].append(len(encoded))
        values["string_frequencies"].append(frequency)
        values["edge_counts"].append(count)
        values["fixed_edge_counts"].append(fixed_count)
    result = {key: [value] for key, value in values.items()}
    result.update({
        "batch_start": [indices[0]],
        "vertex_count": [sum(length + 1 for length in values["string_lengths"])],
        "free_edge_count": [sum(values["fixed_edge_counts"])],
        "filtered_edge_count": [sum(values["edge_counts"])],
    })
    return result


def prepare_document_model(documents, mode, max_token_bytes, fixed_tokens=BYTE_ALPHABET,
                           num_proc=1, batch_size=64, min_token_count=1, verbose=True):
    """Build ordinary cuOpt matrices using only spans accepted by the encoder."""
    if num_proc < 1 or batch_size < 1:
        raise ValueError("NUM_PROC and BATCH_SIZE must be positive")
    fixed_tokens = frozenset(fixed_tokens)
    common = {"mode": mode, "max_token_bytes": max_token_bytes, "fixed_tokens": fixed_tokens}
    tokens = []
    if not len(documents):
        blocks = _build_lp_blocks_from_graph_dataset([], [], verbose=verbose)
    else:
        workers = min(num_proc, len(documents))
        # Each GPU stage owns its cache files; concurrent stages cannot clobber
        # candidate_counts.arrow or graph_edges.arrow from another worker.
        with tempfile.TemporaryDirectory(prefix="document-lp-", dir=_resolve_lp_cache_dir()) as cache:
            candidate_chunks = documents.map(
                _candidate_batch, batched=True, batch_size=batch_size, num_proc=workers,
                fn_kwargs=common, remove_columns=documents.column_names,
                features=Features({"tokens": Sequence(Value("string")), "counts": Sequence(Value("int64")),
                                   "raw_edges": Value("int64")}),
                cache_file_name=str(Path(cache) / "candidates.arrow"), load_from_cache_file=False,
                writer_batch_size=1, desc=f"Counting {mode} LP candidates",
            )
            counts = Counter()
            raw_edges = 0
            for row in candidate_chunks:
                counts.update(dict(zip(row["tokens"], row["counts"])))
                raw_edges += row["raw_edges"]
            tokens = [possibleToken(token, instance_count=counts[token], index=index)
                      for index, token in enumerate(sorted(token for token in counts if counts[token] > min_token_count))]
            if verbose:
                print(f"[document-lp] mode={mode}, rows={len(documents):,}, raw_edges={raw_edges:,}, "
                      f"candidates={len(counts):,}, kept={len(tokens):,}, max_token_bytes={max_token_bytes}")
            del counts, candidate_chunks
            graph_features = Features({key: Sequence(Value("int64")) for key in (
                "source_indices", "string_lengths", "string_frequencies", "edge_counts", "edge_starts",
                "edge_ends", "edge_token_ids", "fixed_edge_counts", "fixed_edge_starts", "fixed_edge_ends",
            )})
            graph_features.update({key: Value("int64") for key in (
                "batch_start", "vertex_count", "free_edge_count", "filtered_edge_count",
            )})
            graph = documents.map(
                _graph_batch, batched=True, with_indices=True, batch_size=batch_size, num_proc=workers,
                fn_kwargs={**common, "token_index": {token.token: token.token_index for token in tokens}},
                remove_columns=documents.column_names, features=graph_features,
                cache_file_name=str(Path(cache) / "graph.arrow"), load_from_cache_file=False,
                writer_batch_size=1, desc=f"Building {mode} LP graph",
            )
            blocks = _build_lp_blocks_from_graph_dataset(graph, tokens, verbose=verbose)
            del graph
    data = build_cuopt_standard_form(blocks, numAllowedTokens=0)
    return {
        "problem": None, "variables": None, "budget_constraint": None,
        "cuopt_lp_data": data, "tokens_to_keep": tokens,
        "num_f": data["num_f"], "num_g": data["num_g"], "num_t": data["num_t"],
        "current_budget": None, "current_vocab_size": None,
        "pently_rho": 0.0, "vocab_utilisation_weight": 0.0,
    }


def prepare_document_training(corpus, pretokenizer, special_tokens, options, verbose=True):
    serialized = serialize_pretokenizer(pretokenizer)
    special_tokens = list(special_tokens)
    if len(set(special_tokens)) != len(special_tokens) or set(special_tokens).intersection(BYTE_ALPHABET):
        raise ValueError("Special tokens must be unique and distinct from byte alphabet entries")
    num_proc = int(os.environ.get("NUM_PROC", "16"))
    batch_size = int(os.environ.get("BATCH_SIZE", "64"))
    if num_proc < 1 or batch_size < 1:
        raise ValueError("NUM_PROC and BATCH_SIZE must be positive")
    documents = prepare_documents(corpus, serialized, special_tokens, num_proc, batch_size)
    training_rows = _aggregate_pretokens(documents) if options.training_mode == "super" else documents
    model = prepare_document_model(
        training_rows, "standard" if options.training_mode == "super" else "boundless",
        options.max_token_bytes, num_proc=num_proc, batch_size=batch_size, verbose=verbose,
    )
    return {
        "document_training": True, "stage_model": model,
        "documents": documents if options.training_mode == "super" else None,
        "options": options, "num_proc": num_proc, "batch_size": batch_size,
        "metadata": {"format_version": FORMAT_VERSION, "training_mode": options.training_mode,
                     "max_token_bytes": options.max_token_bytes, "pretokenizer": serialized,
                     "special_tokens": special_tokens},
    }


def _solve_candidates(model, budget, vocab_size, solve_fn, solver_parameters, verbose):
    if len(model["tokens_to_keep"]) < budget:
        raise ValueError(f"Insufficient candidates: need {budget}, have {len(model['tokens_to_keep'])}; "
                         "use a smaller vocabulary, a larger corpus, or a larger byte cap")
    result = solve_fn(model, numAllowedTokens=budget, vocab_size=vocab_size,
                      solver_parameters=solver_parameters, verbose=verbose)
    # Retain zero-valued candidates too: exact-size rounding may need them
    # when the optimal relaxed solution leaves some of the budget unused.
    scores = {token.token: token.lp_value for token in result["possible_tokens"]}
    result["possible_tokens"] = [possibleToken(
        token.token, scores.get(token.token, 0.0), token.token_instance_count, token.token_index,
    ) for token in model["tokens_to_keep"]]
    return result


def solve_document_vocab(prepared, vocab_size, solve_fn, solver_parameters=None, verbose=True):
    options = prepared["options"]
    metadata = dict(prepared["metadata"])
    metadata["vocab_size"] = vocab_size
    fixed = metadata["special_tokens"] + BYTE_ALPHABET
    reserved = len(fixed)
    if vocab_size <= reserved:
        raise ValueError(f"Vocabulary size must exceed {reserved} reserved tokens")
    if options.training_mode == "boundless":
        model = prepared["stage_model"]
        result = _solve_candidates(model, vocab_size - reserved, vocab_size, solve_fn, solver_parameters, verbose)
    else:
        base_size = options.base_size(vocab_size, reserved)
        print(f"[super-lp] Stage 1: {base_size} tokens; stage 2: {vocab_size - base_size} new tokens")
        first = _solve_candidates(prepared["stage_model"], base_size - reserved, base_size,
                                  solve_fn, solver_parameters, verbose)
        base = round_document_vocab(first["possible_tokens"], fixed, base_size)
        del first
        # Keep the reusable CPU matrices, but release the first GPU problem
        # before constructing the considerably larger document-stage problem.
        for key in ("problem", "variables", "budget_constraint", "current_budget", "current_vocab_size"):
            prepared["stage_model"][key] = None
        gc.collect()
        metadata["base_vocabulary"] = base
        metadata["super_base_vocab_size"] = base_size
        documents = prepared["documents"]
        if len(documents):
            documents = documents.map(
                _base_boundaries_batch, batched=True, batch_size=prepared["batch_size"],
                num_proc=min(prepared["num_proc"], len(documents)),
                fn_kwargs={"base_vocabulary": base, "max_token_bytes": options.max_token_bytes},
                desc="Encoding Super LP base units",
            )
        model = prepare_document_model(documents, "super", options.max_token_bytes,
                                       fixed_tokens=set(base).difference(metadata["special_tokens"]),
                                       num_proc=prepared["num_proc"], batch_size=prepared["batch_size"], verbose=verbose)
        result = _solve_candidates(model, vocab_size - base_size, vocab_size, solve_fn, solver_parameters, verbose)
    return {
        "possible_tokens": result["possible_tokens"], "unique_chars": list(BYTE_ALPHABET),
        "special_tokens": metadata["special_tokens"], "metadata": metadata,
        "x_values": result["x_values"],
    }
