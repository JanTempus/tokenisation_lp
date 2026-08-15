import cvxpy as cp
import numpy as np
import scipy.sparse as sp
import pickle
from collections import defaultdict
import time
from numpy.typing import NDArray
import random
import csv
import psutil
import os
import gc
import shutil
import tempfile
import threading
import matplotlib.pyplot as plt
from datasets import Dataset, Features, Sequence, Value
#import cudf
#import cugraph

from cuopt.linear_programming.solver.solver_parameters import (
    CUOPT_METHOD,
    CUOPT_PDLP_SOLVER_MODE,
    CUOPT_CROSSOVER,
)

from cuopt.linear_programming.solver_settings import (
    SolverMethod,
    PDLPSolverMode,
    SolverSettings,
)


from lp_tokenizer.datastructures import tokenInstance, possibleToken
from lp_tokenizer.celex import (
    EnglishCelex,
    is_morphology_violation,
    validate_morphology_rho,
    write_unmatched_report,
)
import lp_tokenizer.helper_functions as hf


NUM_PROC = int(os.environ.get("NUM_PROC", "16"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "10000"))


def _candidate_count_batch(batch,
                           all_tokens: bool,
                           max_token_length: int):
    token_counts = defaultdict(int)
    raw_edge_count = 0

    for input_string, frequency in zip(batch["pretoken"], batch["frequency"]):
        string_length = len(input_string)
        limit = string_length if all_tokens else min(string_length, max_token_length)
        frequency = int(frequency)

        for token_length in range(2, limit + 1):
            for start in range(string_length - token_length + 1):
                token = input_string[start:start + token_length]
                token_counts[token] += frequency
                raw_edge_count += 1

    return {
        "tokens": [list(token_counts.keys())],
        "counts": [list(token_counts.values())],
        "raw_edge_count": [raw_edge_count],
        "source_row_count": [len(batch["pretoken"])],
    }


def _graph_edge_batch(batch,
                      indices,
                      all_tokens: bool,
                      max_token_length: int,
                      token_index,
                      morphology_enabled: bool = False):
    source_indices = []
    string_lengths = []
    string_frequencies = []
    edge_counts = []
    edge_starts = []
    edge_ends = []
    edge_token_ids = []
    edge_morph_violations = []

    for source_index, input_string, frequency in zip(
        indices, batch["pretoken"], batch["frequency"]
    ):
        string_length = len(input_string)
        limit = string_length if all_tokens else min(string_length, max_token_length)
        edge_count = 0
        if morphology_enabled:
            row_offset = len(source_indices)
            unmatched = bool(batch["celex_unmatched"][row_offset])
            aligned_spans = set(
                zip(
                    batch["celex_aligned_span_starts"][row_offset],
                    batch["celex_aligned_span_ends"][row_offset],
                )
            )

        for token_length in range(2, limit + 1):
            for start in range(string_length - token_length + 1):
                end = start + token_length
                token_id = token_index.get(input_string[start:end])
                if token_id is None:
                    continue
                edge_starts.append(start)
                edge_ends.append(end)
                edge_token_ids.append(token_id)
                if morphology_enabled:
                    edge_morph_violations.append(
                        is_morphology_violation(
                            start, end, unmatched, aligned_spans
                        )
                    )
                edge_count += 1

        source_indices.append(int(source_index))
        string_lengths.append(string_length)
        string_frequencies.append(int(frequency))
        edge_counts.append(edge_count)

    result = {
        "batch_start": [source_indices[0] if source_indices else -1],
        "source_indices": [source_indices],
        "string_lengths": [string_lengths],
        "string_frequencies": [string_frequencies],
        "edge_counts": [edge_counts],
        "edge_starts": [edge_starts],
        "edge_ends": [edge_ends],
        "edge_token_ids": [edge_token_ids],
        "vertex_count": [sum(length + 1 for length in string_lengths)],
        "free_edge_count": [sum(string_lengths)],
        "filtered_edge_count": [len(edge_token_ids)],
    }
    if morphology_enabled:
        result["edge_morph_violations"] = [edge_morph_violations]
    return result


def _celex_annotation_batch(batch, celex):
    decoded_forms = []
    unmatched_flags = []
    unmatched_reasons = []
    aligned_span_starts = []
    aligned_span_ends = []

    for pretoken in batch["pretoken"]:
        match = celex.match_pretoken(pretoken)
        spans = sorted(match.aligned_spans)
        decoded_forms.append(match.decoded_form)
        unmatched_flags.append(match.unmatched)
        unmatched_reasons.append(match.reason)
        aligned_span_starts.append([start for start, _ in spans])
        aligned_span_ends.append([end for _, end in spans])

    return {
        "celex_decoded_form": decoded_forms,
        "celex_unmatched": unmatched_flags,
        "celex_unmatched_reason": unmatched_reasons,
        "celex_aligned_span_starts": aligned_span_starts,
        "celex_aligned_span_ends": aligned_span_ends,
    }


def _resolve_lp_cache_dir():
    configured_dir = os.environ.get("LP_GRAPH_CACHE_DIR")
    if configured_dir:
        cache_dir = os.path.abspath(configured_dir)
    else:
        cache_base = os.environ.get("TMPDIR") or tempfile.gettempdir()
        job_id = os.environ.get("SLURM_JOB_ID") or str(os.getpid())
        cache_dir = os.path.join(cache_base, f"tokenisation_lp_{job_id}")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _format_bytes(byte_count):
    value = float(byte_count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0


def _raw_graph_size(pretoken_dataset, all_tokens, max_token_length):
    raw_edge_count = 0
    free_edge_count = 0

    if hasattr(pretoken_dataset, "iter"):
        batches = pretoken_dataset.iter(batch_size=10_000)
        string_batches = (batch["pretoken"] for batch in batches)
    else:
        string_batches = (pretoken_dataset["pretoken"],)

    for strings in string_batches:
        for input_string in strings:
            string_length = len(input_string)
            limit = string_length if all_tokens else min(string_length, max_token_length)
            if limit >= 2:
                raw_edge_count += (
                    (limit - 1) * (string_length + 1)
                    - (limit * (limit + 1) // 2 - 1)
                )
            free_edge_count += string_length

    return raw_edge_count, free_edge_count


def _dataset_cache_size(dataset):
    return sum(
        os.path.getsize(cache_file["filename"])
        for cache_file in dataset.cache_files
        if os.path.exists(cache_file["filename"])
    )


def prepare_vocab_lp_data(inputStringList: list[str],
                          inputStringFreq: list[int],
                          minTokenCount: int = 1,
                          maxTokenLength: int = 5,
                          all_tokens: bool = True,
                          verbose: bool = True):
    numStrings = len(inputStringList)
    total_start = time.perf_counter()

    edgesList = []
    tokensList = []
    freeEdgesList = []
    numVertices = []

    if verbose:
        mode = "all substrings" if all_tokens else f"substrings up to length {maxTokenLength}"
        print(
            f"[lp-data] Generating candidate edges for "
            f"{numStrings:,} unique pre-tokens ({mode})"
        )
    edge_start = time.perf_counter()
    progress_interval = max(1, numStrings // 10)
    if all_tokens:
        for i in range(numStrings):
            stringLen = len(inputStringList[i])
            edgesList.append(hf.get_all_nonFree_substrings(inputStringList[i]))
            tokensList.append(hf.get_tokens(inputStringList[i]))
            freeEdgesList.append(hf.get_all_free_substrings(inputStringList[i]))
            numVertices.append(stringLen + 1)
            completed = i + 1
            if verbose and (completed % progress_interval == 0 or completed == numStrings):
                print(
                    f"[lp-data] Candidate-edge progress: "
                    f"{completed:,}/{numStrings:,} "
                    f"({100.0 * completed / numStrings:.0f}%) in "
                    f"{time.perf_counter() - edge_start:.1f}s"
                )
    else:
        for i in range(numStrings):
            stringLen = len(inputStringList[i])
            edgesList.append(hf.get_all_nonFree_substrings_upto_len_t(inputStringList[i], maxTokenLength))
            tokensList.append(hf.get_tokens_upto_len_t(inputStringList[i], maxTokenLength))
            freeEdgesList.append(hf.get_all_free_substrings(inputStringList[i]))
            numVertices.append(stringLen + 1)
            completed = i + 1
            if verbose and (completed % progress_interval == 0 or completed == numStrings):
                print(
                    f"[lp-data] Candidate-edge progress: "
                    f"{completed:,}/{numStrings:,} "
                    f"({100.0 * completed / numStrings:.0f}%) in "
                    f"{time.perf_counter() - edge_start:.1f}s"
                )

    if verbose:
        print(
            f"[lp-data] Candidate-edge generation finished in "
            f"{time.perf_counter() - edge_start:.1f}s; "
            f"non-free edges={sum(map(len, edgesList)):,}; "
            f"free edges={sum(map(len, freeEdgesList)):,}"
        )

    token_start = time.perf_counter()
    if verbose:
        print("[lp-data] Deduplicating candidate tokens")
    tokens = list(set([item for sublist in tokensList for item in sublist]))
    if verbose:
        print(
            f"[lp-data] Deduplication finished in "
            f"{time.perf_counter() - token_start:.1f}s; "
            f"candidate tokens={len(tokens):,}"
        )

    count_start = time.perf_counter()
    if verbose:
        print("[lp-data] Counting candidate-token instances")
    hf.update_token_instance_counts(tokens, inputStringFreq, edgesList)
    if verbose:
        print(
            f"[lp-data] Token-instance counting finished in "
            f"{time.perf_counter() - count_start:.1f}s"
        )

    filter_start = time.perf_counter()
    if verbose:
        print(f"[lp-data] Filtering tokens with count <= {minTokenCount:,}")
    tokens_to_keep = [token for token in tokens if token.token_instance_count > minTokenCount]
    keep_set = set(t.token for t in tokens_to_keep)

    filtered_edgesList = [
        [token for token in sublist if token.token in keep_set]
        for sublist in edgesList
    ]
    if verbose:
        print(
            f"[lp-data] Filtering finished in "
            f"{time.perf_counter() - filter_start:.1f}s; "
            f"tokens kept={len(tokens_to_keep):,}; "
            f"non-free edges kept={sum(map(len, filtered_edgesList)):,}; "
            f"LP-data total={time.perf_counter() - total_start:.1f}s"
        )

    return filtered_edgesList, freeEdgesList, numVertices, tokens_to_keep


def build_lp_blocks(edgesList: list[list[tokenInstance]],
                    edgeListWeight: list[int],
                    tokens: list[possibleToken],
                    freeEdgesList: list[list[tokenInstance]],
                    numVerticesList: list[int],
                    verbose: bool = True):
    numStrings = len(edgesList)
    if numStrings != len(freeEdgesList):
        raise ValueError("edgesList and freeEdgesList must have the same length.")
    if numStrings != len(edgeListWeight):
        raise ValueError("edgeListWeight must have one entry per string.")
    if numStrings != len(numVerticesList):
        raise ValueError("numVerticesList must have one entry per string.")

    numTokens = len(tokens)
    token_index_map = {t.token: i for i, t in enumerate(tokens)}

    A_rows, A_cols, A_data = [], [], []
    B_rows, B_cols, B_data = [], [], []
    M_rows, M_cols, M_data = [], [], []

    BigbVector_parts = []
    BigFreewVector_parts = []
    BigNonFreewVector_parts = []

    A_row_offset = 0
    B_row_offset = 0
    M_row_offset = 0
    A_col_offset = 0
    B_col_offset = 0

    build_start = time.perf_counter()
    if verbose:
        print(
            f"[lp-blocks] Building sparse-matrix coordinate arrays for "
            f"{numStrings:,} pre-tokens and {numTokens:,} candidate tokens"
        )
    progress_interval = max(1, numStrings // 10)
    for i in range(numStrings):
        edges = edgesList[i]
        freeEdges = freeEdgesList[i]
        numEdges = len(edges)
        numFreeEdges = len(freeEdges)
        numVertices = numVerticesList[i]

        for idx, edge in enumerate(edges):
            A_rows.append(edge.start + A_row_offset)
            A_cols.append(idx + A_col_offset)
            A_data.append(1)

            A_rows.append(edge.end + A_row_offset)
            A_cols.append(idx + A_col_offset)
            A_data.append(-1)

        for idx, edge in enumerate(freeEdges):
            B_rows.append(edge.start + B_row_offset)
            B_cols.append(idx + B_col_offset)
            B_data.append(1)

            B_rows.append(edge.end + B_row_offset)
            B_cols.append(idx + B_col_offset)
            B_data.append(-1)

        for j, edge in enumerate(edges):
            tokenIndex = token_index_map[edge.token]
            M_rows.append(j + M_row_offset)
            M_cols.append(tokenIndex)
            M_data.append(1)

        b = np.zeros(numVertices, dtype=float)
        b[0] = 1.0
        b[numVertices - 1] = -1.0
        BigbVector_parts.append(b)

        wnonFree = np.full(numEdges, float(edgeListWeight[i]), dtype=float)
        wFree = np.full(numFreeEdges, float(edgeListWeight[i]), dtype=float)
        BigNonFreewVector_parts.append(wnonFree)
        BigFreewVector_parts.append(wFree)

        A_row_offset += numVertices
        B_row_offset += numVertices
        A_col_offset += numEdges
        B_col_offset += numFreeEdges
        M_row_offset += numEdges
        completed = i + 1
        if verbose and (completed % progress_interval == 0 or completed == numStrings):
            print(
                f"[lp-blocks] Coordinate-array progress: "
                f"{completed:,}/{numStrings:,} "
                f"({100.0 * completed / numStrings:.0f}%) in "
                f"{time.perf_counter() - build_start:.1f}s"
            )

    matrix_start = time.perf_counter()
    if verbose:
        print(
            f"[lp-blocks] Converting coordinate arrays to CSR matrices; "
            f"non-free edges={A_col_offset:,}, free edges={B_col_offset:,}"
        )
    BigAConstraint = sp.coo_matrix(
        (A_data, (A_rows, A_cols)),
        shape=(A_row_offset, A_col_offset),
        dtype=float,
    ).tocsr()
    BigBConstraint = sp.coo_matrix(
        (B_data, (B_rows, B_cols)),
        shape=(B_row_offset, B_col_offset),
        dtype=float,
    ).tocsr()
    BigMConstraint = sp.coo_matrix(
        (M_data, (M_rows, M_cols)),
        shape=(M_row_offset, numTokens),
        dtype=float,
    ).tocsr()

    BigbVector = np.hstack(BigbVector_parts) if BigbVector_parts else np.array([], dtype=float)
    BigFreewVector = np.hstack(BigFreewVector_parts) if BigFreewVector_parts else np.array([], dtype=float)
    BigNonFreewVector = np.hstack(BigNonFreewVector_parts) if BigNonFreewVector_parts else np.array([], dtype=float)
    tokensCap = np.ones(numTokens, dtype=float)
    if verbose:
        print(
            f"[lp-blocks] CSR conversion finished in "
            f"{time.perf_counter() - matrix_start:.1f}s; "
            f"A={BigAConstraint.shape}, nnz={BigAConstraint.nnz:,}; "
            f"B={BigBConstraint.shape}, nnz={BigBConstraint.nnz:,}; "
            f"M={BigMConstraint.shape}, nnz={BigMConstraint.nnz:,}; "
            f"LP-block total={time.perf_counter() - build_start:.1f}s"
        )

    return {
        "BigAConstraint": BigAConstraint,
        "BigBConstraint": BigBConstraint,
        "BigMConstraint": BigMConstraint,
        "BigbVector": BigbVector,
        "BigFreewVector": BigFreewVector,
        "BigNonFreewVector": BigNonFreewVector,
        "tokensCap": tokensCap,
        "numNonFreeEdges": A_col_offset,
        "numFreeEdges": B_col_offset,
        "numTokens": numTokens,
    }


def _build_lp_blocks_from_graph_dataset(graph_dataset,
                                        tokens,
                                        morphology_rho=0.0,
                                        verbose=True):
    build_start = time.perf_counter()
    num_tokens = len(tokens)
    morphology_rho = validate_morphology_rho(morphology_rho)
    morphology_enabled = morphology_rho > 0.0

    if len(graph_dataset) == 0:
        return {
            "BigAConstraint": sp.csr_matrix((0, 0), dtype=float),
            "BigBConstraint": sp.csr_matrix((0, 0), dtype=float),
            "BigMConstraint": sp.csr_matrix((0, num_tokens), dtype=float),
            "BigbVector": np.array([], dtype=float),
            "BigFreewVector": np.array([], dtype=float),
            "BigNonFreewVector": np.array([], dtype=float),
            "tokensCap": np.ones(num_tokens, dtype=float),
            "numNonFreeEdges": 0,
            "numFreeEdges": 0,
            "numTokens": num_tokens,
            "numMorphologyPenalizedEdges": 0,
        }

    batch_starts = np.asarray(graph_dataset["batch_start"], dtype=np.int64)
    batch_order = np.argsort(batch_starts, kind="stable")
    num_vertices = int(sum(graph_dataset["vertex_count"]))
    num_non_free_edges = int(sum(graph_dataset["filtered_edge_count"]))
    num_free_edges = int(sum(graph_dataset["free_edge_count"]))

    max_index_value = max(
        num_vertices,
        num_non_free_edges,
        num_free_edges,
        num_tokens,
        2 * num_non_free_edges,
        2 * num_free_edges,
    )
    index_dtype = (
        np.int32
        if max_index_value <= np.iinfo(np.int32).max
        else np.int64
    )

    if verbose:
        print(
            f"[lp-csr] Allocating compact matrix arrays: "
            f"vertices={num_vertices:,}, non-free edges={num_non_free_edges:,}, "
            f"free edges={num_free_edges:,}, tokens={num_tokens:,}, "
            f"index_dtype={np.dtype(index_dtype).name}"
        )

    a_indices = np.empty(2 * num_non_free_edges, dtype=index_dtype)
    a_data = np.empty(2 * num_non_free_edges, dtype=np.float64)
    a_data[0::2] = 1.0
    a_data[1::2] = -1.0

    b_indices = np.empty(2 * num_free_edges, dtype=index_dtype)
    b_data = np.empty(2 * num_free_edges, dtype=np.float64)
    b_data[0::2] = 1.0
    b_data[1::2] = -1.0

    m_indices = np.empty(num_non_free_edges, dtype=index_dtype)
    m_data = np.ones(num_non_free_edges, dtype=np.float64)

    big_b_vector = np.zeros(num_vertices, dtype=np.float64)
    big_free_weight = np.empty(num_free_edges, dtype=np.float64)
    big_non_free_weight = np.empty(num_non_free_edges, dtype=np.float64)

    vertex_cursor = 0
    edge_cursor = 0
    free_edge_cursor = 0
    expected_source_index = 0
    penalized_edge_count = 0
    fill_start = time.perf_counter()
    progress_interval = max(1, len(batch_order) // 10)

    for completed, dataset_index in enumerate(batch_order, start=1):
        row = graph_dataset[int(dataset_index)]
        source_indices = np.asarray(row["source_indices"], dtype=np.int64)
        expected_indices = np.arange(
            expected_source_index,
            expected_source_index + len(source_indices),
            dtype=np.int64,
        )
        if not np.array_equal(source_indices, expected_indices):
            raise ValueError(
                "Dataset.map graph chunks are missing or out of source-row order."
            )
        expected_source_index += len(source_indices)

        string_lengths = np.asarray(row["string_lengths"], dtype=np.int64)
        string_frequencies = np.asarray(row["string_frequencies"], dtype=np.float64)
        edge_counts = np.asarray(row["edge_counts"], dtype=np.int64)
        edge_starts = np.asarray(row["edge_starts"], dtype=index_dtype)
        edge_ends = np.asarray(row["edge_ends"], dtype=index_dtype)
        edge_token_ids = np.asarray(row["edge_token_ids"], dtype=index_dtype)

        local_edge_count = len(edge_token_ids)
        if (
            int(edge_counts.sum()) != local_edge_count
            or len(edge_starts) != local_edge_count
            or len(edge_ends) != local_edge_count
        ):
            raise ValueError("Inconsistent flattened edge arrays in graph map output.")

        vertex_offsets = vertex_cursor + np.concatenate(
            (
                np.array([0], dtype=np.int64),
                np.cumsum(string_lengths[:-1] + 1, dtype=np.int64),
            )
        )
        repeated_vertex_offsets = np.repeat(vertex_offsets, edge_counts)
        edge_end_cursor = edge_cursor + local_edge_count

        a_indices[2 * edge_cursor:2 * edge_end_cursor:2] = (
            repeated_vertex_offsets + edge_starts
        )
        a_indices[2 * edge_cursor + 1:2 * edge_end_cursor:2] = (
            repeated_vertex_offsets + edge_ends
        )
        m_indices[edge_cursor:edge_end_cursor] = edge_token_ids
        big_non_free_weight[edge_cursor:edge_end_cursor] = np.repeat(
            string_frequencies, edge_counts
        )
        if morphology_enabled:
            violations = np.asarray(row["edge_morph_violations"], dtype=np.float64)
            if len(violations) != local_edge_count:
                raise ValueError(
                    "Morphology violation flags do not match the flattened edges."
                )
            big_non_free_weight[edge_cursor:edge_end_cursor] *= (
                1.0 + morphology_rho * violations
            )
            penalized_edge_count += int(violations.sum())

        local_free_edge_count = int(string_lengths.sum())
        free_edge_end_cursor = free_edge_cursor + local_free_edge_count
        repeated_free_offsets = np.repeat(vertex_offsets, string_lengths)
        flat_string_starts = np.repeat(
            np.concatenate(
                (
                    np.array([0], dtype=np.int64),
                    np.cumsum(string_lengths[:-1], dtype=np.int64),
                )
            ),
            string_lengths,
        )
        free_local_starts = (
            np.arange(local_free_edge_count, dtype=np.int64) - flat_string_starts
        )
        b_indices[2 * free_edge_cursor:2 * free_edge_end_cursor:2] = (
            repeated_free_offsets + free_local_starts
        )
        b_indices[2 * free_edge_cursor + 1:2 * free_edge_end_cursor:2] = (
            repeated_free_offsets + free_local_starts + 1
        )
        big_free_weight[free_edge_cursor:free_edge_end_cursor] = np.repeat(
            string_frequencies, string_lengths
        )

        big_b_vector[vertex_offsets] = 1.0
        big_b_vector[vertex_offsets + string_lengths] = -1.0

        vertex_cursor += int((string_lengths + 1).sum())
        edge_cursor = edge_end_cursor
        free_edge_cursor = free_edge_end_cursor

        if verbose and (
            completed % progress_interval == 0 or completed == len(batch_order)
        ):
            print(
                f"[lp-csr] Array-fill progress: {completed:,}/{len(batch_order):,} "
                f"chunks ({100.0 * completed / len(batch_order):.0f}%) in "
                f"{time.perf_counter() - fill_start:.1f}s"
            )

    if (
        vertex_cursor != num_vertices
        or edge_cursor != num_non_free_edges
        or free_edge_cursor != num_free_edges
    ):
        raise ValueError("Graph map totals do not match the assembled matrix arrays.")

    matrix_start = time.perf_counter()
    if verbose:
        print("[lp-csr] Converting compact column data to CSR matrices")

    a_indptr = np.arange(
        0, 2 * num_non_free_edges + 1, 2, dtype=index_dtype
    )
    big_a_constraint = sp.csc_matrix(
        (a_data, a_indices, a_indptr),
        shape=(num_vertices, num_non_free_edges),
    ).tocsr()

    b_indptr = np.arange(
        0, 2 * num_free_edges + 1, 2, dtype=index_dtype
    )
    big_b_constraint = sp.csc_matrix(
        (b_data, b_indices, b_indptr),
        shape=(num_vertices, num_free_edges),
    ).tocsr()

    m_indptr = np.arange(num_non_free_edges + 1, dtype=index_dtype)
    big_m_constraint = sp.csr_matrix(
        (m_data, m_indices, m_indptr),
        shape=(num_non_free_edges, num_tokens),
    )

    if verbose:
        print(
            f"[lp-csr] CSR construction finished in "
            f"{time.perf_counter() - matrix_start:.1f}s; "
            f"A={big_a_constraint.shape}, nnz={big_a_constraint.nnz:,}; "
            f"B={big_b_constraint.shape}, nnz={big_b_constraint.nnz:,}; "
            f"M={big_m_constraint.shape}, nnz={big_m_constraint.nnz:,}; "
            f"total={time.perf_counter() - build_start:.1f}s"
        )

    return {
        "BigAConstraint": big_a_constraint,
        "BigBConstraint": big_b_constraint,
        "BigMConstraint": big_m_constraint,
        "BigbVector": big_b_vector,
        "BigFreewVector": big_free_weight,
        "BigNonFreewVector": big_non_free_weight,
        "tokensCap": np.ones(num_tokens, dtype=float),
        "numNonFreeEdges": num_non_free_edges,
        "numFreeEdges": num_free_edges,
        "numTokens": num_tokens,
        "numMorphologyPenalizedEdges": penalized_edge_count,
    }


def prepare_vocab_lp_blocks_dataset(pretoken_dataset,
                                    min_token_count=1,
                                    max_token_length=5,
                                    all_tokens=True,
                                    num_proc=NUM_PROC,
                                    map_batch_size=BATCH_SIZE,
                                    morphology_rho=0.0,
                                    celex_dir=None,
                                    unmatched_report_path=None,
                                    verbose=True):
    required_columns = {"pretoken", "frequency"}
    missing_columns = required_columns.difference(pretoken_dataset.column_names)
    if missing_columns:
        raise ValueError(
            f"Pretoken dataset is missing required columns: {sorted(missing_columns)}"
        )

    morphology_rho = validate_morphology_rho(morphology_rho)
    morphology_enabled = morphology_rho > 0.0
    worker_count = min(num_proc, max(1, len(pretoken_dataset)))
    cache_dir = _resolve_lp_cache_dir()

    raw_edge_count, raw_free_edge_count = _raw_graph_size(
        pretoken_dataset, all_tokens, max_token_length
    )
    numeric_edge_payload = raw_edge_count * 12
    disk_free = shutil.disk_usage(cache_dir).free
    mode = (
        "all substrings"
        if all_tokens
        else f"substrings up to length {max_token_length}"
    )
    if verbose:
        print(
            f"[lp-map] Configuration: rows={len(pretoken_dataset):,}, "
            f"workers={worker_count}, batch_size={map_batch_size:,}, mode={mode}"
        )
        print(
            f"[lp-map] Raw graph estimate: non-free edges={raw_edge_count:,}, "
            f"free edges={raw_free_edge_count:,}; unfiltered numeric edge "
            f"payload≈{_format_bytes(numeric_edge_payload)}"
        )
        print(
            f"[lp-map] Arrow cache: {cache_dir}; "
            f"available={_format_bytes(disk_free)}"
        )
    if disk_free < numeric_edge_payload:
        print(
            f"[lp-map] WARNING: available cache space is below the unfiltered "
            f"numeric edge estimate. The run can still fit if count filtering "
            f"removes enough edges."
        )

    candidate_features = Features(
        {
            "tokens": Sequence(Value("string")),
            "counts": Sequence(Value("int64")),
            "raw_edge_count": Value("int64"),
            "source_row_count": Value("int64"),
        }
    )
    candidate_start = time.perf_counter()
    try:
        candidate_chunks = pretoken_dataset.map(
            _candidate_count_batch,
            batched=True,
            batch_size=map_batch_size,
            num_proc=worker_count,
            fn_kwargs={
                "all_tokens": all_tokens,
                "max_token_length": max_token_length,
            },
            remove_columns=pretoken_dataset.column_names,
            features=candidate_features,
            writer_batch_size=1,
            load_from_cache_file=False,
            cache_file_name=os.path.join(cache_dir, "candidate_counts.arrow"),
            new_fingerprint=f"lp-candidates-{os.getpid()}-{time.time_ns()}",
            desc="Counting LP candidate substrings",
        )
    except Exception as error:
        current_free = shutil.disk_usage(cache_dir).free
        raise RuntimeError(
            f"Candidate Dataset.map failed. Arrow cache={cache_dir}, "
            f"available={_format_bytes(current_free)}"
        ) from error
    candidate_map_time = time.perf_counter() - candidate_start
    candidate_cache_size = _dataset_cache_size(candidate_chunks)
    if verbose:
        print(
            f"[lp-map] Candidate map finished in {candidate_map_time:.1f}s; "
            f"chunks={len(candidate_chunks):,}; "
            f"Arrow={_format_bytes(candidate_cache_size)}"
        )

    reduce_start = time.perf_counter()
    token_counts = defaultdict(int)
    mapped_raw_edge_count = 0
    mapped_source_rows = 0
    for row in candidate_chunks:
        mapped_raw_edge_count += int(row["raw_edge_count"])
        mapped_source_rows += int(row["source_row_count"])
        for token, count in zip(row["tokens"], row["counts"]):
            token_counts[token] += int(count)

    if mapped_raw_edge_count != raw_edge_count:
        raise ValueError(
            f"Candidate map counted {mapped_raw_edge_count:,} raw edges, "
            f"but {raw_edge_count:,} were expected."
        )
    if mapped_source_rows != len(pretoken_dataset):
        raise ValueError(
            f"Candidate map covered {mapped_source_rows:,} rows, "
            f"but {len(pretoken_dataset):,} were expected."
        )

    candidate_token_count = len(token_counts)
    kept_tokens = sorted(
        token
        for token, count in token_counts.items()
        if count > min_token_count
    )
    kept_counts = [token_counts[token] for token in kept_tokens]
    token_index = {
        token: token_id
        for token_id, token in enumerate(kept_tokens)
    }
    del token_counts
    gc.collect()
    reduce_time = time.perf_counter() - reduce_start
    if verbose:
        print(
            f"[lp-reduce] Global count merge and filtering finished in "
            f"{reduce_time:.1f}s; candidates={candidate_token_count:,}; "
            f"kept={len(kept_tokens):,} with count > {min_token_count:,}"
        )
    del candidate_chunks
    gc.collect()

    if morphology_enabled:
        morphology_start = time.perf_counter()
        if verbose:
            print(f"[celex] Loading English morphology with rho={morphology_rho:g}")
        celex = EnglishCelex.load(celex_dir)
        pretoken_dataset = pretoken_dataset.map(
            _celex_annotation_batch,
            batched=True,
            batch_size=map_batch_size,
            fn_kwargs={"celex": celex},
            load_from_cache_file=False,
            cache_file_name=os.path.join(cache_dir, "celex_annotations.arrow"),
            new_fingerprint=f"celex-annotations-{os.getpid()}-{time.time_ns()}",
            desc="Matching pre-tokens against English CELEX",
        )
        del celex
        gc.collect()
        total_types = len(pretoken_dataset)
        unmatched_types = int(sum(pretoken_dataset["celex_unmatched"]))
        total_frequency = int(sum(pretoken_dataset["frequency"]))
        unmatched_frequency = int(
            sum(
                int(frequency)
                for frequency, unmatched in zip(
                    pretoken_dataset["frequency"], pretoken_dataset["celex_unmatched"]
                )
                if unmatched
            )
        )
        if unmatched_report_path:
            report_count = write_unmatched_report(
                pretoken_dataset, unmatched_report_path
            )
            if report_count != unmatched_types:
                raise ValueError("CELEX unmatched report count is inconsistent.")
        if verbose:
            print(
                f"[celex] Matched types={total_types - unmatched_types:,}/{total_types:,}; "
                f"matched frequency={total_frequency - unmatched_frequency:,}/"
                f"{total_frequency:,}; unmatched types={unmatched_types:,}; "
                f"unmatched frequency={unmatched_frequency:,}; "
                f"elapsed={time.perf_counter() - morphology_start:.1f}s"
            )
            if unmatched_report_path:
                print(f"[celex] Unmatched pre-token report: {unmatched_report_path}")

    graph_features = Features(
        {
            "batch_start": Value("int64"),
            "source_indices": Sequence(Value("int64")),
            "string_lengths": Sequence(Value("int32")),
            "string_frequencies": Sequence(Value("int64")),
            "edge_counts": Sequence(Value("int64")),
            "edge_starts": Sequence(Value("int32")),
            "edge_ends": Sequence(Value("int32")),
            "edge_token_ids": Sequence(Value("int64")),
            "vertex_count": Value("int64"),
            "free_edge_count": Value("int64"),
            "filtered_edge_count": Value("int64"),
        }
    )
    if morphology_enabled:
        graph_features["edge_morph_violations"] = Sequence(Value("bool"))
    graph_start = time.perf_counter()
    try:
        graph_chunks = pretoken_dataset.map(
            _graph_edge_batch,
            batched=True,
            batch_size=map_batch_size,
            num_proc=worker_count,
            with_indices=True,
            fn_kwargs={
                "all_tokens": all_tokens,
                "max_token_length": max_token_length,
                "token_index": token_index,
                "morphology_enabled": morphology_enabled,
            },
            remove_columns=pretoken_dataset.column_names,
            features=graph_features,
            writer_batch_size=1,
            load_from_cache_file=False,
            cache_file_name=os.path.join(cache_dir, "graph_edges.arrow"),
            new_fingerprint=f"lp-graph-{os.getpid()}-{time.time_ns()}",
            desc="Building filtered LP graph edges",
        )
    except Exception as error:
        current_free = shutil.disk_usage(cache_dir).free
        raise RuntimeError(
            f"Graph Dataset.map failed. Arrow cache={cache_dir}, "
            f"available={_format_bytes(current_free)}"
        ) from error
    del token_index
    gc.collect()
    tokens_to_keep = [
        possibleToken(
            token,
            instance_count=count,
            index=token_id,
        )
        for token_id, (token, count) in enumerate(zip(kept_tokens, kept_counts))
    ]
    del kept_tokens
    del kept_counts
    gc.collect()
    graph_map_time = time.perf_counter() - graph_start
    graph_cache_size = _dataset_cache_size(graph_chunks)
    filtered_edge_count = int(sum(graph_chunks["filtered_edge_count"]))
    if verbose:
        print(
            f"[lp-map] Graph map finished in {graph_map_time:.1f}s; "
            f"chunks={len(graph_chunks):,}; filtered edges={filtered_edge_count:,}; "
            f"Arrow={_format_bytes(graph_cache_size)}"
        )

    csr_start = time.perf_counter()
    lp_blocks = _build_lp_blocks_from_graph_dataset(
        graph_chunks,
        tokens_to_keep,
        morphology_rho=morphology_rho,
        verbose=verbose,
    )
    csr_time = time.perf_counter() - csr_start
    if verbose:
        print(
            f"[lp-map] LP graph pipeline timings: candidate_map={candidate_map_time:.1f}s, "
            f"reduce={reduce_time:.1f}s, graph_map={graph_map_time:.1f}s, "
            f"csr={csr_time:.1f}s, "
            f"Arrow_total={_format_bytes(candidate_cache_size + graph_cache_size)}"
        )
        if morphology_enabled:
            print(
                f"[celex] Penalized non-free edges="
                f"{lp_blocks['numMorphologyPenalizedEdges']:,}/"
                f"{lp_blocks['numNonFreeEdges']:,}"
            )

    return lp_blocks, tokens_to_keep


def build_cuopt_standard_form(lp_blocks, numAllowedTokens: int):
    BigAConstraint = lp_blocks["BigAConstraint"]
    BigBConstraint = lp_blocks["BigBConstraint"]
    BigMConstraint = lp_blocks["BigMConstraint"]
    BigbVector = lp_blocks["BigbVector"]
    BigFreewVector = lp_blocks["BigFreewVector"]
    BigNonFreewVector = lp_blocks["BigNonFreewVector"]

    num_f = lp_blocks["numNonFreeEdges"]
    num_g = lp_blocks["numFreeEdges"]
    num_t = lp_blocks["numTokens"]
    num_x = num_f + num_g + num_t

    zeros_eq_t = sp.csr_matrix((BigAConstraint.shape[0], num_t), dtype=float)
    A_eq = sp.hstack([BigAConstraint, BigBConstraint, zeros_eq_t], format="csr")
    b_eq = BigbVector.astype(float)

    eye_f = sp.identity(num_f, format="csr", dtype=float)
    zeros_fg = sp.csr_matrix((num_f, num_g), dtype=float)
    A_ub_flow = sp.hstack([eye_f, zeros_fg, -BigMConstraint], format="csr")
    b_ub_flow = np.zeros(num_f, dtype=float)

    if num_t > 0:
        sum_t_data = np.ones(num_t, dtype=float)
        sum_t_row = np.zeros(num_t, dtype=int)
        sum_t_col = np.arange(num_f + num_g, num_x, dtype=int)
        A_ub_budget = sp.coo_matrix(
            (sum_t_data, (sum_t_row, sum_t_col)),
            shape=(1, num_x),
            dtype=float,
        ).tocsr()
    else:
        A_ub_budget = sp.csr_matrix((1, num_x), dtype=float)
    b_ub_budget = np.array([float(numAllowedTokens)], dtype=float)

    A_ub = sp.vstack([A_ub_flow, A_ub_budget], format="csr")
    b_ub = np.hstack([b_ub_flow, b_ub_budget])

    # Weighted objective: minimize sum_i w_i * f_i + sum_j w_j * g_j
    # where weights come from pretokenized-string frequencies.
    c = np.hstack([BigNonFreewVector, BigFreewVector, np.zeros(num_t, dtype=float)])
    lower_bounds = np.zeros(num_x, dtype=float)
    upper_bounds = np.full(num_x, 1.0, dtype=float)
    upper_bounds[num_f + num_g:] = 1.0

    return {
        "A_eq": A_eq,
        "b_eq": b_eq,
        "A_ub": A_ub,
        "b_ub": b_ub,
        "c": c,
        "lb": lower_bounds,
        "ub": upper_bounds,
        "num_f": num_f,
        "num_g": num_g,
        "num_t": num_t,
    }


def _build_linear_expression(variables, coeff_indices, coeff_values):
    expr = 0.0
    for col_idx, value in zip(coeff_indices, coeff_values):
        expr += float(value) * variables[int(col_idx)]
    return expr


def _iter_csr_rows(matrix: sp.csr_matrix):
    indptr = matrix.indptr
    indices = matrix.indices
    data = matrix.data
    for row_idx in range(matrix.shape[0]):
        start = indptr[row_idx]
        end = indptr[row_idx + 1]
        yield row_idx, indices[start:end], data[start:end]


def _get_var_value(variable_obj):
    if hasattr(variable_obj, "getValue"):
        return variable_obj.getValue()
    return variable_obj.Value


def _import_cuopt_problem():
    try:
        from cuopt.linear_programming.problem import MINIMIZE, Problem, LinearExpression
    except ImportError:
        from cuopt.linear_programming.problem import Problem, sense
        MINIMIZE = sense.MINIMIZE
        LinearExpression = None
    return Problem, MINIMIZE, LinearExpression


def build_cuopt_problem(cuopt_lp_data, numAllowedTokens: int, verbose: bool = True):
    total_start = time.perf_counter()
    Problem, MINIMIZE, LinearExpression = _import_cuopt_problem()

    A_eq = cuopt_lp_data["A_eq"]
    b_eq = cuopt_lp_data["b_eq"]
    A_ub = cuopt_lp_data["A_ub"]
    b_ub = cuopt_lp_data["b_ub"]
    c = cuopt_lp_data["c"]
    lb = cuopt_lp_data["lb"]
    ub = cuopt_lp_data["ub"]

    num_ub_rows = A_ub.shape[0]
    budget_row_idx = num_ub_rows - 1

    problem = Problem("tokenizer_lp_cuopt")
    variables = []

    phase_start = time.perf_counter()
    if verbose:
        print(f"[cuopt-build] Creating {len(c):,} LP variables")
    for idx in range(len(c)):
        var = problem.addVariable(
            lb=float(lb[idx]),
            ub=float(ub[idx]),
            obj=float(c[idx]),
            name=f"x_{idx}",
        )
        variables.append(var)
    if verbose:
        print(
            f"[cuopt-build] Created {len(variables):,} variables in "
            f"{time.perf_counter() - phase_start:.1f}s"
        )

    phase_start = time.perf_counter()
    if verbose:
        print(f"[cuopt-build] Adding {A_eq.shape[0]:,} equality constraints")
    for row_idx, cols, vals in _iter_csr_rows(A_eq):
        rhs = float(b_eq[row_idx])
        if len(cols) == 0:
            if abs(rhs) > 1e-12:
                raise ValueError("Inconsistent empty equality row encountered while building cuOpt model.")
            continue
        expr = _build_linear_expression(variables, cols, vals)
        problem.addConstraint(expr == rhs, f"eq_{row_idx}")
    if verbose:
        print(
            f"[cuopt-build] Equality constraints finished in "
            f"{time.perf_counter() - phase_start:.1f}s"
        )

    phase_start = time.perf_counter()
    if verbose:
        print(f"[cuopt-build] Adding {num_ub_rows:,} inequality constraints")
    budget_constraint = None
    for row_idx, cols, vals in _iter_csr_rows(A_ub):
        is_budget = row_idx == budget_row_idx
        rhs = float(numAllowedTokens) if is_budget else float(b_ub[row_idx])
        if len(cols) == 0:
            if rhs < -1e-12:
                raise ValueError("Inconsistent empty inequality row encountered while building cuOpt model.")
            continue
        expr = _build_linear_expression(variables, cols, vals)
        name = "budget" if is_budget else f"ub_{row_idx}"
        handle = problem.addConstraint(expr <= rhs, name)
        if is_budget:
            budget_constraint = handle
    if verbose:
        print(
            f"[cuopt-build] Inequality constraints finished in "
            f"{time.perf_counter() - phase_start:.1f}s"
        )

    phase_start = time.perf_counter()
    problem.setObjective(LinearExpression(variables, c, 0.0), MINIMIZE)
    if verbose:
        print(
            f"[cuopt-build] Objective set in "
            f"{time.perf_counter() - phase_start:.1f}s; "
            f"cuOpt build total={time.perf_counter() - total_start:.1f}s"
        )

    return {
        "problem": problem,
        "variables": variables,
        "budget_constraint": budget_constraint,
        "cuopt_lp_data": cuopt_lp_data,
        "num_f": cuopt_lp_data["num_f"],
        "num_g": cuopt_lp_data["num_g"],
        "num_t": cuopt_lp_data["num_t"],
        "current_budget": int(numAllowedTokens),
    }


def solve_cuopt_problem(model, numAllowedTokens: int,
                        solver_parameters=None, verbose: bool = True):
    # Always rebuild the cuOpt Problem from the cached cuopt_lp_data matrices
    # with the requested budget. In-place mutation of the budget constraint
    # RHS on an existing cuOpt Problem was observed to silently fail (the
    # Python-side attribute changed but the solver kept using the original
    # RHS), so every vocab size produced identical primal/dual objectives.
    # The expensive matrices (A_eq, A_ub, ...) are built once in
    # prepare_cuopt_model and reused here; only the Problem/variable wrappers
    # are recreated.
    print(f"[solve_cuopt_problem] Building cuOpt Problem with numAllowedTokens={numAllowedTokens}")
    rebuilt = build_cuopt_problem(
        model["cuopt_lp_data"], numAllowedTokens, verbose=verbose
    )
    model["problem"] = rebuilt["problem"]
    model["variables"] = rebuilt["variables"]
    model["budget_constraint"] = rebuilt["budget_constraint"]
    model["current_budget"] = int(numAllowedTokens)

    num_f = model["num_f"]
    num_g = model["num_g"]
    num_t = model["num_t"]
    problem = model["problem"]
    variables = model["variables"]

    settings = SolverSettings()
    settings.set_parameter(CUOPT_CROSSOVER, False)

    start = time.time()
    problem.solve(settings)
    end = time.time()

    status_obj = getattr(problem, "Status", getattr(problem, "status", None))
    status_name = getattr(status_obj, "name", str(status_obj))
    print(f"problem status name: {status_name}")
    solve_time = getattr(problem, "SolveTime", getattr(problem, "solve_time", end - start))

    x_values = np.array([float(_get_var_value(v)) for v in variables], dtype=float)
    t_offset = num_f + num_g
    t_values = x_values[t_offset:t_offset + num_t]

    count_above_0999 = int((t_values > 0.999).sum())
    count_positive = int((t_values > 0).sum())
    total_t = len(t_values)
    t_sum = float(t_values.sum())
    print(f"[INFO] numAllowedTokens={numAllowedTokens}  sum(t_values)={t_sum:.3f}")
    print(f"[INFO] t_values stats: {count_above_0999} above 0.999, {count_positive} strictly positive, out of {total_t} total")

    return {
        "status_name": status_name,
        "solve_time": solve_time,
        "wall_time": end - start,
        "t_values": t_values,
        "x_values": x_values,
    }


def solve_lp_direct_cuopt(cuopt_lp_data, solver_parameters=None, verbose: bool = True):
    numAllowedTokens = int(cuopt_lp_data["b_ub"][-1])
    model = build_cuopt_problem(cuopt_lp_data, numAllowedTokens, verbose=verbose)
    return solve_cuopt_problem(
        model, numAllowedTokens,
        solver_parameters=solver_parameters, verbose=verbose,
    )

def setup_LP_tokenization(edgesList: list[list[tokenInstance]] , 
            edgeListWeight:list[int] , 
            tokens: list[possibleToken], 
            freeEdgesList: list[list[tokenInstance]], 
            numVerticesList:list[int]):
    
    numStrings = len(edgesList)
    if numStrings != len(freeEdgesList):
        raise ValueError

    numTokens = len(tokens)
    token_index_map = {t.token: i for i, t in enumerate(tokens)}

  

    # Data holders for constructing big sparse matrices in COO format
    A_rows, A_cols, A_data = [], [], []
    B_rows, B_cols, B_data = [], [], []
    M_rows, M_cols, M_data = [], [], []

    BigbVector_parts = []
    BigFreewVector_parts = []
    BigNonFreewVector_parts = []

    A_row_offset = 0
    B_row_offset = 0
    M_row_offset = 0
    A_col_offset = 0
    B_col_offset = 0

    
    for i in range(numStrings):
        edges = edgesList[i]
        freeEdges = freeEdgesList[i]
        numEdges = len(edges)
        numFreeEdges = len(freeEdges)
        numVertices = numVerticesList[i]

        # Flow constraints (A) for non-free edges
        for idx, edge in enumerate(edges):
            A_rows.append(edge.start + A_row_offset)
            A_cols.append(idx + A_col_offset)
            A_data.append(1)

            A_rows.append(edge.end + A_row_offset)
            A_cols.append(idx + A_col_offset)
            A_data.append(-1)

        # Flow constraints (B) for free edges
        for idx, edge in enumerate(freeEdges):
            B_rows.append(edge.start + B_row_offset)
            B_cols.append(idx + B_col_offset)
            B_data.append(1)

            B_rows.append(edge.end + B_row_offset)
            B_cols.append(idx + B_col_offset)
            B_data.append(-1)

        # Token preservation matrix (M)
        for j, edge in enumerate(edges):
            tokenIndex = token_index_map[edge.token]
            M_rows.append(j + M_row_offset)
            M_cols.append(tokenIndex)
            M_data.append(1)

        # b vector
        b = np.zeros(numVertices, dtype=int)
        b[0] = 1
        b[numVertices - 1] = -1
        BigbVector_parts.append(b)

        # weights
        wnonFree = np.full(numEdges, edgeListWeight[i])
        wFree = np.full(numFreeEdges, edgeListWeight[i])
        BigNonFreewVector_parts.append(wnonFree)
        BigFreewVector_parts.append(wFree)

        # Update offsets
        A_row_offset += numVertices
        B_row_offset += numVertices
        A_col_offset += numEdges
        B_col_offset += numFreeEdges
        M_row_offset += numEdges

    # Construct final sparse matrices
    BigAConstraint = sp.coo_matrix((A_data, (A_rows, A_cols)), shape=(A_row_offset, A_col_offset)).tocsr()
    BigBConstraint = sp.coo_matrix((B_data, (B_rows, B_cols)), shape=(B_row_offset, B_col_offset)).tocsr()
    BigMConstraint = sp.coo_matrix((M_data, (M_rows, M_cols)), shape=(M_row_offset, numTokens)).tocsr()
    
    BigbVector = np.hstack(BigbVector_parts)
    BigFreewVector = np.hstack(BigFreewVector_parts)
    BigNonFreewVector = np.hstack(BigNonFreewVector_parts)
    tokensCap = np.ones(numTokens, dtype=float)

    
  

    f=cp.Variable(A_col_offset,nonneg=True )
    g=cp.Variable(B_col_offset,nonneg=True)
    t=cp.Variable(numTokens,nonneg=True)
    numAllowedTokens = cp.Parameter(nonneg=True)

    constraints=[BigAConstraint@f+ BigBConstraint@g==BigbVector,
                 f <= BigMConstraint @ t,
                 cp.sum(t)<=numAllowedTokens,
                 t <=tokensCap]   


    objective=cp.Minimize(BigNonFreewVector.T@f +BigFreewVector.T@g)

    problem = cp.Problem(objective, constraints)
    print("setup_LP_tokenization finished")

    return problem


def tokenize(edgesList: list[list[tokenInstance]] , 
            edgeListWeight:list[int] , 
            numVerticesList:list[int],
            just_size:bool=False):
    
    numStrings = len(edgesList)

    A_rows, A_cols, A_data = [], [], []
   
    BigbVector_parts = []
    BigNonFreewVector_parts = []

    A_row_offset = 0
    A_col_offset = 0

    for i in range(numStrings):
        edges = edgesList[i]
        numEdges = len(edges)
        numVertices = numVerticesList[i]

        # Flow constraints (A) for non-free edges
        for idx, edge in enumerate(edges):
            A_rows.append(edge.start + A_row_offset)
            A_cols.append(idx + A_col_offset)
            A_data.append(1)

            A_rows.append(edge.end + A_row_offset)
            A_cols.append(idx + A_col_offset)
            A_data.append(-1)

        # b vector
        b = np.zeros(numVertices, dtype=int)
        b[0] = 1
        b[numVertices - 1] = -1
        BigbVector_parts.append(b)

        # weights
        wnonFree = np.full(numEdges, edgeListWeight[i])
        BigNonFreewVector_parts.append(wnonFree)

        # Update offsets
        A_row_offset += numVertices
        A_col_offset += numEdges
    # Construct final sparse matrices
    BigAConstraint = sp.coo_matrix((A_data, (A_rows, A_cols)), shape=(A_row_offset, A_col_offset)).tocsr()
   
    
    BigbVector = np.hstack(BigbVector_parts)
    BigNonFreewVector = np.hstack(BigNonFreewVector_parts)  

    f=cp.Variable(A_col_offset,nonneg=True )

    constraints=[BigAConstraint@f==BigbVector]   


    objective=cp.Minimize(BigNonFreewVector.T@f)

    problem = cp.Problem(objective, constraints)

    problem.solve(solver=cp.GLOP)
    #problem.solve(solver=cp.CUOPT)
    flow_values = f.value 
    shortest_paths = []
    offset = 0

    if flow_values is not None:
        if not just_size:
            for i in range(numStrings):
                edges = edgesList[i]
                numEdges = len(edges)
                flows = flow_values[offset:offset+numEdges]
                used_edges = [edges[j].token_index for j in range(numEdges) if flows[j] > 1e-6]  # tolerance for numerical noise
                shortest_paths.append(used_edges)
                offset += numEdges

               
            flat_tokens = []
            for sublist in shortest_paths:
                flat_tokens.extend(sublist)
            return flat_tokens
        else:
            return f.value
    else:
      raise ValueError("Cannot represent data")


def create_vocab(inputStringList: list[str],
                 inputStringFreq: list[int],
                 numAllowedTokens: int, 
                 vocab_size:int,
                 minTokenCount: int = 1,  
                 maxTokenLength: int = 5, 
                 all_tokens: bool = True):

    numStrings = len(inputStringList)

    edgesList = []
    tokensList = []
    freeEdgesList = []
    numVertices = []

    if all_tokens:  
        for i in range(numStrings):
            stringLen = len(inputStringList[i])
            edgesList.append(hf.get_all_nonFree_substrings(inputStringList[i]))
            tokensList.append(hf.get_tokens(inputStringList[i]))
            freeEdgesList.append(hf.get_all_free_substrings(inputStringList[i]))
            numVertices.append(stringLen + 1)
    else:
        for i in range(numStrings):
            stringLen = len(inputStringList[i])
            edgesList.append(hf.get_all_nonFree_substrings_upto_len_t(inputStringList[i], maxTokenLength))
            tokensList.append(hf.get_tokens_upto_len_t(inputStringList[i], maxTokenLength))
            freeEdgesList.append(hf.get_all_free_substrings(inputStringList[i]))
            numVertices.append(stringLen + 1)

    tokens = list(set([item for sublist in tokensList for item in sublist]))
    hf.update_token_instance_counts(tokens, inputStringFreq, edgesList)
    tokens_to_keep = [token for token in tokens if token.token_instance_count > minTokenCount]
    keep_set = set(t.token for t in tokens_to_keep)

    filtered_edgesList = [
        [token for token in sublist if token.token in keep_set]
        for sublist in edgesList
    ]

    lpProblem = setup_LP_tokenization(filtered_edgesList, inputStringFreq, tokens_to_keep, freeEdgesList, numVertices)
    numAllowedTokensParam = lpProblem.parameters()[0]
    numAllowedTokensParam.value = numAllowedTokens

    # # --- Memory tracking setup ---
    # process = psutil.Process(os.getpid())
    # memory_samples = []
    # timestamps = []
    # stop_flag = False

    # def track_memory(interval=0.05):
    #     start_time = time.time()
    #     while not stop_flag:
    #         mem = process.memory_info().rss / (1024**2)  # in MB
    #         memory_samples.append(mem)
    #         timestamps.append(time.time() - start_time)
    #         time.sleep(interval)

    # tracker_thread = threading.Thread(target=track_memory, daemon=True)
    # tracker_thread.start()
    # # --- End memory tracking setup ---

    start = time.time()
    lpProblem.solve(solver=cp.CUOPT,verbose=True)
    # lpProblem.solve(
    #     solver=cp.PDLP,
    #     verbose=True,
    #     solver_opts={
    #         "eps_optimal_absolute": 1.0e-6,
    #         "num_threads": 8,
    #         "num_shards": 32
    #     }
    # )
    end = time.time()

    internal_time=lpProblem.solver_stats.solve_time
    my_time= end - start
    output_file="computation_time.csv"
    with open(output_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([f"Interal Time {internal_time}"])
        writer.writerow([f"My Time {my_time}"])


    # Stop memory tracking
    stop_flag = True
    # tracker_thread.join()

    #print(f"The LP solve took {my_time:.4f} seconds")
    # print(f"Peak memory: {max(memory_samples):.2f} MB, Average memory: {sum(memory_samples)/len(memory_samples):.2f} MB")

    # # Save memory usage plot
    # plt.figure(figsize=(10, 5))
    # plt.plot(timestamps, memory_samples, label="RSS Memory (MB)")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Memory (MB)")
    # plt.title("Memory Usage During LP Solve")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(f"lp_memory_usage_{vocab_size}.png")
    # print(f"Memory usage plot saved to lp_memory_usage_{vocab_size}.png")

    lpVariables = lpProblem.variables()
    tVar = lpVariables[2].value

    possibleTokens = []
    for i in range(len(tokens_to_keep)):
        if tVar[i] > 0.0:
            nonZeroToken = possibleToken(
                tokens_to_keep[i].get_token(),
                tVar[i],
                tokens_to_keep[i].get_count(),
                tokens_to_keep[i].get_index()
            )
            possibleTokens.append(nonZeroToken)
    print("create_vocab finished")

    return possibleTokens


def prepare_cuopt_model(inputStringList: list[str] = None,
                        inputStringFreq: list[int] = None,
                        minTokenCount: int = 1,
                        maxTokenLength: int = 5,
                        all_tokens: bool = True,
                        verbose: bool = True,
                        pretoken_dataset=None,
                        num_proc=NUM_PROC,
                        map_batch_size=BATCH_SIZE,
                        morphology_rho: float = 0.0,
                        celex_dir: str = None,
                        unmatched_report_path: str = None):
    total_start = time.perf_counter()
    if pretoken_dataset is None:
        if inputStringList is None or inputStringFreq is None:
            raise ValueError(
                "Provide either pretoken_dataset or both inputStringList "
                "and inputStringFreq."
            )
        if len(inputStringList) != len(inputStringFreq):
            raise ValueError(
                "inputStringList and inputStringFreq must have the same length."
            )
        pretoken_dataset = Dataset.from_dict(
            {
                "pretoken": inputStringList,
                "frequency": inputStringFreq,
            },
            features=Features(
                {
                    "pretoken": Value("string"),
                    "frequency": Value("int64"),
                }
            ),
        )

    lp_blocks, tokens_to_keep = prepare_vocab_lp_blocks_dataset(
        pretoken_dataset=pretoken_dataset,
        min_token_count=minTokenCount,
        max_token_length=maxTokenLength,
        all_tokens=all_tokens,
        num_proc=num_proc,
        map_batch_size=map_batch_size,
        morphology_rho=morphology_rho,
        celex_dir=celex_dir,
        unmatched_report_path=unmatched_report_path,
        verbose=verbose,
    )
    if verbose:
        print(
            f"[prepare-cuopt] LP blocks finished in "
            f"{time.perf_counter() - total_start:.1f}s"
        )

    phase_start = time.perf_counter()
    if verbose:
        print("[prepare-cuopt] Building cuOpt standard-form matrices")
    cuopt_lp_data = build_cuopt_standard_form(lp_blocks, numAllowedTokens=0)
    if verbose:
        print(
            f"[prepare-cuopt] Standard form finished in "
            f"{time.perf_counter() - phase_start:.1f}s; "
            f"variables={len(cuopt_lp_data['c']):,}; "
            f"equalities={cuopt_lp_data['A_eq'].shape[0]:,}; "
            f"inequalities={cuopt_lp_data['A_ub'].shape[0]:,}"
        )

    phase_start = time.perf_counter()
    if verbose:
        print("[prepare-cuopt] Building initial cuOpt problem wrapper")
    try:
        model = build_cuopt_problem(cuopt_lp_data, numAllowedTokens=0, verbose=verbose)
    except ImportError as import_error:
        raise ImportError(
            "Direct cuOpt LP solve requested, but cuOpt Python modules are not available in this environment."
        ) from import_error

    model["tokens_to_keep"] = tokens_to_keep
    model["morphology_rho"] = validate_morphology_rho(morphology_rho)
    model["num_morphology_penalized_edges"] = lp_blocks[
        "numMorphologyPenalizedEdges"
    ]
    if verbose:
        print(
            f"[prepare-cuopt] Initial cuOpt wrapper finished in "
            f"{time.perf_counter() - phase_start:.1f}s; "
            f"complete model preparation={time.perf_counter() - total_start:.1f}s"
        )
    return model


def _possible_tokens_from_tvar(tokens_to_keep, tVar):
    possibleTokens = []
    for i in range(len(tokens_to_keep)):
        if tVar[i] > 0.0:
            nonZeroToken = possibleToken(
                tokens_to_keep[i].get_token(),
                tVar[i],
                tokens_to_keep[i].get_count(),
                tokens_to_keep[i].get_index(),
            )
            possibleTokens.append(nonZeroToken)
    return possibleTokens


def solve_vocab_on_model(model, numAllowedTokens: int,
                         solver_parameters=None, verbose: bool = True):
    solve_output = solve_cuopt_problem(
        model,
        numAllowedTokens=numAllowedTokens,
        solver_parameters=solver_parameters,
        verbose=verbose,
    )

    status_name = solve_output["status_name"]
    if status_name is not None:
        normalized_status = str(status_name).lower()
        if "optimal" not in normalized_status and "feasible" not in normalized_status:
            raise RuntimeError(f"cuOpt solve did not return an optimal/feasible status. Status={status_name}")

    internal_time = solve_output["solve_time"]
    wall_time = solve_output["wall_time"]
    output_file = "computation_time.csv"
    with open(output_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([f"Interal Time {internal_time}"])
        writer.writerow([f"My Time {wall_time}"])

    possibleTokens = _possible_tokens_from_tvar(model["tokens_to_keep"], solve_output["t_values"])
    return {
        "possible_tokens": possibleTokens,
        "x_values": solve_output["x_values"],
        "t_values": solve_output["t_values"],
        "status_name": status_name,
    }


def create_vocab_cuopt(inputStringList: list[str],
                       inputStringFreq: list[int],
                       numAllowedTokens: int,
                       vocab_size: int,
                       minTokenCount: int = 1,
                       maxTokenLength: int = 5,
                       all_tokens: bool = True,
                       solver_parameters=None,
                       verbose: bool = True,
                       pretoken_dataset=None,
                       morphology_rho: float = 0.0,
                       celex_dir: str = None,
                       unmatched_report_path: str = None):
    model = prepare_cuopt_model(
        inputStringList=inputStringList,
        inputStringFreq=inputStringFreq,
        minTokenCount=minTokenCount,
        maxTokenLength=maxTokenLength,
        all_tokens=all_tokens,
        verbose=verbose,
        pretoken_dataset=pretoken_dataset,
        morphology_rho=morphology_rho,
        celex_dir=celex_dir,
        unmatched_report_path=unmatched_report_path,
    )
    print(
        f"[create-vocab] Model preparation finished; "
        f"starting solve with token budget={numAllowedTokens:,}"
    )
    result = solve_vocab_on_model(
        model,
        numAllowedTokens=numAllowedTokens,
        solver_parameters=solver_parameters,
        verbose=verbose,
    )
    print("create_vocab_cuopt finished")
    return result["possible_tokens"]


def create_vocab_old(inputStringList: list[str],
                    inputStringFreq:list[int],
                    numAllowedTokens:int, 
                    minTokenCount:int=1,  
                    maxTokenLength: int=5, 
                    all_tokens:bool=True ):
    
    numStrings=len(inputStringList)

    edgesList=[]
    tokensList=[]
    freeEdgesList=[]
    numVertices=[]


    if all_tokens:  
        for i in range(numStrings):
            stringLen=len(inputStringList[i])
            edgesList.append(hf.get_all_nonFree_substrings(inputStringList[i]) )
            tokensList.append(hf.get_tokens(inputStringList[i]))
            freeEdgesList.append(hf.get_all_free_substrings(inputStringList[i]))
            numVertices.append(stringLen+1)
        
        tokens=tokensList[0]
        tokens=list(set([item for sublist in tokensList for item in sublist] ))

    else:
        for i in range(numStrings):
            stringLen=len(inputStringList[i])
            edgesList.append(hf.get_all_nonFree_substrings_upto_len_t(inputStringList[i],maxTokenLength) )
            tokensList.append(hf.get_tokens_upto_len_t(inputStringList[i],maxTokenLength))
            freeEdgesList.append(hf.get_all_free_substrings(inputStringList[i]))
            numVertices.append(stringLen+1)

       
        tokens=tokensList[0]
        tokens=list(set([item for sublist in tokensList for item in sublist] ))
    



    hf.update_token_instance_counts(tokens,inputStringFreq,edgesList)

    tokens_to_keep = [token for token in tokens if token.token_instance_count > minTokenCount]

    # Create a set of valid token strings
    keep_set = set(t.token for t in tokens_to_keep)


    # Create a new edgesList that only contains tokens in keep_set
    filtered_edgesList = [
        [token for token in sublist if token.token in keep_set]
        for sublist in edgesList
    ]


    lpProblem=setup_LP_tokenization(filtered_edgesList,inputStringFreq,tokens_to_keep , freeEdgesList,numVertices)

    numAllowedTokensParam = lpProblem.parameters()[0]
    numAllowedTokensParam.value = numAllowedTokens

    start = time.time()
    #lpProblem.solve(solver=cp.GLOP)
    lpProblem.solve(
    solver=cp.PDLP,
    verbose=True,
    solver_opts={
        "eps_optimal_absolute": 1.0e-6,
        "num_threads": 8,
        "num_shards": 32
                 }
    )
    end=time.time()
    print(f"The first iteration took {end - start:.4f} seconds")

    lpVariables=lpProblem.variables()
   
    tVar=lpVariables[2].value
    
    
    possibleTokens=[]
    for i in range(len(tokens_to_keep)):
        if(tVar[i]>0.0):
            nonZeroToken=possibleToken(tokens_to_keep[i].get_token(),
                                       tVar[i],
                                       tokens_to_keep[i].get_count(),
                                       tokens_to_keep[i].get_index()  )

            
            possibleTokens.append(nonZeroToken)
    print("create_vocab_old finished")
    
    return possibleTokens



def deterministic_rounding(possible_tokens:list[possibleToken],unique_chars:list[str] ,vocab_size:int):
    if(vocab_size<len(unique_chars)):
        raise(ValueError( "Number of unique characters is greater than vocab size "))
    sorted_tokens=sorted(possible_tokens, key=lambda obj: obj.lp_value, reverse=True)

    tokens_to_choose=vocab_size-len(unique_chars)

    chosen_tokens=[token.token for token in sorted_tokens[0:tokens_to_choose]]

    tokens=list(set(unique_chars+chosen_tokens))


    return tokens


def biased_rounding(possible_tokens:list[possibleToken],unique_chars:list[str] ,vocab_size:int):
    if(vocab_size<len(unique_chars)):
        raise(ValueError( "Number of unique characters is greater than vocab size "))

    tokens_to_consider=[token for token in possible_tokens if token.lp_value>0]
    sorted_tokens=sorted(tokens_to_consider, key=lambda obj: obj.lp_value/len(obj.token), reverse=True)

    tokens_to_choose=vocab_size-len(unique_chars)
    chosen_tokens=[token.token for token in sorted_tokens[0:tokens_to_choose]]

    tokens=list(set(unique_chars+chosen_tokens))


    return tokens

def probabilistic_rounding(possible_tokens: list, unique_chars: list[str], vocab_size: int):
    if vocab_size < len(unique_chars):
        raise ValueError("Number of unique characters is greater than vocab size.")

    # Tokens that are always taken
    always_taking = [token.token for token in possible_tokens if token.lp_value > 0.99]

    # All candidate tokens (excluding those already taken)
    candidate_tokens = [token for token in possible_tokens 
                        if token.token not in always_taking]

    # If there are not enough tokens to sample, raise error
    remaining_budget = vocab_size - len(unique_chars)
    if len(always_taking) > remaining_budget:
        raise ValueError("Too many always-taking tokens to fit in vocabulary.")

    # Adjust remaining budget
    remaining_budget -= len(always_taking)

    # Get tokens and their associated probabilities
    token_list = [token.token for token in candidate_tokens]
    lp_values = np.array([token.lp_value for token in candidate_tokens])

    if len(lp_values) == 0 and remaining_budget > 0:
        raise ValueError("No available tokens to sample from.")
        
    # Weighted sampling without replacement via the Gumbel-top-k trick
    # (equivalent to Efraimidis-Spirakis). O(N) instead of O(N^2) that
    # np.random.choice(replace=False, p=...) incurs for large N.
    if remaining_budget > 0:
        log_w = np.log(lp_values)
        gumbel = -np.log(-np.log(np.random.random(len(lp_values))))
        keys = log_w + gumbel
        top_idx = np.argpartition(-keys, remaining_budget - 1)[:remaining_budget]
        sampled_tokens = [token_list[i] for i in top_idx]
    else:
        sampled_tokens = []

    # Final vocabulary
    final_vocab = list(set(unique_chars) | set(always_taking) | set(sampled_tokens))

    # Sanity check
    if len(final_vocab) != vocab_size:
        raise ValueError(f"Final vocabulary size {len(final_vocab)} does not match expected size {vocab_size}.")

    return final_vocab


def fill_missing_edges_with_unk(edges: list[tokenInstance], num_vertices: int, unk_token:str,unk_id: int ):
   
    # Keep only direct edges i -> i+1
    direct_edges = {(e.start, e.end): e for e in edges if e.end == e.start + 1}

    
    result_edges = edges

    # Walk through consecutive vertices
    for i in range(num_vertices - 1):
        if (i, i + 1) in direct_edges:
            # Keep the original edge
            result_edges.append(direct_edges[(i, i + 1)])
        else:
            # Insert UNK edge for missing step
            unk_edge = tokenInstance(
                token=unk_token,
                start=i,
                end=i + 1,
                token_index=unk_id
            )
            result_edges.append(unk_edge)

    return result_edges


    
