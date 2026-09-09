from lp_tokenizer.lp_tokenizer import BYTE_LEVEL_ALPHABET, Tokenizer
from lp_tokenizer.celex import default_celex_dir, validate_morphology_rho
from transformers import AutoTokenizer
from datasets import Value, concatenate_datasets, load_dataset, load_from_disk
from tokenizers import Regex
from tokenizers.pre_tokenizers import ByteLevel, Sequence, Split
from lp_tokenizer.lp_functions import solve_vocab_on_model
import pickle
import os
import multiprocessing
import gc
import tempfile
from pathlib import Path
import traceback


PRETOKENIZER_MODE = os.environ.get("PRETOKENIZER_MODE", "custom").strip().lower()
_APERTUS_SPLIT_PATTERN = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n/]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)
_NANOCHAT_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
SOURCE_TEXT_COLUMN = {
    "finemath": "text",
    "fineweb": "text",
    "fineweb2": "text",
    "infimath": "text",
    "megamath": "text",
    "starcoder": "content",
}


def build_pretokenizer(mode):
    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-70m-deduped",
        revision="step3000",
    )

    if mode == "pythia":
        return tokenizer

    if mode == "split_bytelevel":
        tokenizer.backend_tokenizer.pre_tokenizer = Sequence(
            [ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=True)]
        )
        return tokenizer

    patterns = {
        "apertus": _APERTUS_SPLIT_PATTERN,
        "nanochat": _NANOCHAT_SPLIT_PATTERN,
    }

    if mode in patterns:
        tokenizer.backend_tokenizer.pre_tokenizer = Sequence(
            [
                Split(
                    pattern=Regex(patterns[mode]),
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
        f"Expected one of: pythia, split_bytelevel, apertus, nanochat"
    )


pretokenizer = (
    None
    if os.environ.get("_LP_GPU_SOLVE_WORKER") == "1"
    else build_pretokenizer(PRETOKENIZER_MODE)
)


def train_lp_tokenizer(dataset, unique_chars, vocab_size, save_dir, pretokenizer_obj,
                       special_tokens, morphology_rho=0.0, celex_dir=None,
                       vocab_utilisation_weight=0.0):
    tokenizer = Tokenizer(
        corpus=dataset,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        unique_chars=unique_chars,
        pretokenizer=pretokenizer_obj,
    )
    unmatched_report_path = (
        os.path.join(save_dir, "celex_unmatched.tsv")
        if morphology_rho > 0.0
        else None
    )
    tokens = tokenizer.make_vocab_cuopt(
        morphology_rho=morphology_rho,
        vocab_utilisation_weight=vocab_utilisation_weight,
        celex_dir=celex_dir,
        unmatched_report_path=unmatched_report_path,
    )
    file_name = os.path.join(save_dir, f"lp_tokens_{vocab_size}.pkl")
    os.makedirs(save_dir, exist_ok=True)
    with open(file_name, "wb") as f:
        pickle.dump(tokens, f)


def print_lp_variable_counts(vocab_size, x_values, cuopt_model):
    num_f = int(cuopt_model["num_f"])
    num_g = int(cuopt_model["num_g"])
    num_t = int(cuopt_model["num_t"])
    expected_total = num_f + num_g + num_t

    if len(x_values) != expected_total:
        print(
            f"[lp-variable-counts] WARNING: vocab_size={vocab_size} "
            f"has {len(x_values)} x_values but expected {expected_total} "
            f"(num_f={num_f}, num_g={num_g}, num_t={num_t})"
        )

    f_values = x_values[:num_f]
    g_values = x_values[num_f:num_f + num_g]
    t_values = x_values[num_f + num_g:num_f + num_g + num_t]

    print(f"[lp-variable-counts] vocab_size={vocab_size}")
    for name, values in (
        ("f", f_values),
        ("g", g_values),
        ("t", t_values),
        ("all", x_values),
    ):
        print(
            f"[lp-variable-counts] {name}: total={len(values)} "
            f">0.999={int((values > 0.999).sum())} "
            f">0.001={int((values > 0.001).sum())}"
        )


def _solve_and_save_lp_vocab(
    cuopt_model,
    unique_chars,
    special_tokens,
    vocab_size,
    save_dir,
):
    print("---------------------------------------", flush=True)
    print(f"[sweep] Solving for vocab_size={vocab_size}", flush=True)
    print("---------------------------------------", flush=True)
    lp_budget = vocab_size - len(unique_chars) - len(special_tokens)
    if lp_budget <= 0:
        raise ValueError(
            f"Vocab size {vocab_size} too small: unique_chars={len(unique_chars)} "
            f"+ special_tokens={len(special_tokens)} already exceeds budget."
        )
    print(
        f"[solve_for_vocab_size] vocab_size={vocab_size} lp_budget={lp_budget}",
        flush=True,
    )
    result = solve_vocab_on_model(
        cuopt_model, numAllowedTokens=lp_budget, vocab_size=vocab_size,
    )
    tokens = {
        "possible_tokens": result["possible_tokens"],
        "unique_chars": unique_chars,
        "special_tokens": special_tokens,
        "x_values": result["x_values"],
    }
    print_lp_variable_counts(vocab_size, tokens["x_values"], cuopt_model)
    # Drop x_values before pickling to keep output shape identical to the
    # single-size path.
    tokens.pop("x_values", None)
    file_name = os.path.join(save_dir, f"lp_tokens_{vocab_size}.pkl")
    with open(file_name, "wb") as f:
        pickle.dump(tokens, f)
    print(f"[sweep] Saved vocabulary output to {file_name}", flush=True)


def _solve_lp_gpu_worker(
    prepared_model_path,
    save_dir,
    cuda_device,
    task_queue,
    status_connection,
):
    """Solve vocabulary budgets on one GPU in a spawned worker process."""
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    try:
        print(
            f"[sweep-worker] pid={os.getpid()} using "
            f"CUDA_VISIBLE_DEVICES={cuda_device}",
            flush=True,
        )
        with open(prepared_model_path, "rb") as model_file:
            payload = pickle.load(model_file)
        cuopt_model = payload["cuopt_model"]
        unique_chars = payload["unique_chars"]
        special_tokens = payload["special_tokens"]
        del payload
        while True:
            vocab_size = task_queue.get()
            if vocab_size is None:
                break
            _solve_and_save_lp_vocab(
                cuopt_model,
                unique_chars,
                special_tokens,
                vocab_size,
                save_dir,
            )
        status_connection.send(None)
    except BaseException:
        status_connection.send(traceback.format_exc())
        raise
    finally:
        status_connection.close()


def _visible_cuda_devices(requested_count):
    configured = os.environ.get("CUDA_VISIBLE_DEVICES")
    if configured is None:
        devices = [str(device_index) for device_index in range(requested_count)]
    else:
        devices = [device.strip() for device in configured.split(",") if device.strip()]
        if len(devices) < requested_count:
            raise ValueError(
                f"LP_NUM_GPUS={requested_count}, but CUDA_VISIBLE_DEVICES exposes "
                f"only {len(devices)} device(s): {configured!r}"
            )
    return devices[:requested_count]


def train_lp_tokenizer_sweep(dataset, unique_chars, vocab_sizes, save_dir,
                             pretokenizer_obj, special_tokens,
                             morphology_rho=0.0, celex_dir=None,
                             vocab_utilisation_weight=0.0):
    if not vocab_sizes:
        return

    # Build the LP once with the largest vocab size so the lp_budget > 0 check
    # in the Tokenizer holds for every entry in the sweep.
    sorted_sizes = sorted(set(int(vs) for vs in vocab_sizes))

    tokenizer = Tokenizer(
        corpus=dataset,
        vocab_size=sorted_sizes[-1],
        special_tokens=special_tokens,
        unique_chars=unique_chars,
        pretokenizer=pretokenizer_obj,
    )
    print(
        f"[pipeline] Tokenizer initialized: rows={len(dataset):,}, "
        f"vocab_sizes={sorted_sizes}"
    )
    unmatched_report_path = (
        os.path.join(save_dir, "celex_unmatched.tsv")
        if morphology_rho > 0.0
        else None
    )
    tokenizer.prepare_cuopt_model(
        morphology_rho=morphology_rho,
        vocab_utilisation_weight=vocab_utilisation_weight,
        celex_dir=celex_dir,
        unmatched_report_path=unmatched_report_path,
    )

    os.makedirs(save_dir, exist_ok=True)
    configured_gpu_count = int(os.environ.get("LP_NUM_GPUS", "1"))
    if configured_gpu_count <= 0:
        raise ValueError(f"LP_NUM_GPUS must be positive, got {configured_gpu_count}")
    worker_count = min(configured_gpu_count, len(sorted_sizes))

    if worker_count == 1:
        for vs in sorted_sizes:
            _solve_and_save_lp_vocab(
                tokenizer._cuopt_model,
                tokenizer.unique_chars,
                list(tokenizer.special_tokens_list),
                vs,
                save_dir,
            )
        return

    cuda_devices = _visible_cuda_devices(worker_count)
    print(
        f"[sweep] Solving {len(sorted_sizes)} vocabulary budgets across "
        f"{worker_count} GPUs: {cuda_devices}",
        flush=True,
    )
    # CUDA and RMM cannot safely initialize in a process created with fork.
    # Spawn gives each GPU worker a fresh interpreter and CUDA runtime.
    context = multiprocessing.get_context("spawn")
    task_queue = context.Queue()
    workers = []
    status_connections = []
    configured_tmp_dir = os.environ.get("TMPDIR")
    model_tmp_dir = (
        configured_tmp_dir
        if configured_tmp_dir and os.path.isdir(configured_tmp_dir)
        else None
    )
    model_file = tempfile.NamedTemporaryFile(
        mode="wb",
        prefix="lp_sweep_model_",
        suffix=".pkl",
        dir=model_tmp_dir,
        delete=False,
    )
    prepared_model_path = model_file.name
    try:
        print(
            f"[sweep] Serializing prepared LP model once to "
            f"{prepared_model_path}",
            flush=True,
        )
        with model_file:
            pickle.dump(
                {
                    "cuopt_model": tokenizer._cuopt_model,
                    "unique_chars": tokenizer.unique_chars,
                    "special_tokens": list(tokenizer.special_tokens_list),
                },
                model_file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        tokenizer._cuopt_model = None
        gc.collect()

        original_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        original_worker_marker = os.environ.get("_LP_GPU_SOLVE_WORKER")
        try:
            for cuda_device in cuda_devices:
                parent_connection, child_connection = context.Pipe(duplex=False)
                process = context.Process(
                    target=_solve_lp_gpu_worker,
                    args=(
                        prepared_model_path,
                        save_dir,
                        cuda_device,
                        task_queue,
                        child_connection,
                    ),
                )
                # Pin the environment before spawn so CUDA-aware imports during
                # interpreter bootstrap also see exactly one GPU.
                os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
                os.environ["_LP_GPU_SOLVE_WORKER"] = "1"
                process.start()
                child_connection.close()
                workers.append(process)
                status_connections.append(parent_connection)
        finally:
            if original_visible_devices is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_visible_devices
            if original_worker_marker is None:
                os.environ.pop("_LP_GPU_SOLVE_WORKER", None)
            else:
                os.environ["_LP_GPU_SOLVE_WORKER"] = original_worker_marker

        for vocab_size in reversed(sorted_sizes):
            task_queue.put(vocab_size)
        for _ in workers:
            task_queue.put(None)

        for process in workers:
            process.join()
    except BaseException:
        for process in workers:
            if process.is_alive():
                process.terminate()
        for process in workers:
            process.join()
        for connection in status_connections:
            connection.close()
        raise
    finally:
        task_queue.close()
        task_queue.join_thread()
        if not model_file.closed:
            model_file.close()
        if os.path.exists(prepared_model_path):
            os.remove(prepared_model_path)

    failures = []
    for process, connection in zip(workers, status_connections):
        message = connection.recv() if connection.poll() else None
        connection.close()
        if process.exitcode != 0:
            failures.append(
                message
                or f"GPU worker pid={process.pid} exited with code {process.exitcode}"
            )
    if failures:
        raise RuntimeError(
            "One or more parallel LP solve workers failed:\n" + "\n".join(failures)
        )


def infer_text_column(dataset):
    preferred_columns = ("text", "content", "code")
    for column in preferred_columns:
        if column in dataset.column_names:
            return column

    for name, feature in dataset.features.items():
        dtype = getattr(feature, "dtype", None)
        if dtype in {"string", "large_string"}:
            return name

    raise ValueError(f"Could not infer text column from columns: {dataset.column_names}")


def normalize_to_text_column(dataset, source_name=None):
    preferred_text_column = SOURCE_TEXT_COLUMN.get(source_name) if source_name else None
    if preferred_text_column in dataset.column_names:
        text_column = preferred_text_column
    else:
        text_column = infer_text_column(dataset)
        if preferred_text_column is not None and preferred_text_column != text_column:
            print(
                f"[WARN] Source '{source_name}' expected text column '{preferred_text_column}' "
                f"but using inferred column '{text_column}'."
            )

    if text_column != "text":
        dataset = dataset.rename_column(text_column, "text")
    columns_to_remove = [column for column in dataset.column_names if column != "text"]
    if columns_to_remove:
        dataset = dataset.remove_columns(columns_to_remove)
    text_dtype = getattr(dataset.features.get("text"), "dtype", None)
    if text_dtype != "string":
        dataset = dataset.cast_column("text", Value("string"))
    return dataset


def load_training_dataset(path):
    try:
        dataset_obj = load_from_disk(path)
        if hasattr(dataset_obj, "keys"):
            if "train" in dataset_obj:
                dataset = dataset_obj["train"]
            else:
                raise ValueError(
                    f"DatasetDict at {path} does not contain a 'train' split. "
                    f"Available splits: {list(dataset_obj.keys())}"
                )
        else:
            dataset = dataset_obj

        print("Loaded dataset using load_from_disk")
        return normalize_to_text_column(dataset)
    except Exception as load_from_disk_error:
        print(
            "[INFO] load_from_disk failed "
            f"({type(load_from_disk_error).__name__}: {load_from_disk_error}). "
            "Trying parquet-based loading."
        )
        base_path = Path(path)
        if not base_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {path}") from load_from_disk_error

        source_dirs = sorted(entry for entry in base_path.iterdir() if entry.is_dir())
        source_datasets = []

        for source_dir in source_dirs:
            parquet_files = sorted(str(parquet_path) for parquet_path in source_dir.rglob("*.parquet"))
            if not parquet_files:
                continue

            try:
                source_chunks = []
                total_rows = 0
                for index, parquet_file in enumerate(parquet_files, start=1):
                    source_chunk = load_dataset("parquet", data_files=parquet_file, split="train")
                    source_chunk = normalize_to_text_column(source_chunk, source_name=source_dir.name)
                    source_chunks.append(source_chunk)
                    total_rows += len(source_chunk)
                    if index % 200 == 0 or index == len(parquet_files):
                        print(
                            f"Source '{source_dir.name}': loaded {index}/{len(parquet_files)} parquet files"
                        )

                if len(source_chunks) == 1:
                    source_dataset = source_chunks[0]
                else:
                    source_dataset = concatenate_datasets(source_chunks)
                source_datasets.append(source_dataset)
                print(
                    f"Loaded source '{source_dir.name}' via parquet "
                    f"({len(parquet_files)} files, {total_rows} rows)"
                )
            except Exception as source_error:
                print(
                    f"[ERROR] Failed loading source '{source_dir.name}' "
                    f"with {len(parquet_files)} parquet files."
                )
                print(f"[ERROR] Exception type: {type(source_error).__name__}")
                print(f"[ERROR] Exception message: {source_error}")
                print("[ERROR] Traceback:")
                print(traceback.format_exc())
                raise RuntimeError(
                    f"Parquet source load failed for '{source_dir.name}'. "
                    f"First file: {parquet_files[0]}"
                ) from source_error

        if source_datasets:
            return concatenate_datasets(source_datasets)

        parquet_files = sorted(str(parquet_path) for parquet_path in base_path.rglob("*.parquet"))
        if not parquet_files:
            raise RuntimeError(
                f"Failed to load as Dataset/DatasetDict and found no parquet files under: {path}"
            ) from load_from_disk_error

        print(f"Falling back to recursive parquet load ({len(parquet_files)} files)")
        try:
            dataset = load_dataset("parquet", data_files=parquet_files, split="train")
            return normalize_to_text_column(dataset)
        except Exception as parquet_error:
            print("[ERROR] Recursive parquet fallback load failed.")
            print(f"[ERROR] Exception type: {type(parquet_error).__name__}")
            print(f"[ERROR] Exception message: {parquet_error}")
            print("[ERROR] Traceback:")
            print(traceback.format_exc())
            raise


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

_SPECIAL_TOKENS_BY_MODE = {
    "pythia": APERTUS_SPECIAL_TOKENS,
    "split_bytelevel": APERTUS_SPECIAL_TOKENS,
    "apertus": APERTUS_SPECIAL_TOKENS,
    "nanochat": NANOCHAT_SPECIAL_TOKENS,
}


def get_special_tokens(mode):
    if mode not in _SPECIAL_TOKENS_BY_MODE:
        raise ValueError(
            f"No special tokens defined for PRETOKENIZER_MODE='{mode}'. "
            f"Expected one of: {list(_SPECIAL_TOKENS_BY_MODE.keys())}"
        )
    return _SPECIAL_TOKENS_BY_MODE[mode]


if __name__ == "__main__":
    TRAIN_DATASET_PATH = os.environ.get(
        "TRAIN_DATASET_PATH",
        "/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/tokenizer_training_dataset",
    )
    vocab_size = [int(size) for size in os.environ.get("VOCAB_SIZES", "131072").split(",") if size.strip()]
    save_dir = os.environ.get("RAW_VOCAB_PATH", "rounding_vocabs_apertus_2/")
    try:
        morphology_rho = validate_morphology_rho(
            float(os.environ.get("MORPHOLOGY_RHO", "0"))
        )
    except ValueError as error:
        raise ValueError(
            "MORPHOLOGY_RHO must be a finite, non-negative number."
        ) from error
    # Normalised objective: L_existing / D_V + lambda / LP_budget * sum_c (t_c - U_c / N_c).
    # Example: VOCAB_UTILISATION_WEIGHT=0.1 python train_tokenizer.py
    vocab_utilisation_weight = float(os.environ.get("VOCAB_UTILISATION_WEIGHT", "0"))
    configured_celex_dir = os.environ.get("CELEX_DIR")
    celex_dir = configured_celex_dir or str(default_celex_dir())
    special_tokens = get_special_tokens(PRETOKENIZER_MODE)
    print(f"Using PRETOKENIZER_MODE={PRETOKENIZER_MODE}")
    print(f"Special tokens ({len(special_tokens)}): {special_tokens}")
    print(f"Vocabulary utilisation weight: {vocab_utilisation_weight:g}")
    print(f"Morphology rho: {morphology_rho:g}")
    if morphology_rho > 0.0:
        print(f"CELEX directory: {celex_dir}")
    print(f"Loading training dataset from {TRAIN_DATASET_PATH}")

    dataset = load_training_dataset(TRAIN_DATASET_PATH)
    print(f"Loaded {len(dataset)} rows")

    unique_chars = list(BYTE_LEVEL_ALPHABET)
    print(f"Using fixed ByteLevel alphabet ({len(unique_chars)} symbols)")

    train_lp_tokenizer_sweep(
        dataset,
        unique_chars,
        vocab_size,
        save_dir,
        pretokenizer,
        special_tokens,
        morphology_rho=morphology_rho,
        vocab_utilisation_weight=vocab_utilisation_weight,
        celex_dir=celex_dir,
    )
