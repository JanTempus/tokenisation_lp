from collections import defaultdict
from pathlib import Path
import argparse

import pyarrow as pa
import pyarrow.parquet as pq


PREFERRED_TEXT_COLUMNS = ("text", "content", "code")


def schema_signature(schema):
    return tuple((field.name, str(field.type)) for field in schema)


def schema_column_names(schema):
    return [field.name for field in schema]


def schema_string_columns(schema):
    return [
        field.name
        for field in schema
        if pa.types.is_string(field.type) or pa.types.is_large_string(field.type)
    ]


def choose_schema_text_column(schema):
    names = set(schema_column_names(schema))
    for preferred in PREFERRED_TEXT_COLUMNS:
        if preferred in names:
            return preferred

    string_cols = schema_string_columns(schema)
    if string_cols:
        return string_cols[0]
    return None


def inspect_source(source_dir):
    parquet_files = sorted(source_dir.rglob("*.parquet"))
    print(f"\n=== {source_dir.name} ===")
    print(f"files: {len(parquet_files)}")

    if not parquet_files:
        print("no parquet files found")
        return

    signatures = defaultdict(list)
    schema_by_signature = {}

    for parquet_file in parquet_files:
        schema = pq.read_schema(parquet_file)
        sig = schema_signature(schema)
        signatures[sig].append(parquet_file)
        schema_by_signature[sig] = schema

    print(f"distinct_schema_variants: {len(signatures)}")

    signature_items = sorted(signatures.items(), key=lambda item: len(item[1]), reverse=True)

    common_columns = None
    common_string_columns = None

    for index, (sig, files) in enumerate(signature_items, start=1):
        schema = schema_by_signature[sig]
        col_names = schema_column_names(schema)
        string_cols = schema_string_columns(schema)
        recommended = choose_schema_text_column(schema)

        print(f"\n  [schema #{index}] files={len(files)} sample={files[0]}")
        print("  columns:")
        for field in schema:
            print(f"    - {field.name}: {field.type}")
        print(f"  string_columns: {string_cols}")
        print(f"  suggested_text_column_for_schema: {recommended}")

        cols_set = set(col_names)
        str_set = set(string_cols)
        common_columns = cols_set if common_columns is None else common_columns & cols_set
        common_string_columns = str_set if common_string_columns is None else common_string_columns & str_set

    if common_columns is None:
        common_columns = set()
    if common_string_columns is None:
        common_string_columns = set()

    global_recommended = None
    for preferred in PREFERRED_TEXT_COLUMNS:
        if preferred in common_columns:
            global_recommended = preferred
            break
    if global_recommended is None and common_string_columns:
        global_recommended = sorted(common_string_columns)[0]

    print("\n  summary:")
    print(f"    common_columns_across_all_files: {sorted(common_columns)}")
    print(f"    common_string_columns_across_all_files: {sorted(common_string_columns)}")
    print(f"    suggested_text_column_for_source: {global_recommended}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect parquet schemas under tokenizer_training_dataset and suggest text-like columns."
    )
    parser.add_argument(
        "--base-path",
        default="/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/tokenizer_training_dataset",
        help="Root directory containing source subfolders with parquet files.",
    )
    args = parser.parse_args()

    base_path = Path(args.base_path)
    if not base_path.exists():
        raise FileNotFoundError(f"Base path not found: {base_path}")

    subdirs = sorted(path for path in base_path.iterdir() if path.is_dir())
    if not subdirs:
        raise RuntimeError(f"No subdirectories found under: {base_path}")

    print(f"base_path: {base_path}")
    print(f"sources_found: {[path.name for path in subdirs]}")

    for subdir in subdirs:
        inspect_source(subdir)


if __name__ == "__main__":
    main()
