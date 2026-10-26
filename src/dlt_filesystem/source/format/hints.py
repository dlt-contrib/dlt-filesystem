"""Validate the hint namespace before binding a filesystem reader."""

import inspect
from typing import Any, Callable, Mapping

# DuckDB's Python CSV API exposes **kwargs rather than named parameters. Keep its
# accepted options explicit, including Python aliases such as skiprows and dtype.
_DUCKDB_CSV_HINTS = frozenset(
    "header strict_mode compression comment sep delimiter files_to_sniff dtype "
    "na_values skiprows quotechar escapechar encoding parallel date_format "
    "timestamp_format sample_size auto_detect all_varchar normalize_names "
    "null_padding names lineterminator columns auto_type_candidates max_line_size "
    "ignore_errors store_rejects rejects_table rejects_scan rejects_limit "
    "force_not_null buffer_size decimal allow_quoted_nulls filename "
    "hive_partitioning union_by_name hive_types hive_types_autocast thousands".split()
)


def _keyword_names(function: Callable) -> set[str]:
    return {
        name
        for name, parameter in inspect.signature(function).parameters.items()
        if parameter.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }


def prepare_reader_hints(reader_name: str, hints: Mapping[str, Any]) -> dict[str, Any]:
    """Reject unknown arguments before resource binding.

    Polars readers expose backend keywords as hints. XML and YAML expose only
    their declared options. The filesystem supplies input streams and worksheet
    collision state, so those internal parameters are never hint names.
    """
    from dlt_filesystem.source.format import readers

    accepted = _keyword_names(getattr(readers, reader_name))
    if reader_name in {"read_csv", "read_csv_headless", "read_excel", "read_ods"}:
        import polars as pl

        backend = {
            "read_csv": pl.read_csv,
            "read_csv_headless": pl.read_csv,
            "read_excel": pl.read_excel,
            "read_ods": pl.read_ods,
        }[reader_name]
        accepted.update(_keyword_names(backend) - {"source"})
        if reader_name in {"read_csv", "read_csv_headless"}:
            # This legacy Polars read-buffer option is ignored on Polars 2.
            accepted.add("batch_size")
    elif reader_name == "read_csv_duckdb":
        accepted.update(_DUCKDB_CSV_HINTS)
    accepted.difference_update({"items", "worksheet_names"})
    unknown = hints.keys() - accepted
    if unknown:
        first = sorted(unknown)[0]
        raise TypeError(
            f"{reader_name}(): got an unexpected keyword argument {first!r}. "
            f"Unknown reader hints: {', '.join(sorted(unknown))}. "
            f"Accepted hints: {', '.join(sorted(accepted))}"
        )
    return dict(hints)
