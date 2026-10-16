import typing

import polars as pl
import polars._typing

from dlt_filesystem.source.format.readers import _polars_csv_symbols
from dlt_filesystem.util.python import cast_kwargs_to_signature

# Like polars itself, the stand-in below names these only for type checkers, so its string
# annotations resolve at runtime only through the symbols `_polars_csv_symbols()` supplies.
if typing.TYPE_CHECKING:
    from polars import DataFrame  # noqa: F401
    from polars._typing import CsvEncoding, SchemaDict  # noqa: F401


def _read_csv_before_polars_1_38(
    source: "str",
    *,
    has_header: "bool" = True,
    skip_rows: "int" = 0,
    encoding: "CsvEncoding | str" = "utf8",
    schema_overrides: "SchemaDict | None" = None,
) -> "DataFrame":
    """Stand-in for `polars.read_csv` whose annotations, like 1.34-1.37's, never name `StorageOptionsDict`."""
    raise NotImplementedError


def test_csv_hints_cast_on_a_polars_without_storage_options_dict(monkeypatch):
    """polars 1.34-1.37 neither define `StorageOptionsDict` nor reference it from `read_csv`."""
    monkeypatch.delattr(polars._typing, "StorageOptionsDict", raising=False)

    symbols = _polars_csv_symbols()

    assert "StorageOptionsDict" not in symbols
    casted = cast_kwargs_to_signature(
        _read_csv_before_polars_1_38,
        {"has_header": "false", "skip_rows": "2"},
        symbols=symbols,
    )
    assert casted == {"has_header": False, "skip_rows": 2}


def test_csv_symbols_resolve_the_installed_read_csv_signature():
    """Every name the installed `polars.read_csv` annotations reference is supplied."""
    symbols = _polars_csv_symbols()

    assert ("StorageOptionsDict" in symbols) == hasattr(
        polars._typing, "StorageOptionsDict"
    )
    typing.get_type_hints(pl.read_csv, localns=symbols)
