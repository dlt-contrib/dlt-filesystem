"""Reader arguments come from the caller, never from dlt config.

A transformer built without a spec resolves every defaulted argument of the function
it wraps from dlt config, and dlt's lookup falls back to bare keys. Shells and
readline-linked interpreters export `COLUMNS`, which would then reach `columns` on the
columnar readers. These tests set such keys and check that no reader sees them, while
explicit arguments, URI hints and the source's own configuration keep working.
"""

import inspect
import json
from pathlib import Path
from typing import Any

import dlt
import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from dlt.extract.exceptions import ResourceExtractionError

from dlt_filesystem.source.adapter import filesystem, readers
from dlt_filesystem.source.format import readers as reader_functions
from dlt_filesystem.source.format.registry import READER_REGISTRATIONS
from dlt_filesystem.source.fsspec.local import LocalFilesystemSource
from dlt_filesystem.testing.writer import write_avro, write_feather, write_orc

ROWS = [{"a": i, "b": f"v{i}"} for i in range(5)]

#: A value for every defaulted reader parameter, each of the parameter's type and
#: different from its default, as it would be spelled in the environment.
AMBIENT_VALUES = {
    "chunksize": "7",
    "chunk_size": "7",
    "columns": '["a"]',
    "column_names": '["x", "y"]',
    "worksheet_names": '{"sheet": ["A1", "B2"]}',
    "tagname": "item",
    "use_pyarrow": "true",
}


def _local_source(directory: Path, file_glob: str):
    return readers(
        f"file://{directory}", fsspec.filesystem("file"), file_glob=file_glob
    )


def _write_jsonl(path: Path) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in ROWS))
    return path


def _write_parquet(path: Path) -> Path:
    pq.write_table(pa.Table.from_pylist(ROWS), path)
    return path


def _defaulted(signature: inspect.Signature) -> dict[str, inspect.Parameter]:
    return {
        name: parameter
        for name, parameter in signature.parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }


@pytest.mark.parametrize(
    "reader_name", [registration.reader_name for registration in READER_REGISTRATIONS]
)
def test_no_reader_argument_resolves_from_the_environment(
    reader_name, tmp_path, monkeypatch
):
    """Every registered reader is called with its defaults, whatever the env holds.

    dlt passes resolved values positionally where it can, so the call is bound back
    onto the reader's signature rather than read from its keyword arguments.
    """
    signature = inspect.signature(getattr(reader_functions, reader_name))
    defaulted = _defaulted(signature)
    assert set(defaulted) <= set(AMBIENT_VALUES), "add the new parameter above"
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def spy(*args: Any, **kwargs: Any):
        calls.append((args, kwargs))
        yield from ()

    setattr(spy, "__signature__", signature)  # noqa: B010
    # Patched before the source is built: `readers()` looks each reader up by name.
    monkeypatch.setattr(reader_functions, reader_name, spy)
    for name in defaulted:
        monkeypatch.setenv(name.upper(), AMBIENT_VALUES[name])
    (tmp_path / "data.bin").write_bytes(b"x")

    source = _local_source(tmp_path, "*")
    list(source.resources[reader_name])

    assert calls
    for args, kwargs in calls:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        assert {name: bound.arguments[name] for name in defaulted} == {
            name: parameter.default for name, parameter in defaulted.items()
        }


COLUMNAR = [
    pytest.param("read_feather", "rows.feather", write_feather, id="feather"),
    pytest.param("read_orc", "rows.orc", write_orc, id="orc"),
    pytest.param("read_avro", "rows.avro", write_avro, id="avro"),
]


@pytest.mark.parametrize(("reader_name", "file_name", "write"), COLUMNAR)
def test_terminal_width_does_not_select_columns(
    reader_name, file_name, write, tmp_path, monkeypatch
):
    monkeypatch.setenv("COLUMNS", "80")
    path = write(tmp_path / file_name, ROWS)

    through_readers = list(_local_source(tmp_path, file_name).resources[reader_name])
    through_uri = list(LocalFilesystemSource().dlt_source(f"file://{path}", ""))

    assert through_readers == ROWS
    assert through_uri == ROWS


@pytest.mark.parametrize(
    ("reader_name", "file_name", "write"),
    [
        pytest.param("read_jsonl", "rows.jsonl", _write_jsonl, id="jsonl"),
        pytest.param("read_parquet", "rows.parquet", _write_parquet, id="parquet"),
    ],
)
def test_a_bare_chunksize_key_does_not_reach_the_reader(
    reader_name, file_name, write, tmp_path, monkeypatch
):
    """`chunksize=0` raises inside the reader, so rows prove it never arrived."""
    monkeypatch.setenv("CHUNKSIZE", "0")
    write(tmp_path / file_name)

    assert list(_local_source(tmp_path, file_name).resources[reader_name]) == ROWS


def test_a_bare_use_pyarrow_key_does_not_reach_the_reader(tmp_path, monkeypatch):
    monkeypatch.setenv("USE_PYARROW", "true")
    _write_parquet(tmp_path / "rows.parquet")

    assert list(_local_source(tmp_path, "*.parquet").read_parquet) == ROWS


def test_a_resource_section_key_does_not_reach_the_reader(tmp_path, monkeypatch):
    """Iterating the resource looks the reader up under the source's own section."""
    monkeypatch.setenv("SOURCES__READERS__READ_JSONL__CHUNKSIZE", "0")
    _write_jsonl(tmp_path / "rows.jsonl")

    assert list(_local_source(tmp_path, "*.jsonl").read_jsonl) == ROWS


def test_a_source_section_key_does_not_reach_the_reader(tmp_path, monkeypatch):
    """Iterating the source looks the reader up under the module's section."""
    monkeypatch.setenv("SOURCES__ADAPTER__READ_JSONL__CHUNKSIZE", "0")
    _write_jsonl(tmp_path / "rows.jsonl")

    source = _local_source(tmp_path, "*.jsonl").with_resources("read_jsonl")

    assert list(source) == ROWS


def test_a_columns_hint_still_selects_columns(tmp_path, monkeypatch):
    monkeypatch.setenv("COLUMNS", "80")
    path = write_feather(tmp_path / "rows.feather", ROWS)

    rows = list(LocalFilesystemSource().dlt_source(f'file://{path}#columns=["a"]', ""))

    assert rows == [{"a": row["a"]} for row in ROWS]


def test_a_columns_argument_still_selects_columns(tmp_path, monkeypatch):
    monkeypatch.setenv("COLUMNS", "80")
    write_feather(tmp_path / "rows.feather", ROWS)

    source = _local_source(tmp_path, "*.feather")

    assert list(source.read_feather(columns=["a"])) == [{"a": row["a"]} for row in ROWS]


def test_call_arguments_still_reach_the_reader(tmp_path, monkeypatch):
    monkeypatch.setenv("CHUNKSIZE", "0")
    monkeypatch.setenv("USE_PYARROW", "false")
    _write_parquet(tmp_path / "rows.parquet")

    source = _local_source(tmp_path, "*.parquet")
    batches = list(source.read_parquet(chunksize=2, use_pyarrow=True))

    assert [batch.num_rows for batch in batches] == [2, 2, 1]


@pytest.fixture
def pristine_read_jsonl():
    """Clear and afterwards restore what dlt stores on the public reader function."""
    function = reader_functions.read_jsonl
    saved = dict(function.__dict__)
    function.__dict__.clear()
    yield function
    function.__dict__.clear()
    function.__dict__.update(saved)


def test_another_transformer_over_a_reader_keeps_its_config(
    pristine_read_jsonl, tmp_path, monkeypatch
):
    """Building `readers()` must not change transformers someone else built.

    dlt stores a transformer's spec on the function it wraps and reads it back when the
    transformer is cloned, so a spec set on the public reader would reach this one.
    """
    theirs = dlt.transformer()(pristine_read_jsonl)
    _write_jsonl(tmp_path / "rows.jsonl")
    _local_source(tmp_path, "*.jsonl")
    monkeypatch.setenv("CHUNKSIZE", "0")

    clone = theirs.with_name("their_jsonl")
    files = filesystem(f"file://{tmp_path}", fsspec.filesystem("file"), file_glob="*")

    with pytest.raises(ResourceExtractionError, match="chunksize must be greater"):
        list(files | clone)


def test_the_source_still_reads_its_bucket_from_config(tmp_path, monkeypatch):
    monkeypatch.setenv("SOURCES__READERS__BUCKET_URL", f"file://{tmp_path}")
    _write_jsonl(tmp_path / "rows.jsonl")

    source = readers(credentials=fsspec.filesystem("file"), file_glob="*.jsonl")

    assert list(source.resources["read_jsonl"]) == ROWS
