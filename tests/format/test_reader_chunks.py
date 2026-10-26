"""Chunk bounds on direct readers and on URI-fragment hints."""

import inspect
from io import BytesIO
from typing import Any

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dlt_filesystem.source.adapter import readers
from dlt_filesystem.source.format import readers as formats
from dlt_filesystem.source.format.registry import READER_REGISTRATIONS
from dlt_filesystem.source.fsspec.local import LocalFilesystemSource
from dlt_filesystem.testing.stub import FileItemStub
from dlt_filesystem.testing.writer import write_avro

CHUNKED_READERS = tuple(
    entry.reader_name
    for entry in READER_REGISTRATIONS
    if "chunksize" in inspect.signature(getattr(formats, entry.reader_name)).parameters
    or entry.reader_name == "read_csv_duckdb"
)


def _items(item) -> Any:
    """Pass a file-item double through the readers' FileItemDict annotation."""
    return iter([item])


def _write_rows(path, name):
    rows = [{"id": i} for i in range(5)]
    frame = pl.DataFrame(rows)
    if name == "read_csv_headless":
        path.write_text("0\n1\n2\n3\n4\n")
    elif name in {"read_csv", "read_csv_duckdb"}:
        frame.write_csv(path)
    elif name == "read_jsonl":
        frame.write_ndjson(path)
    elif name == "read_json":
        path.write_text('[{"id":0},{"id":1},{"id":2},{"id":3},{"id":4}]')
    elif name == "read_bson":
        bson = pytest.importorskip("bson")
        path.write_bytes(b"".join(bson.BSON.encode(row) for row in rows))
    elif name == "read_msgpack":
        msgpack = pytest.importorskip("msgpack")
        path.write_bytes(b"".join(msgpack.packb(row) for row in rows))
    elif name == "read_cbor":
        path.write_bytes(pytest.importorskip("cbor2").dumps(rows))
    elif name == "read_yaml":
        path.write_text(pytest.importorskip("yaml").safe_dump(rows))
    elif name == "read_xml":
        pytest.importorskip("lxml")
        path.write_text(
            "<data>"
            + "".join(f"<item><id>{i}</id></item>" for i in range(5))
            + "</data>"
        )
    elif name == "read_avro":
        write_avro(path, rows)
    elif name == "read_orc":
        from pyarrow import orc

        orc.write_table(pa.Table.from_pylist(rows), path)
    elif name == "read_feather":
        from pyarrow import feather

        feather.write_feather(pa.Table.from_pylist(rows), path)
    elif name == "read_parquet":
        pq.write_table(pa.Table.from_pylist(rows), path)
    elif name == "read_vortex":
        vx = pytest.importorskip("vortex")
        vx.io.write(pa.Table.from_pylist(rows), str(path))
    else:
        raise AssertionError(f"Missing fixture for {name}")


@pytest.mark.parametrize("reader_name", CHUNKED_READERS)
def test_string_chunksize_on_direct_transformer(tmp_path, reader_name):
    path = tmp_path / "data.bin"
    _write_rows(path, reader_name)
    hints = {"chunksize": "2"}
    if reader_name == "read_xml":
        hints["tagname"] = "item"
    source = readers(str(tmp_path), file_glob=path.name)
    # dlt flattens row lists, so use the underlying generator for chunk boundaries.
    direct = getattr(formats, reader_name)
    chunks = list(direct(_items(FileItemStub(path)), **hints))
    assert [len(chunk) for chunk in chunks] == [2, 2, 1]
    assert len(list(getattr(source, reader_name)(**hints))) == 5


@pytest.mark.parametrize("reader_name", CHUNKED_READERS)
def test_uri_chunksize_preserves_all_rows(tmp_path, reader_name):
    path = tmp_path / "data.bin"
    _write_rows(path, reader_name)
    fmt = next(
        e.format_keys[0] for e in READER_REGISTRATIONS if e.reader_name == reader_name
    )
    fragment = f"#{fmt}&chunksize=2"
    if reader_name == "read_xml":
        fragment += "&tagname=item"
    result = list(LocalFilesystemSource().dlt_source(f"file://{path}{fragment}", ""))
    assert len(result) == 5
    assert [int(next(iter(row.values()))) for row in result] == list(range(5))


@pytest.mark.parametrize("reader_name", CHUNKED_READERS)
@pytest.mark.parametrize("value", ["0", "-2", "nope", "2.5", 1.5, True, None])
def test_invalid_chunk_size_is_rejected_without_reading(reader_name, value):
    if reader_name == "read_csv_duckdb" and value is None:
        pytest.skip("None leaves the optional alias unset")
    hints = {"chunksize": value}
    if reader_name == "read_xml":
        hints["tagname"] = "item"
    with pytest.raises((TypeError, ValueError), match="chunksize"):
        list(getattr(formats, reader_name)(iter(()), **hints))


@pytest.mark.parametrize("names", [None, ["a", "b", "c"]])
def test_headless_separator_without_rewinding(names):
    class NoRewind(BytesIO):
        def seek(self, *args):
            raise AssertionError("Headerless CSV must not rewind for a probe read")

    class Item:
        def open(self):
            return NoRewind(b"1;2;3\n4;5;6\n7;8;9\n")

    chunks = list(
        formats.read_csv_headless(
            _items(Item()),
            chunksize=2,
            column_names=names,
            separator=";",
        )
    )
    keys = names or ["unknown_col_0", "unknown_col_1", "unknown_col_2"]
    assert chunks == [
        [dict(zip(keys, row)) for row in [(1, 2, 3), (4, 5, 6)]],
        [dict(zip(keys, (7, 8, 9)))],
    ]


def test_parquet_default_and_arrow_batches(tmp_path):
    path = tmp_path / "data.parquet"
    pq.write_table(pa.table({"id": range(2001)}), path)
    batches: list[Any] = list(
        formats.read_parquet(_items(FileItemStub(path)), use_pyarrow=True)
    )
    assert all(isinstance(batch, pa.RecordBatch) for batch in batches)
    assert [batch.num_rows for batch in batches] == [1000, 1000, 1]
    assert [row for batch in batches for row in batch.to_pylist()] == [
        {"id": i} for i in range(2001)
    ]


@pytest.mark.parametrize("hint, arrow", [("true", True), ("false", False)])
def test_parquet_arrow_hint(tmp_path, hint, arrow):
    path = tmp_path / "data.parquet"
    pq.write_table(pa.table({"id": range(5)}), path)
    result = list(
        LocalFilesystemSource().dlt_source(
            f"file://{path}#chunksize=2&use_pyarrow={hint}", ""
        )
    )
    if arrow:
        assert [batch.num_rows for batch in result] == [2, 2, 1]
        assert all(isinstance(batch, pa.RecordBatch) for batch in result)
    else:
        assert result == [{"id": i} for i in range(5)]


@pytest.mark.parametrize("keyword", ["chunksize", "chunk_size"])
def test_duckdb_chunk_spellings(tmp_path, keyword):
    path = tmp_path / "rows.csv"
    path.write_text("id,name\n0,a\n1,b\n2,c\n3,d\n4,e\n")
    hints: dict[str, Any] = {keyword: "2"}
    chunks = list(formats.read_csv_duckdb(_items(FileItemStub(path)), **hints))
    assert [len(chunk) for chunk in chunks] == [2, 2, 1]
    assert [row for chunk in chunks for row in chunk] == [
        {"id": i, "name": c} for i, c in enumerate("abcde")
    ]


@pytest.mark.parametrize("chunk_size", [1, 50_000, None])
def test_duckdb_chunksize_takes_precedence(tmp_path, chunk_size):
    path = tmp_path / "rows.csv"
    path.write_text("id\n0\n1\n2\n3\n4\n")
    chunks = list(
        formats.read_csv_duckdb(
            _items(FileItemStub(path)),
            chunk_size=chunk_size,
            chunksize=2,
        )
    )
    assert [len(chunk) for chunk in chunks] == [2, 2, 1]
    assert list(
        LocalFilesystemSource().dlt_source(
            f"file://{path}#csv_duckdb&chunk_size=2&chunksize=3", ""
        )
    ) == [{"id": i} for i in range(5)]


def test_csv_backend_options_and_chunking(tmp_path):
    path = tmp_path / "rows.csv"
    path.write_text("a;b\n1;11\n2;12\n3;13\n4;14\n")
    chunks = list(
        formats.read_csv(
            _items(FileItemStub(path)),
            chunksize=2,
            separator=";",
            skip_rows="1",
            has_header="false",
            n_rows="3",
            new_columns='["id","value"]',
        )
    )
    assert chunks == [
        [{"id": 1, "value": 11}, {"id": 2, "value": 12}],
        [{"id": 3, "value": 13}],
    ]


def test_headless_column_names_apply_before_projection(tmp_path):
    path = tmp_path / "rows.csv"
    path.write_text("1;2;3\n4;5;6\n")
    assert list(
        formats.read_csv_headless(
            _items(FileItemStub(path)),
            column_names=["a", "b", "c"],
            separator=";",
            columns=["c", "a"],
            chunksize=1,
        )
    ) == [[{"c": 3, "a": 1}], [{"c": 6, "a": 4}]]


@pytest.mark.parametrize("headless", [True, False])
def test_typed_polars_hints_are_preserved(tmp_path, headless):
    path = tmp_path / "rows.csv"
    path.write_text("1;2\n3;4\n")
    reader = formats.read_csv_headless if headless else formats.read_csv
    chunks = list(
        reader(
            _items(FileItemStub(path)),
            chunksize=1,
            separator=";",
            has_header=False,
            columns=[0],
            n_rows=1,
        )
    )
    assert len(chunks) == len(chunks[0]) == 1
    assert list(chunks[0][0].values()) == [1]
    if headless:
        assert chunks[0][0] == {"unknown_col_0": 1}


@pytest.mark.parametrize("row_index", [None, "rownum"])
def test_headless_positional_projection(tmp_path, row_index):
    path = tmp_path / "rows.csv"
    path.write_text("1;2;3\n4;5;6\n")
    chunks = list(
        formats.read_csv_headless(
            _items(FileItemStub(path)),
            separator=";",
            new_columns=["id", "a", "b"],
            columns=[0],
            chunksize=1,
            row_index_name=row_index,
        )
    )
    expected = [{"id": 1}, {"id": 4}]
    if row_index:
        expected = [{row_index: i, **row} for i, row in enumerate(expected)]
    assert chunks == [[row] for row in expected]


def test_headless_schema_supplies_column_names(tmp_path):
    path = tmp_path / "rows.csv"
    path.write_text("1,2\n3,4\n")
    assert list(
        formats.read_csv_headless(
            _items(FileItemStub(path)),
            chunksize=1,
            schema={"id": pl.Int64, "value": pl.Int64},
        )
    ) == [[{"id": 1, "value": 2}], [{"id": 3, "value": 4}]]


@pytest.mark.parametrize("metadata_name", ["path", "column_99"])
def test_headless_preserves_file_path_metadata_name(tmp_path, metadata_name):
    path = tmp_path / "rows.csv"
    path.write_text("1,2\n3,4\n")
    chunks = list(
        formats.read_csv_headless(
            _items(FileItemStub(path)),
            chunksize=1,
            include_file_paths=metadata_name,
        )
    )
    rows = [row for chunk in chunks for row in chunk]
    assert [list(row) for row in rows] == [
        ["unknown_col_0", "unknown_col_1", metadata_name],
        ["unknown_col_0", "unknown_col_1", metadata_name],
    ]
    assert [row["unknown_col_0"] for row in rows] == [1, 3]
    assert [row["unknown_col_1"] for row in rows] == [2, 4]


@pytest.mark.parametrize("index_name", ["column_0", "column_1"])
def test_headless_index_can_use_an_autogenerated_data_name(tmp_path, index_name):
    path = tmp_path / "rows.csv"
    path.write_text("1,2\n3,4\n")
    assert list(
        formats.read_csv_headless(
            _items(FileItemStub(path)),
            chunksize=1,
            row_index_name=index_name,
            row_index_offset=2,
        )
    ) == [
        [{index_name: 2, "unknown_col_0": 1, "unknown_col_1": 2}],
        [{index_name: 3, "unknown_col_0": 3, "unknown_col_1": 4}],
    ]
