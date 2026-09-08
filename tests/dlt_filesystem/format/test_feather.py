"""Feather V2 (Arrow IPC file) read and write.

Two things here are not shared with the ORC suite this file otherwise mirrors. The
chunking test is instrumented rather than counted: a reader that materialized the whole
table and sliced it afterwards produces the same chunk lengths as one that reads batch by
batch, so the batches actually requested are recorded and `read_all` is forbidden. And
Feather V1 is a separate container, so it is rejected by its own magic and the rejection
is pinned.
"""

import datetime
import decimal
import json
import re
import warnings
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet  # noqa: F401  # `pa.parquet` is not bound by importing `pyarrow`
import pytest
from dlt.extract.exceptions import ResourceExtractionError

from dlt_filesystem.source.format.readers import read_feather, read_parquet
from dlt_filesystem.source.fsspec.local import LocalFilesystemSource
from dlt_filesystem.target.local import LocalFilesystemDestination
from dlt_filesystem.target.registry import writer_for_format
from dlt_filesystem.target.writer import write_jsonl
from dlt_filesystem.testing.stub import FileItemStub
from dlt_filesystem.testing.writer import write_feather

ALIASES = ["feather", "arrow", "ipc"]


def _read_via_source(path, suffix=""):
    """Read a local Feather file end-to-end through the shared filesystem reader."""
    return list(LocalFilesystemSource().dlt_source(f"file://{path}{suffix}", ""))


def _read_table(path):
    """Read a written file back with PyArrow directly, independent of `src/`."""
    return pa.ipc.open_file(str(path)).read_all()


# --- end-to-end reader (fsspec, no Docker) ---


def test_read_single_row(tmp_path):
    path = write_feather(tmp_path / "one.feather", [{"id": 1, "name": "alice"}])
    assert _read_via_source(path) == [{"id": 1, "name": "alice"}]


def test_read_multiple_rows(tmp_path):
    docs = [{"id": i, "name": n} for i, n in enumerate(["a", "b", "c"], start=1)]
    path = write_feather(tmp_path / "arr.feather", docs)
    rows = _read_via_source(path)
    assert [r["id"] for r in rows] == [1, 2, 3]
    assert sorted(r["name"] for r in rows) == ["a", "b", "c"]


@pytest.mark.parametrize("alias", ALIASES)
def test_every_extension_resolves_to_the_reader(tmp_path, alias):
    """`.feather`, `.arrow` and `.ipc` all name the same container."""
    docs = [{"id": 1}, {"id": 2}, {"id": 3}]
    path = write_feather(tmp_path / f"by_ext.{alias}", docs)
    assert len(_read_via_source(path)) == 3


@pytest.mark.parametrize("alias", ALIASES)
def test_every_format_hint_resolves_to_the_reader(tmp_path, alias):
    """The same three spellings work as a `#hint` on a name that carries no format."""
    path = write_feather(tmp_path / f"feed-{alias}.dat", [{"id": 1}, {"id": 2}])
    assert len(_read_via_source(path, f"#{alias}")) == 2


def test_read_with_columns_single(tmp_path):
    data = [{"id": 1, "name": "alice", "age": 44}]
    path = write_feather(tmp_path / "one.feather", data)
    assert _read_via_source(path, "#columns=name") == [{"name": "alice"}]


def test_read_with_columns_json(tmp_path):
    data = [{"id": 1, "name": "alice", "age": 44}]
    path = write_feather(tmp_path / "one.feather", data)
    columns = json.dumps(["id", "name"])
    assert _read_via_source(path, f"#columns={columns}") == [{"id": 1, "name": "alice"}]


def test_read_with_columns_unknown(tmp_path):
    """The same message ORC gives, so a typo reads the same on either columnar format."""
    data = [{"id": 1, "name": "alice", "age": 44}]
    path = write_feather(tmp_path / "one.feather", data)
    with pytest.raises(ResourceExtractionError) as excinfo:
        _read_via_source(path, "#columns=unknown")
    assert excinfo.match(
        "Invalid column selected unknown. Valid names are age, id, name"
    )


def test_read_with_chunksize_success(tmp_path):
    data = [{"id": i} for i in range(5)]
    path = write_feather(tmp_path / "data.feather", data)
    chunks = list(
        read_feather(iter([FileItemStub(path)]), chunksize=2)  # ty: ignore[invalid-argument-type]
    )
    assert [len(chunk) for chunk in chunks] == [2, 2, 1]
    assert [row["id"] for chunk in chunks for row in chunk] == list(range(5))


def test_read_with_chunksize_invalid(tmp_path):
    path = write_feather(tmp_path / "one.feather", [{"id": 1}])
    with pytest.raises(ResourceExtractionError) as excinfo:
        _read_via_source(path, "#chunksize=foo")
    assert excinfo.match("chunksize must be an integer, not foo")


@pytest.mark.parametrize("chunksize", [0, -1])
def test_read_with_non_positive_chunksize(tmp_path, chunksize):
    path = write_feather(tmp_path / "one.feather", [{"id": 1}])
    with pytest.raises(ResourceExtractionError) as excinfo:
        _read_via_source(path, f"#chunksize={chunksize}")
    assert excinfo.match(f"chunksize must be greater than zero, not {chunksize}")


def test_read_with_invalid_option(tmp_path):
    path = write_feather(tmp_path / "one.feather", [{"id": 1}])
    with pytest.raises(TypeError) as excinfo:
        _read_via_source(path, "#invalid=true")
    assert excinfo.match(
        re.escape("read_feather(): got an unexpected keyword argument 'invalid'")
    )


@pytest.mark.parametrize("alias", ALIASES)
def test_a_gzipped_file_reads_under_every_extension(tmp_path, alias):
    """`.gz` stripping is format-agnostic, but the page claims it for these three names."""
    import gzip

    raw = write_feather(tmp_path / f"data.{alias}", [{"id": 1}, {"id": 2}])
    path = tmp_path / f"gz.{alias}.gz"
    path.write_bytes(gzip.compress(Path(raw).read_bytes()))

    assert _read_via_source(path) == [{"id": 1}, {"id": 2}]


def test_read_multiple_files_flushes_each_remainder(tmp_path):
    write_feather(tmp_path / "a.feather", [{"id": 1}, {"id": 2}, {"id": 3}])
    write_feather(tmp_path / "b.feather", [{"id": 4}, {"id": 5}])
    rows = list(LocalFilesystemSource().dlt_source(f"file://{tmp_path}/*.feather", ""))
    assert sorted(r["id"] for r in rows) == [1, 2, 3, 4, 5]


# --- batch boundaries ---


class _RecordingReader:
    """Wrap an `ipc.open_file` reader, recording which batches were asked for.

    `read_all` raises rather than returning: a reader that materialized the table and
    sliced it afterwards would satisfy every chunk-length assertion, so the point of
    this double is that such an implementation cannot pass.
    """

    def __init__(self, reader):
        self._reader = reader
        self.requested = []

    @property
    def num_record_batches(self):
        return self._reader.num_record_batches

    @property
    def schema(self):
        return self._reader.schema

    def get_batch(self, index):
        self.requested.append(index)
        return self._reader.get_batch(index)

    def read_all(self):
        raise AssertionError("read_feather must not materialize the whole file")


@pytest.fixture
def recording_open_file(monkeypatch):
    """Patch `pa.ipc.open_file` so the reader's batch requests are visible."""
    readers = []
    original = pa.ipc.open_file

    def spy(source, *args, **kwargs):
        reader = _RecordingReader(original(source, *args, **kwargs))
        readers.append(reader)
        return reader

    monkeypatch.setattr(pa.ipc, "open_file", spy)
    return readers


def test_a_batch_is_only_read_when_its_rows_are_wanted(tmp_path, recording_open_file):
    """Consuming one chunk must not pull the batches behind it.

    Written as three 1000-row batches; taking the first chunk may request batch 0 and
    nothing else. This is the assertion a slice-the-whole-table implementation fails.
    """
    path = write_feather(
        tmp_path / "batched.feather",
        [{"id": i} for i in range(3000)],
        batch_size=1000,
    )
    chunks = read_feather(iter([FileItemStub(path)]), chunksize=10)  # ty: ignore[invalid-argument-type]
    first = next(chunks)

    assert [row["id"] for row in first] == list(range(10))
    assert recording_open_file[0].requested == [0]
    # Closing the abandoned generator runs its `finally`, so the file handle is released
    # here rather than whenever the object is collected.
    chunks.close()  # ty: ignore[unresolved-attribute]


def test_physical_batches_and_chunksize_are_separate(tmp_path):
    """A chunk never spans two batches, so both sizes are visible in the output.

    2500 rows in batches of 1000, read at a chunksize of 400: each batch is chunked on
    its own, so the tail of every batch is a short chunk rather than being topped up
    from the next one.
    """
    path = write_feather(
        tmp_path / "batched.feather",
        [{"id": i} for i in range(2500)],
        batch_size=1000,
    )
    chunks = list(
        read_feather(iter([FileItemStub(path)]), chunksize=400)  # ty: ignore[invalid-argument-type]
    )

    assert [len(chunk) for chunk in chunks] == [400, 400, 200, 400, 400, 200, 400, 100]
    assert [row["id"] for chunk in chunks for row in chunk] == list(range(2500))


def test_a_batch_smaller_than_chunksize_yields_as_it_stands(tmp_path):
    """The reader never merges batches to fill a chunk."""
    path = write_feather(
        tmp_path / "small.feather", [{"id": i} for i in range(6)], batch_size=2
    )
    chunks = list(
        read_feather(iter([FileItemStub(path)]), chunksize=1000)  # ty: ignore[invalid-argument-type]
    )

    assert [len(chunk) for chunk in chunks] == [2, 2, 2]


# --- the dtype matrix ---


def test_the_container_carries_every_arrow_type(tmp_path):
    """The matrix behind the docs' type table.

    Two claims, deliberately separated. The `equals` assertion is about the *container*:
    PyArrow in, PyArrow out, no `src/` in between, because several of these types cannot
    be produced from a Python dict at all (a null column, a fixed-scale decimal). The
    row assertions after it are about the *reader*, which is what a load actually gets.
    `test_nanosecond_columns_narrow_on_the_row_path` holds the one place those two
    answers differ.
    """
    table = pa.table(
        {
            "i": pa.array([1, 2]),
            "s": pa.array(["a", "Zoë"]),
            "f": pa.array([1.5, 2.5]),
            "b": pa.array([True, False]),
            "date": pa.array([datetime.date(2020, 1, 1)] * 2),
            "naive": pa.array([datetime.datetime(2020, 1, 2, 3, 4, 5)] * 2),
            "aware": pa.array(
                [datetime.datetime(2020, 1, 2, 3, 4, 5, tzinfo=datetime.timezone.utc)]
                * 2
            ),
            "time": pa.array([datetime.time(9, 30)] * 2),
            "blob": pa.array([b"hi", b"yo"]),
            "dec": pa.array([decimal.Decimal("3.14")] * 2, type=pa.decimal128(38, 2)),
            "lst": pa.array([[1, 2], [3]]),
            "struct": pa.array([{"n": 1}, {"n": 2}]),
            "nul": pa.array([None, None], type=pa.null()),
        }
    )
    path = tmp_path / "types.feather"
    with pa.ipc.new_file(str(path), table.schema) as writer:
        writer.write_table(table)

    assert _read_table(path).equals(table)

    row = _read_via_source(path)[0]
    assert row["aware"].utcoffset() == datetime.timedelta(0)
    assert row["naive"].tzinfo is None
    assert row["time"] == datetime.time(9, 30)
    assert row["dec"] == decimal.Decimal("3.14")
    assert row["nul"] is None
    assert row["lst"] == [1, 2]
    assert row["struct"] == {"n": 1}


def test_nanosecond_columns_narrow_on_the_row_path(tmp_path):
    """A row carries Python values, so nanoseconds do not survive a full round trip.

    Measured rather than assumed, because the docs claim the type table above and this is
    its one exception. The reader keeps nanosecond timestamps and durations (pandas types
    carry them) and loses them on a `time64[ns]`, which becomes a `datetime.time`; writing
    any of the three back emits a microsecond column, because the writer infers its schema
    from those Python values.

    Parquet is asserted alongside so the claim that this belongs to the row pipeline rather
    than to Feather is mechanical rather than a comment: a reader that diverged would fail
    here.
    """
    import pyarrow.parquet as pq

    table = pa.table(
        {
            "t": pa.array([123456789], type=pa.time64("ns")),
            "ts": pa.array([1_600_000_000_123_456_789], type=pa.timestamp("ns")),
            "dur": pa.array([1234567890], type=pa.duration("ns")),
        }
    )
    path = tmp_path / "ns.feather"
    with pa.ipc.new_file(str(path), table.schema) as writer:
        writer.write_table(table)

    # The container kept every nanosecond; the schema on disk still says so.
    assert _read_table(path).equals(table)

    row = next(iter(read_feather(iter([FileItemStub(path)]))))[0]  # ty: ignore[invalid-argument-type]
    assert row["t"] == datetime.time(0, 0, 0, 123456), (
        "datetime.time has no nanoseconds"
    )
    assert row["ts"].nanosecond == 789, "a pandas Timestamp carries them"
    assert row["dur"].nanoseconds == 890, "and so does a pandas Timedelta"

    parquet_path = tmp_path / "ns.parquet"
    pq.write_table(table, str(parquet_path))
    assert next(iter(read_parquet(iter([FileItemStub(parquet_path)]))))[0] == row  # ty: ignore[invalid-argument-type]

    # Writing those rows back is where the surviving nanoseconds go: the writer infers
    # its schema from Python values, so every one of the three lands at microseconds.
    out = tmp_path / "out.feather"
    writer_for_format("feather")(str(out), [row])
    assert [field.type for field in _read_table(out).schema] == [
        pa.time64("us"),
        pa.timestamp("us"),
        pa.duration("us"),
    ]


def test_read_adversarial_values_are_normalized(tmp_path):
    """A timezone-aware datetime and a Decimal pass through the source unchanged."""
    doc = {
        "when": datetime.datetime(2020, 1, 2, 3, 4, 5, tzinfo=datetime.timezone.utc),
        "amt": decimal.Decimal("3.14"),
    }
    path = write_feather(tmp_path / "adv.feather", [doc])
    row = _read_via_source(path)[0]
    assert isinstance(row["when"], datetime.datetime)
    assert row["when"].utcoffset() == datetime.timedelta(0)
    assert row["amt"] == decimal.Decimal("3.14")


# --- rejected containers and damaged files ---


def test_feather_v1_is_rejected_by_name(tmp_path):
    """V1 is a different container. `pa.ipc.open_file` calls it "Not an Arrow file",
    which reads as damage rather than as a version this reader does not open."""
    from pyarrow import feather

    path = tmp_path / "v1.feather"
    with warnings.catch_warnings():
        # `pyarrow.feather` and V1 itself are both deprecated; it is the only way to
        # produce a V1 file, and that deprecation is what makes V1 a dead end.
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", FutureWarning)
        feather.write_feather(pa.table({"id": [1, 2]}), str(path), version=1)

    with pytest.raises(ResourceExtractionError) as excinfo:
        _read_via_source(path)
    assert excinfo.match("Feather V1 files are not supported")
    assert excinfo.match("Feather V2")


def test_read_empty_feather_file_raises_resource_extraction_error(tmp_path):
    path = tmp_path / "empty.feather"
    path.write_bytes(b"")
    with pytest.raises(ResourceExtractionError) as excinfo:
        _read_via_source(path)
    assert excinfo.match("File is too small")


def test_read_garbage_raises_rather_than_loading_partial_data(tmp_path):
    path = tmp_path / "garbage.feather"
    path.write_bytes(b"not an arrow file at all")
    with pytest.raises(ResourceExtractionError):
        _read_via_source(path)


# --- the writer ---


def test_write_preserves_sparse_rows_and_column_order(tmp_path):
    rows = [
        {"id": 1, "name": "Zoë"},
        {"id": 2, "name": "Ōtautahi", "note": "late column"},
    ]
    path = tmp_path / "out.feather"
    writer_for_format("feather")(str(path), rows)

    table = _read_table(path)
    assert table.column_names == ["id", "name", "note"]
    assert table.to_pylist() == [
        {"id": 1, "name": "Zoë", "note": None},
        {"id": 2, "name": "Ōtautahi", "note": "late column"},
    ]


@pytest.mark.parametrize("alias", ALIASES)
def test_every_alias_routes_to_the_same_writer(alias):
    assert writer_for_format(alias) is writer_for_format("feather")


def test_write_of_no_rows_is_valid(tmp_path):
    path = tmp_path / "empty.feather"
    writer_for_format("feather")(str(path), [])

    assert _read_table(path).num_rows == 0


def test_write_rejects_nonempty_fieldless_rows(tmp_path):
    """Nonempty fieldless rows would write a file with rows nothing can address."""
    output = tmp_path / "fieldless.feather"

    with pytest.raises(ValueError, match="requires at least one column"):
        writer_for_format("feather")(str(output), [{}, {}])

    assert not output.exists()


def test_written_file_is_v2_not_v1(tmp_path):
    """Pinned on the bytes, because a V1 file would still satisfy the row assertions
    above while being unreadable by this package's own reader."""
    path = tmp_path / "out.feather"
    writer_for_format("feather")(str(path), [{"id": 1}])

    assert path.read_bytes()[:6] == b"ARROW1"


def test_write_round_trips_the_type_matrix_from_python_rows(tmp_path):
    """The writer's own half of the type table, from dict rows rather than a `pa.table`.

    `test_the_container_carries_every_arrow_type` builds its columns with `pa.array`, so
    it pins the container and the reader and never exercises the dict-to-`pa.table`
    inference the writer actually relies on. This is that path: values a load could
    plausibly hand a writer, out through `write_feather` and back through the source.
    """
    row = {
        "i": 1,
        "s": "Zoë",
        "f": 1.5,
        "b": True,
        "date": datetime.date(2020, 1, 1),
        "naive": datetime.datetime(2020, 1, 2, 3, 4, 5),
        "aware": datetime.datetime(2020, 1, 2, 3, 4, 5, tzinfo=datetime.timezone.utc),
        "time": datetime.time(9, 30),
        "blob": b"hi",
        "dec": decimal.Decimal("3.14"),
        "lst": [1, 2],
        "st": {"n": 1},
        "nul": None,
    }
    path = tmp_path / "matrix.feather"
    writer_for_format("feather")(str(path), [row])

    assert _read_via_source(path) == [row]

    # The values alone are not the claim. Python calls `True == 1` and `1 == 1.0`, so a
    # writer that inferred a boolean as an integer, or an integer as a double, satisfies
    # the row comparison above; only the schema separates them.
    assert {field.name: str(field.type) for field in _read_table(path).schema} == {
        "i": "int64",
        "s": "string",
        "f": "double",
        "b": "bool",
        "date": "date32[day]",
        "naive": "timestamp[us]",
        "aware": "timestamp[us, tz=UTC]",
        "time": "time64[us]",
        "blob": "binary",
        "dec": "decimal128(3, 2)",
        "lst": "list<item: int64>",
        "st": "struct<n: int64>",
        "nul": "null",
    }


def test_write_refuses_an_unsigned_integer_wider_than_a_signed_64(tmp_path):
    """A limit of the writer rather than of the format, shared with `write_orc`.

    Both build their table from Python values, and PyArrow's inference from a Python int
    tops out at a signed 64-bit. The value reads back fine, so this is reachable by
    loading such a column and exporting it, and it raises rather than writing a wrong
    number.
    """
    source = tmp_path / "u64.feather"
    table = pa.table({"u": pa.array([2**64 - 1], type=pa.uint64())})
    with pa.ipc.new_file(str(source), table.schema) as writer:
        writer.write_table(table)

    row = _read_via_source(source)[0]
    assert row["u"] == 2**64 - 1, "the read side carries it"

    out = tmp_path / "out.feather"
    with pytest.raises(OverflowError):
        writer_for_format("feather")(str(out), [row])

    # The alternatives the page names, asserted rather than assumed: all four keep the
    # value digit for digit. Parquet is named as a non-alternative for the same reason,
    # and that is the case worth pinning: it writes without complaint and its own reader
    # refuses the result, which is a worse failure than this one and easy to walk into.
    import json

    for file_format, decode in (
        ("json", lambda p: json.loads(p.read_text())[0]["u"]),
        ("jsonl", lambda p: json.loads(p.read_text().splitlines()[0])["u"]),
        ("csv", lambda p: p.read_text().splitlines()[1]),
        ("yaml", lambda p: __import__("yaml").safe_load(p.read_text())[0]["u"]),
    ):
        alt = tmp_path / f"alt.{file_format}"
        writer_for_format(file_format)(str(alt), [row])
        assert str(decode(alt)) == str(2**64 - 1), file_format

    parquet_out = tmp_path / "alt.parquet"
    writer_for_format("parquet")(str(parquet_out), [row])
    with pytest.raises(pa.lib.ArrowNotImplementedError, match="more than 64 bits"):
        pa.parquet.read_table(str(parquet_out))


def test_written_columns_are_not_arrow_view_types(tmp_path):
    """The reason this writer stayed on PyArrow when csv and parquet moved to Polars.

    For Feather the Arrow schema is the file, so `DataFrame.write_ipc` at its default
    compat level puts `string_view` and `binary_view` into it, and those are an Arrow 15
    feature rather than something every reader opens. Parquet has its own type system and
    was insulated from that. Pinned here so a later move to Polars has to answer it.
    """
    path = tmp_path / "out.feather"
    writer_for_format("feather")(str(path), [{"s": "a", "b": b"hi"}])

    types = {field.name: field.type for field in _read_table(path).schema}
    assert types == {"s": pa.string(), "b": pa.binary()}


def test_written_file_is_readable_by_polars(tmp_path):
    """Interop in the other direction: what we write is Arrow IPC, not our own dialect."""
    import polars as pl

    path = tmp_path / "out.feather"
    writer_for_format("feather")(str(path), [{"id": 1}, {"id": 2}])

    assert pl.read_ipc(str(path))["id"].to_list() == [1, 2]


def test_write_destination_round_trips_without_dlt_columns(tmp_path):
    destination = LocalFilesystemDestination()
    output_path = tmp_path / "out.feather"
    destination.dlt_dest(f"file://{output_path}")
    destination.dataset_name, destination.table_name = "public", "rows"
    table_dir = Path(destination.temp_path) / "public" / "rows"
    table_dir.mkdir(parents=True)
    rows = [
        {"id": 1, "name": "alice", "_dlt_id": "internal"},
        {"id": 2, "name": "bob", "note": "later", "_dlt_load_id": "internal"},
    ]
    write_jsonl(str(table_dir / "load.jsonl"), rows)
    destination.post_load()

    assert _read_via_source(output_path) == [
        {"id": 1, "name": "alice", "note": None},
        {"id": 2, "name": "bob", "note": "later"},
    ]
