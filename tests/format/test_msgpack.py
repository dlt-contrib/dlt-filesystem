"""Test the MessagePack filesystem reader: the generic iterabledata harness + msgpack.

Mock-only unit lane (no Docker, no credentials): msgpack files are written to ``tmp_path``
and read back through ``LocalFilesystemSource``, which exercises the same ``read_msgpack``
transformer the S3/GCS/SFTP sources use. The suite covers: the stream bridge (incl. the
non-seekable spool), extended-type normalization, chunk-boundary flushing, the pinned import
path, the missing-decoder message, and that the decoder stays out of ``sys.modules`` on the
import path.

Skipped entirely when the optional ``iterable`` extra isn't installed.
"""

import base64
import copy
import dataclasses
import datetime
import decimal
import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import pytest
from dlt.common.pendulum import pendulum

from dlt_filesystem.source.error import MissingDecoderError
from dlt_filesystem.source.format.iterable_codec import (
    FORMAT_TO_ITERABLE,
    _load_iterable_class,
    read_via_iterable,
)
from dlt_filesystem.source.format.readers import read_msgpack
from dlt_filesystem.source.fsspec.local import LocalFilesystemSource
from dlt_filesystem.target.local import LocalFilesystemDestination
from dlt_filesystem.target.registry import writer_for_format
from dlt_filesystem.target.writer import write_jsonl
from dlt_filesystem.testing.stub import (
    FileItemStub,
    NonSeekableItem,
)
from dlt_filesystem.testing.writer import write_msgpack

pytest.importorskip("iterable.datatypes.msgpack")
msgpack = pytest.importorskip("msgpack")


def _read_via_source(path):
    """Read a local msgpack file end-to-end through the shared filesystem reader."""
    return list(LocalFilesystemSource().dlt_source(f"file://{path}", ""))


# --- end-to-end reader (fsspec, no Docker) ---


def test_reads_single_msgpack_record(tmp_path):
    """A single-record msgpack file loads that one record end-to-end."""
    path = write_msgpack(tmp_path / "one.msgpack", [{"id": 1, "name": "alice"}])
    rows = _read_via_source(path)
    assert rows == [{"id": 1, "name": "alice"}]


def test_reads_multiple_msgpack_records(tmp_path):
    """A multi-record msgpack file loads every record end-to-end."""
    docs = [{"id": i, "name": n} for i, n in enumerate(["a", "b", "c"], start=1)]
    path = write_msgpack(tmp_path / "many.msgpack", docs)
    rows = _read_via_source(path)
    assert [r["id"] for r in rows] == [1, 2, 3]
    assert sorted(r["name"] for r in rows) == ["a", "b", "c"]


def test_extension_and_format_hint_both_resolve(tmp_path):
    """A `.msgpack` extension and an explicit `#msgpack` hint both resolve to the reader."""
    docs = [{"id": 1}, {"id": 2}]
    # extension detection
    ext_path = write_msgpack(tmp_path / "by_ext.msgpack", docs)
    assert len(_read_via_source(ext_path)) == 2
    # explicit #msgpack hint on a non-standard extension
    hint_path = write_msgpack(tmp_path / "feed.dat", docs)
    rows = list(LocalFilesystemSource().dlt_source(f"file://{hint_path}#msgpack", ""))
    assert len(rows) == 2


def test_adversarial_values_are_normalized(tmp_path):
    """Adversarial record: raw bytes, a Timestamp extension, and a nested map with a nested
    bytes value. All become dlt-safe, the same normalization the BSON reader applies."""
    doc = {
        "blob": b"\x00\x01\x02",
        "when": datetime.datetime(2020, 1, 2, 3, 4, 5, tzinfo=datetime.timezone.utc),
        "nested": {"inner": b"AB", "tags": ["x", "y"]},
    }
    path = write_msgpack(tmp_path / "adv.msgpack", [doc], datetime=True)
    row = _read_via_source(path)[0]
    assert base64.b64decode(row["blob"]) == b"\x00\x01\x02"
    assert isinstance(row["when"], pendulum.DateTime)
    assert row["when"].utcoffset() == datetime.timedelta(0)
    assert (row["when"].year, row["when"].month, row["when"].day) == (2020, 1, 2)
    assert base64.b64decode(row["nested"]["inner"]) == b"AB"
    assert row["nested"]["tags"] == ["x", "y"]


def test_custom_ext_type_is_dlt_safe(tmp_path):
    """A custom msgpack extension (non-Timestamp) decodes to ``msgpack.ExtType``, which is a
    namedtuple, so ``map_nested_values_in_place`` turns it into ``{"code", "data"}`` and the
    normalizer base64s its ``data``. Nothing non-serializable reaches dlt."""
    path = write_msgpack(tmp_path / "ext.msgpack", [{"x": msgpack.ExtType(7, b"raw")}])
    row = _read_via_source(path)[0]
    assert row["x"] == {"code": 7, "data": base64.b64encode(b"raw").decode("ascii")}


def test_reads_multiple_files_flushes_each_remainder(tmp_path):
    """A multi-file glob must flush each file's partial final chunk before the next file,
    so no records are dropped at a file boundary."""
    write_msgpack(tmp_path / "a.msgpack", [{"id": 1}, {"id": 2}, {"id": 3}])
    write_msgpack(tmp_path / "b.msgpack", [{"id": 4}, {"id": 5}])
    rows = list(LocalFilesystemSource().dlt_source(f"file://{tmp_path}/*.msgpack", ""))
    assert sorted(r["id"] for r in rows) == [1, 2, 3, 4, 5]


# --- chunking contract (drives read_msgpack directly with a small chunksize) ---


def test_chunks_at_boundary_and_flushes_partial(tmp_path):
    """Records are yielded in chunksize-sized chunks, with the partial final chunk flushed."""
    path = write_msgpack(tmp_path / "five.msgpack", [{"id": i} for i in range(5)])
    chunks = list(read_msgpack(iter([FileItemStub(path)]), chunksize=2))  # ty: ignore[invalid-argument-type]
    assert [len(chunk) for chunk in chunks] == [2, 2, 1]
    assert [doc["id"] for chunk in chunks for doc in chunk] == [0, 1, 2, 3, 4]


def test_exact_chunksize_multiple_yields_no_empty_trailing_chunk(tmp_path):
    """A file whose record count is an exact multiple of chunksize must not emit a trailing
    empty chunk (the read_bulk()==[] EOF sentinel)."""
    path = write_msgpack(tmp_path / "four.msgpack", [{"id": i} for i in range(4)])
    chunks = list(read_msgpack(iter([FileItemStub(path)]), chunksize=2))  # ty: ignore[invalid-argument-type]
    assert [len(chunk) for chunk in chunks] == [2, 2]
    assert [doc["id"] for chunk in chunks for doc in chunk] == [0, 1, 2, 3]


def test_non_seekable_stream_is_spooled(tmp_path):
    """A non-seekable handle (remote pipe / some compressed streams) is spooled into a
    BytesIO so iterabledata's seek(0) reset works. The central stream-bridge claim."""
    data = b"".join(msgpack.packb({"id": i}, use_bin_type=True) for i in range(3))
    chunks = list(
        read_via_iterable(
            iter([NonSeekableItem(data)]),  # ty: ignore[invalid-argument-type]
            file_format="msgpack",
            chunksize=1000,
        )
    )
    assert [doc["id"] for chunk in chunks for doc in chunk] == [0, 1, 2]


# --- registry / import-path / error UX ---


def test_msgpack_import_path_resolves_to_the_class():
    """The `iterable.datatypes.*` module naming is non-obvious (msgpack -> msgpack,
    bson -> bsonf) and could shift; pin it with a direct import-path check."""
    spec = FORMAT_TO_ITERABLE["msgpack"]
    assert spec.class_path == "iterable.datatypes.msgpack.MessagePackIterable"
    klass = _load_iterable_class(spec)
    assert klass.__name__ == "MessagePackIterable"


def test_missing_decoder_raises_typed_install_hint(tmp_path, monkeypatch):
    """With the msgpack decoder unavailable, the reader raises a typed error naming the
    exact `pip install`, not a bare ImportError."""
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        """Report the msgpack decoder as absent, delegating other lookups to the real finder."""
        if name == "msgpack":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    path = write_msgpack(tmp_path / "x.msgpack", [{"id": 1}])
    with pytest.raises(MissingDecoderError) as exc:
        list(read_msgpack(iter([FileItemStub(path)]), chunksize=10))  # ty: ignore[invalid-argument-type]
    message = str(exc.value)
    assert "msgpack" in message
    assert "dlt-filesystem[iterable]" in message


def test_msgpack_advertised_only_when_decoder_installed(monkeypatch):
    """A base install must not advertise msgpack as supported when its decoder is absent,
    even though the format stays routable. With the decoder present it is listed."""
    from dlt_filesystem.source.format.registry import advertised_file_formats

    assert "msgpack" in advertised_file_formats()

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        """Report the msgpack decoder as absent, delegating other lookups to the real finder."""
        if name == "msgpack":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    assert "msgpack" not in advertised_file_formats()


def test_import_path_keeps_decoder_out_of_sys_modules():
    """Importing the filesystem reader path must not pull `iterable` or `msgpack` (they are
    lazy-imported only when a msgpack file is actually read). Runs in a subprocess so an
    earlier test in this module that imported them can't mask the result."""
    code = (
        "import sys\n"
        "import dlt_filesystem.source.adapter\n"
        "decoders = ('iterable', 'msgpack', 'cbor2')\n"
        "leaked = sorted(m for m in sys.modules if m.split('.')[0] in decoders)\n"
        "assert not leaked, leaked\n"
    )
    result = subprocess.run(  # noqa: S603  # trusted: sys.executable + a fixed code string
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("hours", [5, -7])
def test_msgpack_timestamp_preserves_instant_in_utc(tmp_path, hours):
    value = datetime.datetime(
        2020,
        1,
        2,
        3,
        4,
        5,
        123456,
        tzinfo=datetime.timezone(datetime.timedelta(hours=hours)),
    )
    path = write_msgpack(
        tmp_path / "timestamp.msgpack", [{"when": value}], datetime=True
    )
    result = _read_via_source(path)[0]["when"]
    assert isinstance(result, pendulum.DateTime)
    assert result == value.astimezone(datetime.timezone.utc)
    assert result.utcoffset() == datetime.timedelta(0)
    assert result.microsecond == 123456


# --- the writer: what `file://out.msgpack` writes reads back through the reader above ---

NAIVE = datetime.datetime(2020, 1, 2, 3, 4, 5)
AWARE = datetime.datetime(
    2020, 1, 2, 3, 4, 5, 123456, tzinfo=datetime.timezone(datetime.timedelta(hours=5))
)

# What each value a load can hand a writer reads back as. MessagePack has no type for a
# decimal, a date, a time or a naive timestamp, so those come back as the strings the
# JSON writers write for them; an aware timestamp comes back as the same instant in UTC,
# and `bytes` as the base64 string the reader makes of them.
WRITTEN_AND_READ_BACK = {
    "a string": ("Zoë", "Zoë"),
    "the top of 64 bits": (2**64 - 1, 2**64 - 1),
    "the bottom of 64 bits": (-(2**63), -(2**63)),
    "a float": (1.5, 1.5),
    "infinity": (math.inf, math.inf),
    "negative infinity": (-math.inf, -math.inf),
    "a boolean": (True, True),
    "null": (None, None),
    "a decimal": (decimal.Decimal("1.50"), "1.50"),
    "a date": (datetime.date(2020, 1, 2), "2020-01-02"),
    "a time": (datetime.time(3, 4, 5, 6), "03:04:05.000006"),
    "an aware datetime": (AWARE, AWARE.astimezone(datetime.timezone.utc)),
    "a pendulum datetime": (
        pendulum.datetime(2020, 1, 2, 3, 4, 5, tz="Pacific/Auckland"),
        datetime.datetime(2020, 1, 1, 14, 4, 5, tzinfo=datetime.timezone.utc),
    ),
    # The timestamp extension's edges: seconds before the epoch with a positive
    # fraction, and the last microsecond a datetime holds.
    "a pendulum datetime before 1970": (
        pendulum.datetime(1900, 1, 1, 0, 0, 0, 1, tz="UTC"),
        datetime.datetime(1900, 1, 1, 0, 0, 0, 1, tzinfo=datetime.timezone.utc),
    ),
    "half a second before 1970": (
        datetime.datetime(
            1969, 12, 31, 23, 59, 59, 500000, tzinfo=datetime.timezone.utc
        ),
        datetime.datetime(
            1969, 12, 31, 23, 59, 59, 500000, tzinfo=datetime.timezone.utc
        ),
    ),
    "a pendulum datetime at the end of 9999": (
        pendulum.datetime(9999, 12, 31, 23, 59, 59, 999999, tz="UTC"),
        datetime.datetime(
            9999, 12, 31, 23, 59, 59, 999999, tzinfo=datetime.timezone.utc
        ),
    ),
    "a naive datetime": (NAIVE, "2020-01-02T03:04:05"),
    "a naive datetime in a list": ([NAIVE], ["2020-01-02T03:04:05"]),
    "a naive datetime in a tuple": ((NAIVE,), ["2020-01-02T03:04:05"]),
    # The writer turns an aware datetime into a timestamp itself, so a tuple has to be
    # walked for it as a list is.
    "an aware datetime in a tuple": (
        (AWARE,),
        [AWARE.astimezone(datetime.timezone.utc)],
    ),
    "a naive datetime in a dict": ({"at": NAIVE}, {"at": "2020-01-02T03:04:05"}),
    "bytes": (b"hi", "aGk="),
    "a nested document": ({"a": [1, {"b": None}]}, {"a": [1, {"b": None}]}),
}


def _write_and_read(path, rows):
    writer_for_format("msgpack")(str(path), rows)
    return _read_via_source(path)


@pytest.mark.parametrize(
    ("value", "expected"), WRITTEN_AND_READ_BACK.values(), ids=WRITTEN_AND_READ_BACK
)
def test_write_round_trips_through_the_reader(tmp_path, value, expected):
    [row] = _write_and_read(tmp_path / "out.msgpack", [{"v": value}])

    assert row == {"v": expected}
    # `True == 1` and a UTC datetime equals the same instant in any zone, so the type
    # and the offset are asserted as well as the value. `isinstance`, because the
    # reader returns pendulum's datetime.
    assert isinstance(row["v"], type(expected))
    if isinstance(expected, datetime.datetime):
        assert row["v"].utcoffset() == datetime.timedelta(0)


def test_write_keeps_an_ext_type_an_extension(tmp_path):
    """An `ExtType` is a namedtuple, and a tuple walked into a new one would be written
    as an array. Read back the way the reader reads any custom extension."""
    [row] = _write_and_read(
        tmp_path / "out.msgpack", [{"x": msgpack.ExtType(7, b"raw")}]
    )

    assert row == {"x": {"code": 7, "data": base64.b64encode(b"raw").decode("ascii")}}


def test_write_keeps_nan_a_float(tmp_path):
    """NaN compares unequal to itself, so it is asserted apart from the table above."""
    [row] = _write_and_read(tmp_path / "out.msgpack", [{"v": math.nan}])

    assert math.isnan(row["v"])


EMPTY_SHAPES = {
    "no rows": [],
    "one empty row": [{}],
    "an empty row before a full one": [{}, {"a": 1}],
}


@pytest.mark.parametrize("rows", EMPTY_SHAPES.values(), ids=EMPTY_SHAPES)
def test_write_round_trips_an_empty_shape(tmp_path, rows):
    # Against a copy, so a writer that changed its input could not pass by matching it.
    expected = copy.deepcopy(rows)

    assert _write_and_read(tmp_path / "out.msgpack", rows) == expected


def test_write_of_no_rows_is_an_empty_file(tmp_path):
    """A stream of maps with none in it, which the reader reads as no rows."""
    path = tmp_path / "out.msgpack"
    writer_for_format("msgpack")(str(path), [])

    assert path.read_bytes() == b""


# The reader decodes with `strict_map_key=True`, so it refuses an integer, boolean or
# null key, and it hands a `bytes` key back undecoded. A file carrying one would not read
# back as the load that wrote it.
REFUSED_KEYS = {
    "an integer column name": [{1: "x"}],
    "a boolean column name": [{True: "x"}],
    "a null column name": [{None: "x"}],
    "a bytes column name": [{b"k": "x"}],
    "an integer key in a map": [{"v": {1: "x"}}],
    "a bytes key in a map in a list": [{"v": [{b"k": "x"}]}],
}


@pytest.mark.parametrize("rows", REFUSED_KEYS.values(), ids=REFUSED_KEYS)
def test_write_refuses_a_key_that_is_not_a_string(tmp_path, rows):
    path = tmp_path / "out.msgpack"

    with pytest.raises(ValueError, match="only string keys"):
        writer_for_format("msgpack")(str(path), rows)

    assert not path.exists()


@dataclasses.dataclass
class _Document:
    ratio: float
    counts: dict


@pytest.mark.parametrize(
    "document",
    [_Document(math.nan, {"a": 1}), _Document(0.5, {1: "x"})],
    ids=["a NaN inside", "an integer key inside"],
)
def test_write_refuses_an_object_dlt_spells_as_a_document(tmp_path, document):
    """Spelled whole through JSON, the NaN would be written as null and the integer key
    as a string, both past the checks the writer runs on a dict. Which layer refuses
    depends on dlt's JSON backend (simplejson raises on the NaN itself), so this pins
    the refusal and the untouched path rather than the message."""
    path = tmp_path / "out.msgpack"

    with pytest.raises((TypeError, ValueError)):
        writer_for_format("msgpack")(str(path), [{"v": document}])

    assert not path.exists()


@pytest.mark.parametrize(
    "value",
    [
        datetime.datetime(
            1, 1, 1, tzinfo=datetime.timezone(datetime.timedelta(hours=1))
        ),
        datetime.datetime(
            9999, 12, 31, 23, tzinfo=datetime.timezone(datetime.timedelta(hours=-1))
        ),
    ],
    ids=["before year 1 in UTC", "after year 9999 in UTC"],
)
def test_write_refuses_a_datetime_outside_utc(tmp_path, value):
    """Written as UTC, it would encode and then fail to read back."""
    path = tmp_path / "out.msgpack"

    with pytest.raises(ValueError, match="Column 'v'"):
        writer_for_format("msgpack")(str(path), [{"v": [value]}])

    assert not path.exists()


REFUSED_NUMBERS = {
    "one past the top": [{"v": 2**64}],
    "one past the bottom": [{"v": -(2**63) - 1}],
    "in a list": [{"v": [1, 2**64]}],
    "in a tuple": [{"v": (1, 2**64)}],
    "in a map": [{"v": {"n": 2**70}}],
}


@pytest.mark.parametrize("rows", REFUSED_NUMBERS.values(), ids=REFUSED_NUMBERS)
def test_write_refuses_an_integer_outside_64_bits(tmp_path, rows):
    """MessagePack raises mid-encode on these, so they are found before it starts."""
    path = tmp_path / "out.msgpack"

    with pytest.raises(ValueError, match="Column 'v'") as excinfo:
        writer_for_format("msgpack")(str(path), rows)

    assert "64 bits" in str(excinfo.value)
    assert "YAML" in str(excinfo.value)
    assert not path.exists()


REFUSED_LOADS = {
    "a key": [{"id": 1}, {1: "x"}],
    "an integer": [{"id": 1}, {"id": 2**64}],
    # Not refused by name: no JSON backend dlt offers spells a bare object, so the
    # encoder raises. The file is still left alone, because nothing is written until
    # the whole load is encoded.
    "an encoder error": [{"id": 1}, {"v": object()}],
}


@pytest.mark.parametrize("rows", REFUSED_LOADS.values(), ids=REFUSED_LOADS)
def test_a_refused_write_keeps_the_file_already_there(tmp_path, rows):
    path = tmp_path / "out.msgpack"
    writer_for_format("msgpack")(str(path), [{"id": 0}])
    before = path.read_bytes()

    with pytest.raises((ValueError, TypeError)):
        writer_for_format("msgpack")(str(path), rows)

    assert path.read_bytes() == before


def test_write_without_msgpack_names_the_install(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "msgpack", None)

    with pytest.raises(MissingDecoderError, match=r"dlt-filesystem\[iterable\]"):
        writer_for_format("msgpack")(str(tmp_path / "out.msgpack"), [{"id": 1}])


def test_write_destination_round_trips_without_dlt_columns(tmp_path):
    """Through `post_load`, from a staged JSONL file, the way a pipeline reaches it."""
    destination = LocalFilesystemDestination()
    output_path = tmp_path / "out.msgpack"
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
        {"id": 1, "name": "alice"},
        {"id": 2, "name": "bob", "note": "later"},
    ]
