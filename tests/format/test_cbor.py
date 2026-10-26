"""Test the CBOR filesystem reader: the second format on the generic iterabledata harness.

Mock-only unit lane (no Docker, no credentials): CBOR files are written to ``tmp_path`` and
read back through ``LocalFilesystemSource``. Covers: CBOR resolves by extension + ``#cbor``
hint, the adversarial fixture (bytes, datetime, Decimal, a custom tag, nested) loads dlt-safe,
and the documented single-top-level-value constraint holds (concatenated CBOR objects read
only the first, a cbor2 limitation that can't be detected).

Skipped when ``cbor2`` isn't installed (it ships in this package's ``iterable`` extra). CBOR
decodes with ``cbor2`` directly, so it does not need the ``iterable`` package itself.
"""

import base64
import collections
import copy
import dataclasses
import datetime
import decimal
import importlib.util
import math
import sys
import zoneinfo
from pathlib import Path

import pytest
from dlt.common.pendulum import pendulum

from dlt_filesystem.source.error import MissingDecoderError
from dlt_filesystem.source.format.iterable_codec import (
    FORMAT_TO_ITERABLE,
    read_via_iterable,
)
from dlt_filesystem.source.format.readers import read_cbor
from dlt_filesystem.source.fsspec.local import LocalFilesystemSource
from dlt_filesystem.target.local import LocalFilesystemDestination
from dlt_filesystem.target.registry import writer_for_format
from dlt_filesystem.target.writer import write_jsonl
from dlt_filesystem.testing.stub import (
    FileItemStub,
    NonSeekableItem,
)
from dlt_filesystem.testing.writer import write_cbor

cbor2 = pytest.importorskip("cbor2")


def _read_via_source(path):
    """Read a local cbor file end-to-end through the shared filesystem reader."""
    return list(LocalFilesystemSource().dlt_source(f"file://{path}", ""))


# --- end-to-end reader (fsspec, no Docker) ---


def test_reads_single_top_level_object(tmp_path):
    """A single top-level CBOR map loads as one record."""
    path = write_cbor(tmp_path / "one.cbor", {"id": 1, "name": "alice"})
    assert _read_via_source(path) == [{"id": 1, "name": "alice"}]


def test_reads_top_level_array_all_rows(tmp_path):
    """The supported shape: a single top-level array yields one row per element."""
    docs = [{"id": i, "name": n} for i, n in enumerate(["a", "b", "c"], start=1)]
    path = write_cbor(tmp_path / "arr.cbor", docs)
    rows = _read_via_source(path)
    assert [r["id"] for r in rows] == [1, 2, 3]
    assert sorted(r["name"] for r in rows) == ["a", "b", "c"]


def test_shared_container_references_are_not_cycles(tmp_path):
    """Finite shared CBOR values load normally, including a shared tag payload."""
    shared = {"blob": b"AB"}
    doc = {"a": shared, "b": shared, "tagged": cbor2.CBORTag(1234, shared)}
    path = tmp_path / "shared.cbor"
    path.write_bytes(cbor2.dumps(doc, value_sharing=True))
    assert _read_via_source(path) == [
        {
            "a": {"blob": "QUI="},
            "b": {"blob": "QUI="},
            "tagged": {"tag": 1234, "value": {"blob": "QUI="}},
        }
    ]


def test_concatenated_objects_read_only_the_first(tmp_path):
    """Documented limitation: a file that concatenates several top-level CBOR objects is read
    only up to the first; cbor2 cannot stream them and it can't be detected. The fix is to
    write a top-level array (see test above)."""
    path = tmp_path / "concat.cbor"
    with open(path, "wb") as f:
        for doc in [{"id": 1}, {"id": 2}, {"id": 3}]:
            f.write(cbor2.dumps(doc))
    rows = _read_via_source(path)
    assert rows == [{"id": 1}]


def test_extension_and_format_hint_both_resolve(tmp_path):
    """A `.cbor` extension and an explicit `#cbor` hint both resolve to the reader."""
    docs = [{"id": 1}, {"id": 2}]
    ext_path = write_cbor(tmp_path / "by_ext.cbor", docs)
    assert len(_read_via_source(ext_path)) == 2
    hint_path = write_cbor(tmp_path / "feed.dat", docs)
    rows = list(LocalFilesystemSource().dlt_source(f"file://{hint_path}#cbor", ""))
    assert len(rows) == 2


def test_adversarial_values_are_normalized(tmp_path):
    """Adversarial record: raw bytes, a tz-aware datetime, a Decimal, an unknown CBOR tag, and
    a nested map with a nested bytes value. bytes -> base64, an unknown tag -> {"tag","value"};
    datetime and Decimal are already dlt-safe and pass through."""
    doc = {
        "blob": b"\x00\x01\x02",
        "when": datetime.datetime(2020, 1, 2, 3, 4, 5, tzinfo=datetime.timezone.utc),
        "amt": decimal.Decimal("3.14"),
        "tagged": cbor2.CBORTag(1234, "custom"),
        "nested": {"inner": b"AB", "tags": ["x", "y"]},
    }
    path = write_cbor(tmp_path / "adv.cbor", [doc])
    row = _read_via_source(path)[0]
    assert base64.b64decode(row["blob"]) == b"\x00\x01\x02"
    assert isinstance(row["when"], datetime.datetime)
    assert row["when"].utcoffset() == datetime.timedelta(0)
    assert row["amt"] == decimal.Decimal("3.14")
    assert row["tagged"] == {"tag": 1234, "value": "custom"}
    assert base64.b64decode(row["nested"]["inner"]) == b"AB"
    assert row["nested"]["tags"] == ["x", "y"]


def test_reads_multiple_files_flushes_each_remainder(tmp_path):
    """Multi-file glob: each file is a top-level array; all records across files load."""
    write_cbor(tmp_path / "a.cbor", [{"id": 1}, {"id": 2}, {"id": 3}])
    write_cbor(tmp_path / "b.cbor", [{"id": 4}, {"id": 5}])
    rows = list(LocalFilesystemSource().dlt_source(f"file://{tmp_path}/*.cbor", ""))
    assert sorted(r["id"] for r in rows) == [1, 2, 3, 4, 5]


# --- chunking contract (drives read_cbor directly with a small chunksize) ---


def test_chunks_at_boundary_and_flushes_partial(tmp_path):
    """Records are yielded in chunksize-sized chunks, with the partial final chunk flushed."""
    path = write_cbor(tmp_path / "five.cbor", [{"id": i} for i in range(5)])
    chunks = list(read_cbor(iter([FileItemStub(path)]), chunksize=2))  # ty: ignore[invalid-argument-type]
    assert [len(chunk) for chunk in chunks] == [2, 2, 1]
    assert [doc["id"] for chunk in chunks for doc in chunk] == [0, 1, 2, 3, 4]


def test_exact_chunksize_multiple_yields_no_empty_trailing_chunk(tmp_path):
    """A record count that is an exact multiple of chunksize emits no trailing empty chunk."""
    path = write_cbor(tmp_path / "four.cbor", [{"id": i} for i in range(4)])
    chunks = list(read_cbor(iter([FileItemStub(path)]), chunksize=2))  # ty: ignore[invalid-argument-type]
    assert [len(chunk) for chunk in chunks] == [2, 2]
    assert [doc["id"] for chunk in chunks for doc in chunk] == [0, 1, 2, 3]


def test_non_seekable_stream_is_spooled(tmp_path):
    """A non-seekable handle is spooled into a BytesIO so CBORIterable's seek(0) reset works
    (it otherwise silently yields nothing). The central stream-bridge claim, for CBOR."""
    data = cbor2.dumps([{"id": i} for i in range(3)])
    chunks = list(
        read_via_iterable(
            iter([NonSeekableItem(data)]),  # ty: ignore[invalid-argument-type]
            file_format="cbor",
            chunksize=1000,
        )
    )
    assert [doc["id"] for chunk in chunks for doc in chunk] == [0, 1, 2]


# --- registry / import-path / error UX ---


def test_cbor_uses_a_direct_decoder_not_iterabledata():
    """CBOR is read by decoding with cbor2 directly (not iterabledata's error-swallowing
    CBORIterable), so the registry entry carries an eager_decoder and no class_path."""
    spec = FORMAT_TO_ITERABLE["cbor"]
    assert spec.eager_decoder is not None
    assert spec.class_path is None
    assert spec.decoder_dist == "cbor2"


def test_truncated_cbor_raises_instead_of_silently_dropping_rows(tmp_path):
    """A corrupt/truncated CBOR file must raise, not load as zero rows (silent data loss).
    iterabledata's CBORIterable swallows the decode error; decoding via cbor2 surfaces it.
    Driven through the reader directly for the precise exception (the source path wraps it in
    dlt's ResourceExtractionError)."""
    full = cbor2.dumps([{"id": 1}, {"id": 2}, {"id": 3}])
    path = tmp_path / "truncated.cbor"
    path.write_bytes(full[: len(full) // 2])  # cut mid-stream
    with pytest.raises(cbor2.CBORDecodeError):
        list(read_cbor(iter([FileItemStub(path)]), chunksize=10))  # ty: ignore[invalid-argument-type]


def test_empty_cbor_file_yields_no_rows(tmp_path):
    """An empty file is not corrupt; it loads as zero rows (matching the other readers), so the
    truncation guard must not fire on it."""
    path = tmp_path / "empty.cbor"
    path.write_bytes(b"")
    assert _read_via_source(path) == []


def test_missing_decoder_raises_typed_install_hint(tmp_path, monkeypatch):
    """With the cbor2 decoder unavailable, the reader raises a typed error naming the exact
    `pip install`, not a bare ImportError."""
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        """Report the cbor2 decoder as absent, delegating other lookups to the real finder."""
        if name == "cbor2":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    path = write_cbor(tmp_path / "x.cbor", [{"id": 1}])
    with pytest.raises(MissingDecoderError) as exc:
        list(read_cbor(iter([FileItemStub(path)]), chunksize=10))  # ty: ignore[invalid-argument-type]
    message = str(exc.value)
    assert "cbor2" in message
    assert "dlt-filesystem[iterable]" in message


def test_cbor_advertised_only_when_decoder_installed(monkeypatch):
    """A base install must not advertise cbor as supported when its decoder is absent, even
    though the format stays routable."""
    from dlt_filesystem.source.format.registry import advertised_file_formats

    assert "cbor" in advertised_file_formats()

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        """Report the cbor2 decoder as absent, delegating other lookups to the real finder."""
        if name == "cbor2":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    assert "cbor" not in advertised_file_formats()


# --- the writer: what `file://out.cbor` writes reads back through the reader above ---

NAIVE = datetime.datetime(2020, 1, 2, 3, 4, 5)
PLUS_FIVE = datetime.timezone(datetime.timedelta(hours=5))
AWARE = datetime.datetime(2020, 1, 2, 3, 4, 5, 123456, tzinfo=PLUS_FIVE)
# 50 digits: past the 38 a Polars decimal column holds, which CBOR does not stop at.
WIDE_DECIMAL = decimal.Decimal("1234567890123456789012345678901234567890.1234567890")

# What each value a load can hand a writer reads back as. CBOR keeps every integer, a
# decimal, a date and an aware timestamp with its offset as themselves; it has no type
# for a time or a naive timestamp, so those come back as the strings the JSON writers
# write for them, and `bytes` as the base64 string the reader makes of them.
WRITTEN_AND_READ_BACK = {
    "a string": ("Zoë", "Zoë"),
    "the top of 64 bits": (2**64 - 1, 2**64 - 1),
    "the bottom of 64 bits": (-(2**63), -(2**63)),
    "past 64 bits": (2**70, 2**70),
    "past 64 bits, negative": (-(2**70), -(2**70)),
    "a float": (1.5, 1.5),
    "infinity": (math.inf, math.inf),
    "negative infinity": (-math.inf, -math.inf),
    "a boolean": (True, True),
    "null": (None, None),
    "a decimal": (decimal.Decimal("1.50"), decimal.Decimal("1.50")),
    "a decimal past 38 digits": (WIDE_DECIMAL, WIDE_DECIMAL),
    "a date": (datetime.date(2020, 1, 2), datetime.date(2020, 1, 2)),
    "a pendulum date": (pendulum.date(2020, 1, 2), datetime.date(2020, 1, 2)),
    "a time": (datetime.time(3, 4, 5, 6), "03:04:05.000006"),
    "an aware datetime": (AWARE, AWARE),
    "a pendulum datetime": (
        pendulum.datetime(2020, 1, 2, 3, 4, 5, tz="Pacific/Auckland"),
        datetime.datetime(
            2020, 1, 2, 3, 4, 5, tzinfo=datetime.timezone(datetime.timedelta(hours=13))
        ),
    ),
    "a naive datetime": (NAIVE, "2020-01-02T03:04:05"),
    "a naive datetime in a list": ([NAIVE], ["2020-01-02T03:04:05"]),
    "a naive datetime in a tuple": ((NAIVE,), ["2020-01-02T03:04:05"]),
    "a naive datetime in a dict": ({"at": NAIVE}, {"at": "2020-01-02T03:04:05"}),
    "bytes": (b"hi", "aGk="),
    "a nested document": ({"a": [1, {"b": None}]}, {"a": [1, {"b": None}]}),
    # The reader hands a key CBOR has a type for back as written.
    "an integer key": ({1: "x"}, {1: "x"}),
    "a bytes key": ({b"k": "x"}, {b"k": "x"}),
    # An offset that is not a whole number of minutes is written as the same instant in
    # UTC: cbor2 5.x decodes none of them, 6.x rejects one under a minute and moves the
    # instant of a negative one under a second. A historical zone's local mean time,
    # +00:19:32 here, is the ordinary way to meet one.
    "an offset with seconds": (
        datetime.datetime(1900, 1, 1, 12, tzinfo=zoneinfo.ZoneInfo("Europe/Amsterdam")),
        datetime.datetime(1900, 1, 1, 11, 40, 28, tzinfo=datetime.timezone.utc),
    ),
    "a negative offset under a second": (
        datetime.datetime(
            2020,
            1,
            1,
            tzinfo=datetime.timezone(-datetime.timedelta(microseconds=500000)),
        ),
        datetime.datetime(2020, 1, 1, 0, 0, 0, 500000, tzinfo=datetime.timezone.utc),
    ),
    "an offset of whole minutes": (
        datetime.datetime(
            2020,
            1,
            1,
            tzinfo=datetime.timezone(datetime.timedelta(hours=5, minutes=45)),
        ),
        datetime.datetime(
            2020,
            1,
            1,
            tzinfo=datetime.timezone(datetime.timedelta(hours=5, minutes=45)),
        ),
    ),
    "an offset under a minute": (
        datetime.datetime(
            2020, 1, 1, tzinfo=datetime.timezone(datetime.timedelta(seconds=30))
        ),
        datetime.datetime(2019, 12, 31, 23, 59, 30, tzinfo=datetime.timezone.utc),
    ),
    "an offset with a fraction of a second": (
        datetime.datetime(
            2020,
            1,
            1,
            tzinfo=datetime.timezone(datetime.timedelta(hours=1, microseconds=5)),
        ),
        datetime.datetime(
            2019, 12, 31, 22, 59, 59, 999995, tzinfo=datetime.timezone.utc
        ),
    ),
}


def _write_and_read(path, rows):
    writer_for_format("cbor")(str(path), rows)
    return _read_via_source(path)


@pytest.mark.parametrize(
    ("value", "expected"), WRITTEN_AND_READ_BACK.values(), ids=WRITTEN_AND_READ_BACK
)
def test_write_round_trips_through_the_reader(tmp_path, value, expected):
    [row] = _write_and_read(tmp_path / "out.cbor", [{"v": value}])

    assert row == {"v": expected}
    # `True == 1`, `Decimal("1.50") == 1.5` and one instant equals itself in any zone,
    # so the type, the scale and the offset are asserted as well as the value.
    assert type(row["v"]) is type(expected)
    if isinstance(expected, decimal.Decimal):
        assert str(row["v"]) == str(expected)
    if isinstance(expected, datetime.datetime):
        assert row["v"].utcoffset() == expected.utcoffset()


def test_write_keeps_nan_a_float(tmp_path):
    """NaN compares unequal to itself, so it is asserted apart from the table above."""
    [row] = _write_and_read(tmp_path / "out.cbor", [{"v": math.nan}])

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

    assert _write_and_read(tmp_path / "out.cbor", rows) == expected


def test_write_of_no_rows_is_the_empty_array(tmp_path):
    """The reader cannot tell an empty array from an empty file, so this pins what the
    file says: an empty array is a document every CBOR decoder reads as one."""
    path = tmp_path / "out.cbor"
    writer_for_format("cbor")(str(path), [])

    assert path.read_bytes() == b"\x80"


@pytest.mark.parametrize(
    "value",
    [
        datetime.datetime(
            1, 1, 1, tzinfo=datetime.timezone(datetime.timedelta(minutes=1, seconds=1))
        ),
        datetime.datetime(
            9999,
            12,
            31,
            23,
            59,
            59,
            tzinfo=datetime.timezone(-datetime.timedelta(minutes=1, seconds=1)),
        ),
    ],
    ids=["before year 1 in UTC", "after year 9999 in UTC"],
)
def test_write_refuses_an_offset_it_cannot_move_to_utc(tmp_path, value):
    """An offset of seconds is written in UTC, and these two have no UTC datetime."""
    path = tmp_path / "out.cbor"

    with pytest.raises(ValueError, match="Column 'v'"):
        writer_for_format("cbor")(str(path), [{"v": [value]}])

    assert not path.exists()


def test_write_emits_one_array_document(tmp_path):
    """A sequence of items would read back as its first row only. Decoded from a stream
    so a second item after the array is seen, which `cbor2.loads` would ignore."""
    path = tmp_path / "out.cbor"
    writer_for_format("cbor")(str(path), [{"id": 1}, {"id": 2}])

    with path.open("rb") as handle:
        assert cbor2.CBORDecoder(handle).decode() == [{"id": 1}, {"id": 2}]
        assert handle.read() == b"", "nothing follows the array"


# Spelling one of these as a string could land on a string key already in the map and
# drop that entry's value: a time key `03:04:00` beside the string key `"03:04:00"`.
REFUSED_KEYS = {
    "a time beside its own spelling": [
        {"v": {datetime.time(3, 4): "a", "03:04:00": "b"}}
    ],
    "a naive datetime column name": [{NAIVE: "x"}],
    "a tuple key in a list": [{"v": [{(1, 2): "x"}]}],
    # A mapping that is not a dict is walked the same way.
    "a time key in a UserDict": [
        {"v": collections.UserDict({datetime.time(3, 4): "a", "03:04:00": "b"})}
    ],
}


@pytest.mark.parametrize("rows", REFUSED_KEYS.values(), ids=REFUSED_KEYS)
def test_write_refuses_a_key_of_another_type(tmp_path, rows):
    path = tmp_path / "out.cbor"

    with pytest.raises(ValueError, match="keeps only string, bytes"):
        writer_for_format("cbor")(str(path), rows)

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
    """Spelled whole through JSON, its NaN would be written as null and its integer key
    as a string. Which layer refuses the NaN depends on dlt's JSON backend (simplejson
    raises on it itself), so this pins the refusal and the untouched path; the finite
    case reaches the document check under every backend."""
    path = tmp_path / "out.cbor"

    with pytest.raises((TypeError, ValueError)):
        writer_for_format("cbor")(str(path), [{"v": document}])

    assert not path.exists()


def test_a_failed_write_keeps_the_file_already_there(tmp_path):
    """Nothing is written until the whole load is encoded."""
    path = tmp_path / "out.cbor"
    writer_for_format("cbor")(str(path), [{"id": 0}])
    before = path.read_bytes()

    with pytest.raises((TypeError, ValueError)):
        writer_for_format("cbor")(str(path), [{"id": 1}, {"v": object()}])

    assert path.read_bytes() == before


def test_write_without_cbor2_names_the_install(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "cbor2", None)

    with pytest.raises(MissingDecoderError, match=r"dlt-filesystem\[iterable\]"):
        writer_for_format("cbor")(str(tmp_path / "out.cbor"), [{"id": 1}])


def test_write_leaves_the_rows_it_was_given_alone(tmp_path):
    """The naive-datetime pass builds new containers rather than spelling in place."""
    rows = [{"at": NAIVE, "list": [NAIVE], "map": {"at": NAIVE}}]
    writer_for_format("cbor")(str(tmp_path / "out.cbor"), rows)

    assert rows == [{"at": NAIVE, "list": [NAIVE], "map": {"at": NAIVE}}]


def test_write_destination_round_trips_without_dlt_columns(tmp_path):
    """Through `post_load`, from a staged JSONL file, the way a pipeline reaches it."""
    destination = LocalFilesystemDestination()
    output_path = tmp_path / "out.cbor"
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
