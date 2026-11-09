"""Apache Avro object containers: reading them, and writing them on Polars 2.1 and later.

The read fixtures are hand-built (`testing.writer.write_avro`) rather than produced by
`polars.DataFrame.write_avro`. A fixture written by the library under test can only
contain what that library can *write*, and the reader's sharpest failure is on input it
cannot: an Avro `map` field is an ordinary type that `polars.read_avro` panics on.

The write side is the last section. Polars before 2.1 mis-frames a list column holding
an empty list before a non-empty one, and Polars 2.1 needs Python 3.11, so the writer
refuses on an earlier Polars and the tests split by the installed version: CI's 3.10 leg
proves the refusal and its 3.11+ legs prove the file.
"""

import datetime
import decimal
import gzip
import json
import math
import os
import re
import struct
import sys
import zoneinfo
from pathlib import Path

import polars as pl
import pytest
from dlt.common.pendulum import pendulum
from dlt.extract.exceptions import ResourceExtractionError
from verlib2 import Version

from dlt_filesystem.source.error import MissingDecoderError
from dlt_filesystem.source.format.readers import read_avro
from dlt_filesystem.source.fsspec.local import LocalFilesystemSource
from dlt_filesystem.target.local import LocalFilesystemDestination
from dlt_filesystem.target.registry import writer_for_format
from dlt_filesystem.target.util import _resolve_output_target
from dlt_filesystem.target.writer import write_jsonl
from dlt_filesystem.testing.stub import FileItemStub
from dlt_filesystem.testing.writer import (
    encode_avro_bytes,
    encode_avro_long,
    write_avro,
    write_avro_container,
)

# Polars 2.1 renders this error with its detail (`out-of-spec: Avro header ends early`);
# earlier releases render only the variant name.
_OUT_OF_SPEC = re.compile(r"OutOfSpec|out-of-spec")


def _read_via_source(path, suffix=""):
    """Read a local Avro file end-to-end through the shared filesystem reader."""
    return list(LocalFilesystemSource().dlt_source(f"file://{path}{suffix}", ""))


# --- end-to-end reader (fsspec, no Docker) ---


def test_read_single_row(tmp_path):
    path = write_avro(tmp_path / "one.avro", [{"id": 1, "name": "alice"}])
    assert _read_via_source(path) == [{"id": 1, "name": "alice"}]


def test_read_multiple_rows(tmp_path):
    docs = [{"id": i, "name": n} for i, n in enumerate(["a", "b", "c"], start=1)]
    path = write_avro(tmp_path / "arr.avro", docs)
    rows = _read_via_source(path)
    assert [row["id"] for row in rows] == [1, 2, 3]
    assert sorted(row["name"] for row in rows) == ["a", "b", "c"]


def test_read_sparse_rows(tmp_path):
    """A column missing from a row is a `["null", T]` union, and loads as null."""
    path = write_avro(
        tmp_path / "sparse.avro", [{"id": 1, "note": "here"}, {"id": 2, "note": None}]
    )
    assert _read_via_source(path) == [
        {"id": 1, "note": "here"},
        {"id": 2, "note": None},
    ]


def test_the_extension_resolves_to_the_reader(tmp_path):
    path = write_avro(tmp_path / "by_ext.avro", [{"id": 1}, {"id": 2}])
    assert len(_read_via_source(path)) == 2


def test_the_format_hint_resolves_to_the_reader(tmp_path):
    """A bare `#avro`, not `#format=avro`: a `key=value` fragment segment is a named
    reader argument, and only a bare token is matched against the format map."""
    path = write_avro(tmp_path / "feed.dat", [{"id": 1}, {"id": 2}])
    assert len(_read_via_source(path, "#avro")) == 2


def test_a_gzipped_file_reads(tmp_path):
    raw = write_avro(tmp_path / "data.avro", [{"id": 1}, {"id": 2}])
    path = tmp_path / "gz.avro.gz"
    path.write_bytes(gzip.compress(Path(raw).read_bytes()))

    assert _read_via_source(path) == [{"id": 1}, {"id": 2}]


def test_read_multiple_files(tmp_path):
    write_avro(tmp_path / "a.avro", [{"id": 1}, {"id": 2}, {"id": 3}])
    write_avro(tmp_path / "b.avro", [{"id": 4}, {"id": 5}])
    rows = list(LocalFilesystemSource().dlt_source(f"file://{tmp_path}/*.avro", ""))
    assert sorted(row["id"] for row in rows) == [1, 2, 3, 4, 5]


# --- reader hints ---


def test_read_with_columns_single(tmp_path):
    path = write_avro(tmp_path / "one.avro", [{"id": 1, "name": "alice", "age": 44}])
    assert _read_via_source(path, "#columns=name") == [{"name": "alice"}]


def test_read_with_columns_json(tmp_path):
    """The spelling a multi-column hint actually arrives as: a JSON list, as a string.

    Decoded rather than passed through, because `polars.read_avro` takes a bare string
    as a single column *name*, so the undecoded hint would be looked up as a column
    literally called `["id","name"]`.
    """
    path = write_avro(tmp_path / "one.avro", [{"id": 1, "name": "alice", "age": 44}])
    columns = json.dumps(["id", "name"])
    assert _read_via_source(path, f"#columns={columns}") == [{"id": 1, "name": "alice"}]


def test_read_with_columns_unknown(tmp_path):
    """Polars' own message, which already names the valid columns, is left as it is.

    ORC and Feather share a reworded error because both come from PyArrow, which names
    the column and not the alternatives. This reader is Polars-backed and its message
    is complete, so translating it would only make the two libraries lie about being
    one.
    """
    path = write_avro(tmp_path / "one.avro", [{"id": 1, "name": "alice", "age": 44}])
    with pytest.raises(ResourceExtractionError) as excinfo:
        _read_via_source(path, "#columns=unknown")
    assert excinfo.match('unable to find column "unknown"')
    assert excinfo.match(re.escape('valid columns: ["id", "name", "age"]'))


def test_read_with_chunksize(tmp_path):
    path = write_avro(tmp_path / "data.avro", [{"id": i} for i in range(5)])
    chunks = list(
        read_avro(iter([FileItemStub(path)]), chunksize=2)  # ty: ignore[invalid-argument-type]
    )
    assert [len(chunk) for chunk in chunks] == [2, 2, 1]
    assert [row["id"] for chunk in chunks for row in chunk] == list(range(5))


def test_read_with_chunksize_invalid(tmp_path):
    """A hint arrives as a string, so the cast is the validation.

    Without it a literal slicing loop fails on a string, and a negative step yields
    nothing at all -- a silent empty read that looks like an empty source.
    """
    path = write_avro(tmp_path / "one.avro", [{"id": 1}])
    with pytest.raises(ResourceExtractionError) as excinfo:
        _read_via_source(path, "#chunksize=foo")
    assert excinfo.match("chunksize must be an integer, not foo")


@pytest.mark.parametrize("chunksize", [0, -1])
def test_read_with_non_positive_chunksize(tmp_path, chunksize):
    path = write_avro(tmp_path / "one.avro", [{"id": 1}])
    with pytest.raises(ResourceExtractionError) as excinfo:
        _read_via_source(path, f"#chunksize={chunksize}")
    assert excinfo.match(f"chunksize must be greater than zero, not {chunksize}")


def test_read_with_invalid_option(tmp_path):
    """An unknown hint raises rather than being swallowed by a `**kwargs` catch-all."""
    path = write_avro(tmp_path / "one.avro", [{"id": 1}])
    with pytest.raises(TypeError) as excinfo:
        _read_via_source(path, "#invalid=true")
    assert excinfo.match(
        re.escape("read_avro(): got an unexpected keyword argument 'invalid'")
    )


def test_the_reader_is_whole_file(tmp_path):
    """`filesystem.md` says so, and this is the fact behind it.

    Polars has no `scan_avro`, so `chunksize` bounds what a downstream step is handed
    at once and not what is held in memory. Pinned rather than commented, so the page's
    claim fails here if Polars ever adds one.
    """
    assert not hasattr(pl, "scan_avro")


# --- the type matrix ---


def test_the_container_carries_every_type_the_page_tabulates(tmp_path):
    """Written with Polars, deliberately: this half is about what a Polars-written Avro
    file loads back as, which is what the format page's type table claims. The
    hand-built fixtures elsewhere in this file cover the input Polars cannot produce."""
    frame = pl.DataFrame(
        {
            "i": [1],
            "s": ["Zoë"],
            "f": [1.5],
            "b": [True],
            "date": [datetime.date(2020, 1, 1)],
            "naive": [datetime.datetime(2020, 1, 2, 3, 4, 5, 123456)],
            "blob": [b"hi"],
            "lst": [[1, 2]],
            "st": [{"n": 1}],
        }
    ).with_columns(pl.Series("dec", [decimal.Decimal("3.14")], dtype=pl.Decimal(38, 2)))
    path = tmp_path / "types.avro"
    frame.write_avro(str(path))

    assert _read_via_source(path) == [
        {
            "i": 1,
            "s": "Zoë",
            "f": 1.5,
            "b": True,
            "date": datetime.date(2020, 1, 1),
            "naive": datetime.datetime(2020, 1, 2, 3, 4, 5, 123456),
            "blob": b"hi",
            "lst": [1, 2],
            "st": {"n": 1},
            "dec": decimal.Decimal("3.14"),
        }
    ]


#: Every row of the type table on `docs/supported-sources/avro.md`, as (label, Avro field
#: type, encoded value, loaded value). Built by hand rather than round-tripped, because
#: half of these are types the writer side cannot produce: a `fixed`, an `enum`, a
#: millisecond timestamp and a decimal carried in `fixed` bytes all reach a reader from
#: other producers and from no test that writes its own fixture.
AVRO_TYPES = [
    ("int", "int", encode_avro_long(42), 42),
    ("long", "long", encode_avro_long(2**40), 2**40),
    ("float", "float", struct.pack("<f", 1.5), 1.5),
    ("double", "double", struct.pack("<d", 1.5), 1.5),
    ("boolean", "boolean", b"\x01", True),
    ("bytes", "bytes", encode_avro_bytes(b"hi"), b"hi"),
    ("fixed", {"type": "fixed", "name": "F", "size": 4}, b"abcd", b"abcd"),
    (
        "enum",
        {"type": "enum", "name": "E", "symbols": ["A", "B"]},
        encode_avro_long(1),
        "B",
    ),
    (
        "date",
        {"type": "int", "logicalType": "date"},
        encode_avro_long(18262),
        datetime.date(2020, 1, 1),
    ),
    (
        "timestamp-micros",
        {"type": "long", "logicalType": "timestamp-micros"},
        encode_avro_long(1577923445123456),
        datetime.datetime(2020, 1, 2, 0, 4, 5, 123456, tzinfo=datetime.timezone.utc),
    ),
    (
        "timestamp-millis",
        {"type": "long", "logicalType": "timestamp-millis"},
        encode_avro_long(1577923445123),
        datetime.datetime(2020, 1, 2, 0, 4, 5, 123000, tzinfo=datetime.timezone.utc),
    ),
    (
        "time-micros",
        {"type": "long", "logicalType": "time-micros"},
        encode_avro_long(34200000000),
        datetime.time(9, 30),
    ),
    (
        "uuid",
        {"type": "string", "logicalType": "uuid"},
        encode_avro_bytes(b"3f2504e0-4f89-11d3-9a0c-0305e82c3301"),
        "3f2504e0-4f89-11d3-9a0c-0305e82c3301",
    ),
    (
        "decimal in bytes",
        {"type": "bytes", "logicalType": "decimal", "precision": 10, "scale": 2},
        encode_avro_bytes((314).to_bytes(2, "big")),
        decimal.Decimal("3.14"),
    ),
    (
        "decimal in fixed",
        {
            "type": "fixed",
            "name": "D",
            "size": 4,
            # Four bytes hold at most nine decimal digits under the spec
            # (floor(log10(2**31 - 1)) == 9), so a precision of 10 here would be an
            # invalid declaration that only passes because Polars does not check it.
            "logicalType": "decimal",
            "precision": 9,
            "scale": 2,
        },
        (314).to_bytes(4, "big"),
        decimal.Decimal("3.14"),
    ),
    (
        "array",
        {"type": "array", "items": "long"},
        encode_avro_long(2)
        + encode_avro_long(1)
        + encode_avro_long(2)
        + encode_avro_long(0),
        [1, 2],
    ),
    (
        "record",
        {"type": "record", "name": "S", "fields": [{"name": "n", "type": "long"}]},
        encode_avro_long(7),
        {"n": 7},
    ),
    ("union, set", ["null", "long"], encode_avro_long(1) + encode_avro_long(9), 9),
    ("union, null", ["null", "long"], encode_avro_long(0), None),
]


@pytest.mark.parametrize(
    ("field_type", "encoded", "loaded"),
    [case[1:] for case in AVRO_TYPES],
    ids=[case[0] for case in AVRO_TYPES],
)
def test_every_avro_type_on_the_page_loads_as_documented(
    tmp_path, field_type, encoded, loaded
):
    """The type table on the format page, asserted rather than read off the spec."""
    path = write_avro_container(
        tmp_path / "typed.avro",
        {"type": "record", "name": "r", "fields": [{"name": "v", "type": field_type}]},
        encoded,
        record_count=1,
    )
    row = _read_via_source(path)[0]
    assert row["v"] == loaded
    assert type(row["v"]) is type(loaded)


# --- Avro schemas Polars cannot map to Arrow ---


def test_a_map_field_fails_by_name_rather_than_as_a_rust_panic(tmp_path):
    """The reader's worst input, and no round-trip suite could hold it.

    `polars.read_avro` maps Avro onto Arrow, and it has no mapping for `map`: it aborts
    in Rust with `PanicException`, which inherits `BaseException` directly. dlt's
    extract step wraps `Exception`, so an unconverted panic escapes every handler above
    this reader and ends the load in a backtrace naming neither the file nor the field.
    A `map` is an ordinary Avro type, so this arrives with a perfectly valid file.
    """
    path = write_avro_container(
        tmp_path / "map.avro",
        {
            "type": "record",
            "name": "r",
            "fields": [{"name": "m", "type": {"type": "map", "values": "long"}}],
        },
        b"\x02\x02k\x0e\x00",
        record_count=1,
    )

    with pytest.raises(ResourceExtractionError) as excinfo:
        _read_via_source(path)
    assert excinfo.match("aborted inside Polars")
    assert excinfo.match("Avro maps are mapped to MapArrays")
    assert excinfo.match("map.avro")


def test_a_map_field_is_named_without_leaking_credentials(tmp_path):
    """The file is named from the listed URL, which can carry credentials on a remote
    transport, so it goes through the same redaction every other reader error uses."""
    path = write_avro_container(
        tmp_path / "map.avro",
        {
            "type": "record",
            "name": "r",
            "fields": [{"name": "m", "type": {"type": "map", "values": "long"}}],
        },
        b"\x02\x02k\x0e\x00",
        record_count=1,
    )
    stub = FileItemStub(path, file_url="s3://key:secret@bucket/map.avro")

    with pytest.raises(ValueError) as excinfo:
        list(read_avro(iter([stub])))  # ty: ignore[invalid-argument-type]
    assert "secret" not in str(excinfo.value)
    assert "s3://bucket/map.avro" in str(excinfo.value)


def test_a_timestamp_past_year_9999_fails_by_name_rather_than_as_a_rust_panic(tmp_path):
    """The second panic, and it is not on the read.

    `timestamp-micros` is a 64-bit offset from the epoch, so it reaches years Python's
    `datetime` cannot hold. Such a value decodes into the frame without complaint --
    Polars prints it as `+10000-01-01` -- and panics only on the way out to Python
    objects, which is a different call from the one the schema panic comes out of. A
    handler around the read alone leaves this one escaping.
    """
    path = write_avro_container(
        tmp_path / "far.avro",
        {
            "type": "record",
            "name": "r",
            "fields": [
                {
                    "name": "t",
                    "type": {"type": "long", "logicalType": "timestamp-micros"},
                }
            ],
        },
        encode_avro_long(253402300800000000),  # 10000-01-01T00:00:00Z
        record_count=1,
    )

    with pytest.raises(ResourceExtractionError) as excinfo:
        _read_via_source(path)
    assert excinfo.match("`read_avro` caused an exception")
    assert excinfo.match("date value out of range")


def test_a_panic_on_conversion_is_an_ordinary_exception(tmp_path):
    """What the conversion buys, stated as the property rather than as the message.

    `PanicException` inherits `BaseException`, so dlt's extract step -- which wraps
    `Exception` -- does not see it and the load ends in a Rust backtrace. This asserts
    the raised type is one ordinary handling catches, which is the whole point.
    """
    path = write_avro_container(
        tmp_path / "far.avro",
        {
            "type": "record",
            "name": "r",
            "fields": [
                {
                    "name": "t",
                    "type": {"type": "long", "logicalType": "timestamp-micros"},
                }
            ],
        },
        encode_avro_long(253402300800000000),
        record_count=1,
    )

    with pytest.raises(Exception) as excinfo:  # noqa: B017, PT011
        list(read_avro(iter([FileItemStub(path)])))  # ty: ignore[invalid-argument-type]
    assert isinstance(excinfo.value, OverflowError)


@pytest.mark.parametrize(
    ("label", "field_type"),
    [
        ("null field", "null"),
        ("multi-branch union", ["null", "long", "string"]),
    ],
)
def test_schemas_polars_rejects_catchably_stay_catchable(tmp_path, label, field_type):
    """The other two unmapped schemas raise a `ComputeError`, which is an ordinary
    `Exception` and needs no conversion. Pinned so a Polars change that turns one of
    them into a panic shows up here rather than in a load."""
    path = write_avro_container(
        tmp_path / "odd.avro",
        {"type": "record", "name": "r", "fields": [{"name": "f", "type": field_type}]},
        b"\x02\x0a" if field_type != "null" else b"",
        record_count=1,
    )
    with pytest.raises(ResourceExtractionError) as excinfo:
        _read_via_source(path)
    assert excinfo.match("not yet implemented")


# --- damaged files ---


@pytest.mark.parametrize(
    ("name", "content"), [("empty.avro", b""), ("garbage.avro", b"not avro at all")]
)
def test_a_damaged_file_raises_rather_than_loading_partial_data(
    tmp_path, name, content
):
    path = tmp_path / name
    path.write_bytes(content)
    with pytest.raises(ResourceExtractionError) as excinfo:
        _read_via_source(path)
    assert excinfo.match(_OUT_OF_SPEC)


def _two_block_container(path, first, second):
    """An Avro container carrying two data blocks, which one `write_avro` cannot make."""
    schema = {"type": "record", "name": "r", "fields": [{"name": "id", "type": "long"}]}
    body = bytearray(b"Obj\x01")
    body += encode_avro_long(2)
    body += encode_avro_bytes(b"avro.schema") + encode_avro_bytes(
        json.dumps(schema).encode()
    )
    body += encode_avro_bytes(b"avro.codec") + encode_avro_bytes(b"null")
    body += encode_avro_long(0)
    body += _SYNC
    for ids in (first, second):
        block = b"".join(encode_avro_long(i) for i in ids)
        body += (
            encode_avro_long(len(ids)) + encode_avro_long(len(block)) + block + _SYNC
        )
    path.write_bytes(bytes(body))
    return path


#: The sync marker `write_avro_container` uses, needed here to find a block boundary.
_SYNC = b"0123456789abcdef"


def test_a_truncated_tail_separates_the_format_limit_from_the_reader_limit(tmp_path):
    """Documented on the format page rather than guarded, and pinned here as measured.

    Two different things, which the first draft of the page conflated. Cutting a container
    exactly at a block boundary leaves a **shorter valid file**: Avro carries no trailing
    index or record count, so nothing distinguishes it from the original, and no reader
    could. Cutting one byte further leaves a partial record count, which *is* detectable
    corruption, and Polars accepts it as end of file anyway -- that one is a limit of the
    reader, not of the format. Two bytes further it raises, so the silent window past a
    boundary is exactly one byte wide.

    Pinned so the page's claim is measured, and so a reader that starts validating block
    framing has to change the middle case here deliberately.
    """
    path = tmp_path / "two.avro"
    _two_block_container(path, [0, 1], list(range(2, 130)))
    raw = path.read_bytes()
    assert len(_read_via_source(path)) == 130, "the intact file carries both blocks"

    boundary = raw.index(_SYNC, raw.index(_SYNC) + len(_SYNC)) + len(_SYNC)
    cut = tmp_path / "cut.avro"

    # Exactly at the boundary: a shorter valid container, and inherently undetectable.
    cut.write_bytes(raw[:boundary])
    assert [row["id"] for row in _read_via_source(cut)] == [0, 1]

    # One byte in: a partial record count, which a reader could reject and this one does not.
    cut.write_bytes(raw[: boundary + 1])
    assert [row["id"] for row in _read_via_source(cut)] == [0, 1]

    # Two bytes in: a complete count and a partial size, which does raise.
    cut.write_bytes(raw[: boundary + 2])
    with pytest.raises(ResourceExtractionError) as excinfo:
        _read_via_source(cut)
    assert excinfo.match(_OUT_OF_SPEC)


def test_a_container_with_no_records_yields_no_rows(tmp_path):
    """A record schema with no fields and no data blocks is a valid, empty container."""
    path = write_avro_container(
        tmp_path / "none.avro", {"type": "record", "name": "root", "fields": []}
    )
    assert _read_via_source(path) == []


# --- writing ---

#: Polars 2.1 fixed the list framing (pola-rs/polars#30061), and the writer refuses below it.
_POLARS_WRITES_AVRO = Version(pl.__version__) >= Version("2.1")
needs_the_writer = pytest.mark.skipif(
    not _POLARS_WRITES_AVRO,
    reason=f"Polars {pl.__version__} predates 2.1, where the writer refuses",
)
needs_an_early_polars = pytest.mark.skipif(
    _POLARS_WRITES_AVRO,
    reason=f"Polars {pl.__version__} writes Avro; the refusal is for releases before 2.1",
)


def _write(path, rows):
    writer_for_format("avro")(str(path), rows)
    return path


def _decode_avro_records(raw: bytes) -> list[list]:
    """Decode an object container by hand, with no Polars in the loop.

    Deliberately minimal and deliberately independent: the list-framing defect is a
    malformed *file*, and asking Polars' own reader whether Polars' own writer produced
    a good file cannot establish that. Reads the nullable `long`, `double`, `string` and
    array-of-nullable-`string` fields the shapes below use, and nothing else.
    """
    position = 0

    def read_long() -> int:
        nonlocal position
        shift = result = 0
        while True:
            byte = raw[position]
            position += 1
            result |= (byte & 0x7F) << shift
            if not byte & 0x80:
                return (result >> 1) ^ -(result & 1)
            shift += 7

    def read_bytes() -> bytes:
        nonlocal position
        length = read_long()
        position += length
        return raw[position - length : position]

    def read_double() -> float:
        nonlocal position
        position += 8
        return struct.unpack("<d", raw[position - 8 : position])[0]

    def read_block_count() -> int:
        """One block header of a map or array: a count, negative when a byte size follows.

        Both spellings are legal and both are in play here -- Polars writes its metadata
        map in the negative form -- so a decoder that reads only the positive one finds
        an empty map and blames the schema for being absent.
        """
        count = read_long()
        if count < 0:
            read_long()  # the block's byte size, present only in this spelling
            return -count
        return count

    assert raw[:4] == b"Obj\x01"
    position = 4
    metadata = {}
    while (entries := read_block_count()) != 0:
        for _ in range(entries):
            # Named rather than written as a subscript assignment: Python evaluates the
            # right-hand side first, so `metadata[read_bytes()] = read_bytes()` reads
            # the value out of the stream before the key.
            key = read_bytes().decode()
            metadata[key] = read_bytes()
    position += 16  # sync marker

    fields = json.loads(metadata["avro.schema"])["fields"]
    count = read_long()
    body_length = read_long()
    body_start = position
    body_end = position + body_length
    records = []
    for _ in range(count):
        record = []
        for field in fields:
            branch = (
                field["type"][1] if isinstance(field["type"], list) else field["type"]
            )
            if isinstance(field["type"], list) and read_long() == 0:
                record.append(None)
                continue
            if branch == "long":
                record.append(read_long())
            elif branch == "double":
                record.append(read_double())
            elif isinstance(branch, dict) and branch["type"] == "array":
                items = []
                while (block := read_block_count()) != 0:
                    for _ in range(block):
                        items.append(
                            None if read_long() == 0 else read_bytes().decode()
                        )
                record.append(items)
            else:
                record.append(read_bytes().decode())
        records.append(record)
    assert position == body_end, (
        f"the block declares {body_length} bytes and its {count} records consume "
        f"{position - body_start}"
    )
    return records


@needs_the_writer
def test_the_writer_frames_a_list_column_with_an_empty_list_last(tmp_path):
    """The control for the case below: an empty list in the *last* row wrote correctly
    on every Polars, so the defect is about position rather than about empty lists."""
    path = _write(tmp_path / "ok.avro", [{"id": 1, "lst": ["a"]}, {"id": 2, "lst": []}])

    assert _decode_avro_records(path.read_bytes()) == [[1, ["a"]], [2, []]]


#: The shapes Polars before 2.1 wrote as a malformed file. With an `Int64` sibling the
#: overrun block is caught as a short read; with a `Float64` one the misaligned bytes
#: decode as a valid float, so the file reads back without error and `2.0` comes back as
#: `7.7e-319`. Measured on polars 1.34.0, 1.39.3, 1.43.1, 2.0.0rc1 and 2.0.0.
MISFRAMED_BEFORE_2_1 = {
    "beside a long": (
        [{"id": 1, "lst": []}, {"id": 2, "lst": ["a"]}],
        [[1, []], [2, ["a"]]],
    ),
    "beside a double": (
        [{"f": 1.0, "lst": []}, {"f": 2.0, "lst": ["a"]}],
        [[1.0, []], [2.0, ["a"]]],
    ),
}


@needs_the_writer
@pytest.mark.parametrize(
    ("rows", "records"), MISFRAMED_BEFORE_2_1.values(), ids=MISFRAMED_BEFORE_2_1
)
def test_the_writer_frames_a_list_column_with_an_empty_list_first(
    tmp_path, rows, records
):
    """`{"tags": []}` then `{"tags": ["x"]}` is ordinary JSON and reaches a writer as this
    frame on the default load path. Decoded independently, and read back."""
    path = _write(tmp_path / "out.avro", rows)

    assert _decode_avro_records(path.read_bytes()) == records
    assert _read_via_source(path) == rows


@needs_an_early_polars
@pytest.mark.parametrize(
    ("rows", "records"), MISFRAMED_BEFORE_2_1.values(), ids=MISFRAMED_BEFORE_2_1
)
def test_the_writer_refuses_on_a_polars_that_misframes_lists(tmp_path, rows, records):
    """On the installed Polars itself, not a patched version string: this is the 3.10
    leg, which resolves Polars 2.0."""
    path = tmp_path / "out.avro"
    path.write_bytes(b"already here")

    with pytest.raises(MissingDecoderError) as excinfo:
        _write(path, rows)

    assert "Polars 2.1 or later" in str(excinfo.value)
    assert f"Polars {pl.__version__}" in str(excinfo.value)
    assert path.read_bytes() == b"already here"


@pytest.mark.parametrize("version", ["2.0.0", "1.34.0", "2.1.0rc1"])
def test_the_version_gate_refuses_before_writing_anything(
    tmp_path, monkeypatch, version
):
    """The gate on every leg. A pre-release of 2.1 is refused: the writer cannot know
    whether it carries the fix."""
    monkeypatch.setattr(pl, "__version__", version)
    path = tmp_path / "out.avro"

    with pytest.raises(MissingDecoderError, match=r"Python 3\.11") as excinfo:
        _write(path, [{"id": 1}])

    assert f"Polars {version}" in str(excinfo.value)
    assert "polars>=2.1" in str(excinfo.value)
    assert not path.exists()


@pytest.mark.skipif(
    not os.environ.get("CI"), reason="pins the CI matrix, not a local one"
)
def test_the_ci_matrix_still_runs_both_sides_of_the_version_gate():
    """The tests above split by the installed Polars, so a matrix that stopped resolving
    one side would skip that half and stay green. Python 3.10 resolves Polars 2.0 because
    2.1 needs 3.11; every later leg resolves 2.1 or newer."""
    assert _POLARS_WRITES_AVRO == (sys.version_info >= (3, 11))


def test_an_avro_destination_resolves(tmp_path):
    """On every Python, including one whose Polars refuses at write time."""
    assert _resolve_output_target("file:///data/out.avro") == ("/data/out.avro", "avro")
    assert _resolve_output_target("file:///data/out.dat#avro")[1] == "avro"


UTC = datetime.timezone.utc
AWARE = datetime.datetime(2020, 1, 2, 3, 4, 5, 123456, tzinfo=UTC)
WALL_TIME_IN_UTC = datetime.datetime(2020, 1, 2, 3, 4, 5, 123456)

#: Value written, value read back. Rows go through `writer_for_format` and the source
#: reader, and the expectation is written from the input, not from a Polars frame.
WRITTEN_AND_READ_BACK = {
    "text": ("Zoë", "Zoë"),
    "an integer at the bottom of a long": (-(2**63), -(2**63)),
    "an integer at the top of a long": (2**63 - 1, 2**63 - 1),
    "a float": (1.5, 1.5),
    "a boolean": (True, True),
    "a decimal": (decimal.Decimal("1.50"), decimal.Decimal("1.50")),
    "a date": (datetime.date(2020, 1, 2), datetime.date(2020, 1, 2)),
    "a naive datetime": (WALL_TIME_IN_UTC, WALL_TIME_IN_UTC),
    "a datetime in UTC": (AWARE, WALL_TIME_IN_UTC),
    "a datetime at a fixed offset": (
        datetime.datetime(
            2020,
            1,
            2,
            15,
            4,
            5,
            123456,
            tzinfo=datetime.timezone(datetime.timedelta(hours=12)),
        ),
        WALL_TIME_IN_UTC,
    ),
    "a datetime in a named zone": (
        datetime.datetime(
            2020, 1, 2, 16, 4, 5, 123456, tzinfo=zoneinfo.ZoneInfo("Pacific/Auckland")
        ),
        WALL_TIME_IN_UTC,
    ),
    "a pendulum datetime": (pendulum.instance(AWARE), WALL_TIME_IN_UTC),
    "a time of day": (datetime.time(3, 4, 5, 6), "03:04:05.000006"),
    "bytes": (b"hi", b"hi"),
    "a struct": ({"a": 1, "b": "x"}, {"a": 1, "b": "x"}),
    "a list": (["a", None], ["a", None]),
    "a nested list": ([[1], [2, 3]], [[1], [2, 3]]),
    "a list of structs": ([{"a": 1}, {"a": 2}], [{"a": 1}, {"a": 2}]),
    "an aware datetime in a struct": ({"at": AWARE}, {"at": WALL_TIME_IN_UTC}),
    "an aware datetime in a list": ([AWARE], [WALL_TIME_IN_UTC]),
    "a time of day in a list": ([datetime.time(3, 4)], ["03:04:00"]),
}


@needs_the_writer
@pytest.mark.parametrize(
    ("value", "expected"), WRITTEN_AND_READ_BACK.values(), ids=WRITTEN_AND_READ_BACK
)
def test_write_round_trips_through_the_reader(tmp_path, value, expected):
    [row] = _read_via_source(_write(tmp_path / "out.avro", [{"v": value}]))

    assert row == {"v": expected}
    # `True == 1`, and an aware datetime never equals a naive one, so the type and the
    # absence of a zone are asserted as well as the value.
    assert type(row["v"]) is type(expected) or isinstance(row["v"], type(expected))
    if isinstance(expected, datetime.datetime):
        assert row["v"].tzinfo is None


@needs_the_writer
def test_write_keeps_nan_and_infinity_floats(tmp_path):
    rows = [{"v": float("nan")}, {"v": float("inf")}, {"v": float("-inf")}]
    nan, inf, minus_inf = (
        row["v"] for row in _read_via_source(_write(tmp_path / "out.avro", rows))
    )

    assert math.isnan(nan)
    assert inf == float("inf") and minus_inf == float("-inf")


@needs_the_writer
def test_write_keeps_naive_and_aware_datetimes_in_one_column(tmp_path):
    rows = [{"v": WALL_TIME_IN_UTC}, {"v": AWARE}]

    assert _read_via_source(_write(tmp_path / "out.avro", rows)) == [
        {"v": WALL_TIME_IN_UTC},
        {"v": WALL_TIME_IN_UTC},
    ]


@needs_the_writer
def test_write_writes_a_timestamp_as_a_local_timestamp(tmp_path):
    """What another Avro reader is told: a wall-clock value, not an instant."""
    raw = _write(tmp_path / "out.avro", [{"v": AWARE}]).read_bytes()

    assert b'"logicalType":"local-timestamp-micros"' in raw
    assert b'"name":"root"' in raw


#: Columns Polars types as null, which it panics writing to Avro unless given a type.
NULL_TYPED = {
    "a column of only nulls": [{"id": 1, "v": None}, {"id": 2, "v": None}],
    "a column of only empty lists": [{"id": 1, "v": []}, {"id": 2, "v": []}],
    "a list of only nulls": [{"id": 1, "v": [None]}],
    "a struct field that is always null": [{"id": 1, "v": {"a": None, "b": 1}}],
    "nested empty lists": [{"id": 1, "v": [[], []]}],
}


@needs_the_writer
@pytest.mark.parametrize("rows", NULL_TYPED.values(), ids=NULL_TYPED)
def test_write_gives_a_null_typed_column_a_type(tmp_path, rows):
    assert _read_via_source(_write(tmp_path / "out.avro", rows)) == rows


@needs_the_writer
@pytest.mark.parametrize(
    "rows",
    [[], [{}, {"a": 1}]],
    ids=["no rows", "an empty row before a full one"],
)
def test_write_round_trips_an_empty_shape(tmp_path, rows):
    expected = [{"a": None}, {"a": 1}] if rows else []

    assert _read_via_source(_write(tmp_path / "out.avro", rows)) == expected


REFUSED = {
    "an integer past a long": ([{"v": 2**63}], "signed 64-bit"),
    "an unsigned 64-bit integer": ([{"v": 2**64 - 1}], "signed 64-bit"),
    "an integer past 64 bits": ([{"v": 2**64}], "signed 64-bit"),
    "an integer below a long": ([{"v": -(2**63) - 1}], "signed 64-bit"),
    "an integer in a list": ([{"v": [1, 2**63]}], "signed 64-bit"),
    "an integer in a struct": ([{"v": {"n": 2**70}}], "signed 64-bit"),
    "a duration": ([{"v": datetime.timedelta(seconds=5)}], "duration"),
    "a decimal NaN": ([{"v": decimal.Decimal("NaN")}], "decimal"),
    "a decimal infinity": ([{"v": [decimal.Decimal("-Infinity")]}], "decimal"),
    "a datetime outside UTC": (
        [
            {
                "v": datetime.datetime(
                    1, 1, 1, tzinfo=datetime.timezone(datetime.timedelta(hours=5))
                )
            }
        ],
        "outside the range",
    ),
}


@pytest.mark.parametrize(("rows", "reason"), REFUSED.values(), ids=REFUSED)
def test_write_refuses_by_column_before_opening_the_file(
    tmp_path, monkeypatch, rows, reason
):
    """Before Polars is involved, so on every leg: the version gate is patched open."""
    monkeypatch.setattr(pl, "__version__", "2.1.0")
    path = tmp_path / "out.avro"
    path.write_bytes(b"already here")

    with pytest.raises(ValueError, match="Column 'v'") as excinfo:
        _write(path, rows)

    assert reason in str(excinfo.value)
    assert path.read_bytes() == b"already here"


def test_an_integer_refusal_names_where_it_would_keep(tmp_path, monkeypatch):
    monkeypatch.setattr(pl, "__version__", "2.1.0")

    with pytest.raises(ValueError) as inside:
        _write(tmp_path / "a.avro", [{"v": 2**64 - 1}])
    with pytest.raises(ValueError) as past:
        _write(tmp_path / "b.avro", [{"v": 2**64}])

    assert "JSON" in str(inside.value) and "MessagePack" in str(inside.value)
    assert "CBOR or YAML" in str(past.value)


def test_write_refuses_rows_that_carry_no_column(tmp_path, monkeypatch):
    """The file could not keep their count: a frame with no columns has no height."""
    monkeypatch.setattr(pl, "__version__", "2.1.0")
    path = tmp_path / "out.avro"

    with pytest.raises(ValueError, match="at least one column"):
        _write(path, [{}, {}])

    assert not path.exists()


@needs_the_writer
def test_a_panic_inside_polars_is_an_ordinary_exception(tmp_path, monkeypatch):
    """`PanicException` derives from `BaseException`, which dlt's extract step does not
    catch. Whatever still panics is re-raised as an error, and nothing is written."""

    def panic(self, *args, **kwargs):
        raise pl.exceptions.PanicException("not yet implemented: something new")

    monkeypatch.setattr(pl.DataFrame, "write_avro", panic)
    path = tmp_path / "out.avro"
    path.write_bytes(b"already here")

    with pytest.raises(ValueError, match="aborted inside Polars: not yet implemented"):
        _write(path, [{"id": 1}])

    assert path.read_bytes() == b"already here"


@needs_the_writer
def test_a_polars_error_keeps_the_file_already_there(tmp_path, monkeypatch):
    def fail(self, *args, **kwargs):
        raise pl.exceptions.ComputeError("not yet implemented: write X to avro")

    monkeypatch.setattr(pl.DataFrame, "write_avro", fail)
    path = tmp_path / "out.avro"
    path.write_bytes(b"already here")

    with pytest.raises(pl.exceptions.ComputeError):
        _write(path, [{"id": 1}])

    assert path.read_bytes() == b"already here"


@needs_the_writer
def test_write_destination_round_trips_without_dlt_columns(tmp_path):
    """Through `post_load`, from a staged JSONL file, the way a pipeline reaches it."""
    destination = LocalFilesystemDestination()
    output_path = tmp_path / "out.avro"
    destination.dlt_dest(f"file://{output_path}")
    destination.dataset_name, destination.table_name = "public", "rows"
    table_dir = Path(destination.temp_path) / "public" / "rows"
    table_dir.mkdir(parents=True)
    rows = [
        {"id": 1, "tags": [], "_dlt_id": "internal"},
        {"id": 2, "tags": ["x"], "note": "later", "_dlt_load_id": "internal"},
    ]
    write_jsonl(str(table_dir / "load.jsonl"), rows)
    destination.post_load()

    assert _read_via_source(output_path) == [
        {"id": 1, "tags": [], "note": None},
        {"id": 2, "tags": ["x"], "note": "later"},
    ]
