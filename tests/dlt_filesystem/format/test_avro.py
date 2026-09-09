"""Apache Avro object container reads.

Avro is a **read** format here, and the last section of this file is why: Polars can
write Avro, but its writer mis-frames a list column that holds an empty list before a
non-empty one, and the resulting file is malformed on disk. Those cases are `xfail`ed
rather than described in a comment, so the day Polars fixes them the suite says so.

The fixtures are hand-built (`testing.writer.write_avro`) rather than produced by
`polars.DataFrame.write_avro`. A fixture written by the library under test can only
contain what that library can *write*, and the reader's sharpest failure is on input it
cannot: an Avro `map` field is an ordinary type that `polars.read_avro` panics on.
"""

import datetime
import decimal
import gzip
import json
import re
import struct
from pathlib import Path

import polars as pl
import pytest
from dlt.extract.exceptions import ResourceExtractionError

from dlt_filesystem.source.format.readers import read_avro
from dlt_filesystem.source.fsspec.local import LocalFilesystemSource
from dlt_filesystem.target.util import _resolve_output_target
from dlt_filesystem.testing.stub import FileItemStub
from dlt_filesystem.testing.writer import (
    encode_avro_bytes,
    encode_avro_long,
    write_avro,
    write_avro_container,
)


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
    assert excinfo.match("aborted inside Polars")
    assert excinfo.match("far.avro")


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
    assert isinstance(excinfo.value, ValueError)


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
    assert excinfo.match("OutOfSpec")


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
    assert excinfo.match("OutOfSpec")


def test_a_container_with_no_records_yields_no_rows(tmp_path):
    """A record schema with no fields and no data blocks is a valid, empty container."""
    path = write_avro_container(
        tmp_path / "none.avro", {"type": "record", "name": "root", "fields": []}
    )
    assert _read_via_source(path) == []


# --- why Avro is a read format ---


def _decode_avro_records(raw: bytes) -> list[list]:
    """Decode an object container by hand, with no Polars in the loop.

    Deliberately minimal and deliberately independent: the point of the `xfail`s below
    is that the *file* is malformed, and asking Polars' own reader whether Polars' own
    writer produced a good file cannot establish that.
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


_LIST_SCHEMA = {"id": pl.Int64, "lst": pl.List(pl.String)}


def test_polars_writes_a_well_formed_list_column(tmp_path):
    """The control the two `xfail`s below are meaningful against.

    An empty list in the *last* row writes correctly, so the defect is about position
    rather than about empty lists or about list columns as such.
    """
    path = tmp_path / "ok.avro"
    pl.DataFrame({"id": [1, 2], "lst": [["a"], []]}, schema=_LIST_SCHEMA).write_avro(
        str(path)
    )

    assert _decode_avro_records(path.read_bytes()) == [[1, ["a"]], [2, []]]


@pytest.mark.xfail(
    strict=True,
    reason=(
        "polars.DataFrame.write_avro mis-frames a list column holding an empty list "
        "before a non-empty one: the list items are written without their block "
        "headers, so the record body overruns the block. Measured identically on "
        "polars 1.34.0, 1.39.3, 1.43.1 and 2.0.0rc1. Remove this mark, and register "
        "an Avro writer, once it is fixed upstream."
    ),
)
def test_polars_writes_a_well_formed_list_column_with_an_empty_list_first(tmp_path):
    """The reason `avro` has no writer registration.

    `{"tags": []}` followed by `{"tags": ["x"]}` is ordinary JSON and reaches a writer
    as this frame on the default load path, so registering the Polars writer would mean
    exporting a corrupt file for it.
    """
    path = tmp_path / "bad.avro"
    pl.DataFrame({"id": [1, 2], "lst": [[], ["a"]]}, schema=_LIST_SCHEMA).write_avro(
        str(path)
    )

    assert _decode_avro_records(path.read_bytes()) == [[1, []], [2, ["a"]]]


@pytest.mark.xfail(
    strict=True,
    reason="same polars defect; this shape reads back without error and with wrong values",
)
def test_polars_round_trips_a_float_beside_such_a_list_column(tmp_path):
    """The same defect at its worst: no error on either side, and wrong data.

    With an `Int64` sibling the malformed block is caught as a short read. With a
    `Float64` one the misaligned bytes decode as a valid float, so the export succeeds,
    reads back, and is silently wrong -- `2.0` returns as `7.7e-319`.
    """
    frame = pl.DataFrame(
        {"f": [1.0, 2.0], "lst": [[], ["a"]]},
        schema={"f": pl.Float64, "lst": pl.List(pl.String)},
    )
    path = tmp_path / "silent.avro"
    frame.write_avro(str(path))

    assert pl.read_avro(str(path)).equals(frame)


def test_an_avro_destination_is_refused_with_the_writable_formats(tmp_path):
    """Reading a format the destination cannot write is the shape eight formats here
    already have, so `--dest-uri file://out.avro` names what it can write instead."""
    with pytest.raises(ValueError) as excinfo:
        _resolve_output_target("file:///data/out.avro")
    assert excinfo.match("only supports file formats")
    assert excinfo.match(re.escape("(got 'avro')"))
    assert "avro" not in str(excinfo.value).split("formats:")[1].split("(got")[0]
