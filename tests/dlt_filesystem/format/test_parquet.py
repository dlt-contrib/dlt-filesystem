"""Parquet write.

The read side is covered where the reader is shared: `test_feather.py` asserts that the
Feather and Parquet readers narrow a nanosecond column identically on the row path, which
is a claim about the pipeline rather than about either format. What this file holds is
what `write_parquet` puts on disk, and the one place Polars and Parquet disagree about
what an integer is.
"""

import datetime
import decimal

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dlt_filesystem.target.registry import writer_for_format


def _written(tmp_path, rows, name="out.parquet"):
    """Write rows and read the file back with PyArrow, independent of `src/`."""
    path = tmp_path / name
    writer_for_format("parquet")(str(path), rows)
    table = pq.read_table(str(path))
    return {field.name: field.type for field in table.schema}, table.to_pylist()


def test_write_narrows_an_unsigned_integer_that_fits_a_u64(tmp_path):
    """The file a Parquet reader other than Polars can open.

    Polars widens an integer past the `Int64` range to `Int128`, which Parquet has no
    type for and which lands as an untyped 16-byte `FIXED_LEN_BYTE_ARRAY`: PyArrow
    refuses it outright and DuckDB returns the column as a `BLOB`. `UInt64` holds the
    same value and is a type every reader has, so the export stays spendable.
    """
    types, rows = _written(tmp_path, [{"n": 2**64 - 1}])

    assert types == {"n": pa.uint64()}
    assert rows == [{"n": 2**64 - 1}], "the value itself, not a rounded one"


def test_narrowing_reaches_a_struct_field(tmp_path):
    """dlt keeps a nested column as one JSON column, so this is how the value arrives.

    A struct field is typed exactly as a column is, so a walk that stopped at the top
    level would leave the file unreadable for exactly the loads that carry structured
    data.
    """
    types, rows = _written(tmp_path, [{"s": {"n": 2**64 - 1}}])

    assert types["s"] == pa.struct([("n", pa.uint64())])
    assert rows == [{"s": {"n": 2**64 - 1}}]


def test_narrowing_reaches_a_struct_inside_a_list(tmp_path):
    """A repeated JSON object, which is the other shape dlt keeps as one column.

    Worth its own case because it is the only list shape that narrows rather than
    refuses: `from_dicts` infers `List(Struct({"id": Int128}))` here, where a bare
    list of the same integers would already be `List(UInt64)`. Without the list arm
    of the walk this column stays 128-bit and `pyarrow` cannot open the file.
    """
    types, rows = _written(tmp_path, [{"items": [{"id": 2**64 - 1}]}])

    assert types["items"].value_type == pa.struct([("id", pa.uint64())])
    assert rows == [{"items": [{"id": 2**64 - 1}]}]


def test_a_list_that_cannot_narrow_is_refused_rather_than_written(tmp_path):
    """A bare list of integers reaches the walk only when it cannot narrow.

    `from_dicts` infers `List(UInt64)` for one whose elements all fit, so this shape
    arrives 128-bit only when it holds a value the cast will refuse anyway, unlike
    the list of structs above. Walking it is what turns a file no reader can open
    into an error naming the value: left unwalked, this row writes, and `pyarrow`
    then refuses the result with the same error it gives the scalar form.
    """
    path = tmp_path / "out.parquet"

    with pytest.raises(pl.exceptions.InvalidOperationError, match="-1"):
        writer_for_format("parquet")(str(path), [{"l": [2**64 - 1, -1]}])

    assert not path.exists()


def test_ordinary_integers_keep_their_signed_type(tmp_path):
    """The narrowing is scoped to the columns Parquet cannot hold.

    A signed column that fits is left signed rather than swept up by a blanket cast to
    `UInt64`, which would refuse every negative number in the project.
    """
    types, rows = _written(tmp_path, [{"n": -1}, {"n": 2**63 - 1}])

    assert types == {"n": pa.int64()}
    assert rows == [{"n": -1}, {"n": 2**63 - 1}]


def test_write_refuses_an_integer_past_the_unsigned_64_range(tmp_path):
    """Refused by column and by value, and no file is left behind.

    Past `2**64-1` the three writers agree, which they do not on the band below it:
    `write_orc` and `write_feather` refuse everything past a signed 64-bit, PyArrow's
    inference from a Python int stopping there, while this one now writes that band as
    `uint64`. The strict cast runs before the path is opened, so a refused write leaves
    nothing behind, which is true of the other two as well: both raise while building
    their table.
    """
    path = tmp_path / "out.parquet"

    with pytest.raises(pl.exceptions.InvalidOperationError) as excinfo:
        writer_for_format("parquet")(str(path), [{"n": 2**64}])

    assert "'n'" in str(excinfo.value), "names the column"
    assert str(2**64) in str(excinfo.value), "names the value"
    assert not path.exists()


def test_write_refuses_a_negative_in_a_column_another_row_widened(tmp_path):
    """A column is one type, so one wide value decides what the rest must fit.

    Worth its own case because the failure is not a property of the offending row: `-1`
    writes fine on its own, and only becomes unwritable next to a value that pushed the
    column past `Int64`.
    """
    with pytest.raises(pl.exceptions.InvalidOperationError, match="-1"):
        writer_for_format("parquet")(
            str(tmp_path / "out.parquet"), [{"n": 2**64 - 1}, {"n": -1}]
        )


def test_written_extended_types(tmp_path):
    """What the writer puts on disk for the types the row path carries.

    Measured rather than asserted from the docs, and pinned because a Polars change would
    otherwise move these silently. `time64[ns]` is the one that reads as a surprise: a row
    carries a `datetime.time`, whose precision is microseconds, and it comes back out as
    nanoseconds because Polars' `Time` is nanosecond-backed. Timestamps and durations
    narrow to microseconds instead.
    """
    types, rows = _written(
        tmp_path,
        [
            {
                "t": datetime.time(0, 0, 0, 123456),
                "ts": datetime.datetime(2020, 1, 2, 3, 4, 5, 123456),
                "dur": datetime.timedelta(seconds=1, microseconds=234567),
                "dec": decimal.Decimal("3.14"),
                "blob": b"hi",
            }
        ],
    )

    assert types == {
        "t": pa.time64("ns"),
        "ts": pa.timestamp("us"),
        "dur": pa.duration("us"),
        "dec": pa.decimal128(38, 2),
        "blob": pa.large_binary(),
    }
    assert rows[0]["t"] == datetime.time(0, 0, 0, 123456)
    assert rows[0]["dec"] == decimal.Decimal("3.14")
