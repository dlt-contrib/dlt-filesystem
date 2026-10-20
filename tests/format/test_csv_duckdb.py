from datetime import datetime, timezone
from decimal import Decimal
from io import BytesIO

import duckdb
import pytest
from fsspec.implementations.arrow import ArrowFSWrapper
from pyarrow import RecordBatch
from pyarrow.fs import LocalFileSystem

from dlt_filesystem.source.core import resource_for_reader
from dlt_filesystem.source.format.helpers import fetch_json
from dlt_filesystem.source.fsspec.local import LocalFilesystemSource
from dlt_filesystem.source.model import FilesystemReference

COMPLEX_VALUES_QUERY = """
select
    cast('12345678901234567890.123456789' as decimal(38, 9)) as decimal_value,
    timestamptz '2026-10-21 12:34:56.123456+13:00' as timestamp_value,
    null::varchar as null_value,
    [1, null, 3]::integer[] as list_value,
    {'name': 'Alice', 'score': cast('12.34' as decimal(10, 2))} as struct_value,
    map(['a', 'b'], [1, 2]) as map_value,
    cast('1267650600228229401496703205376' as hugeint) as hugeint_value,
    cast('340282366920938463463374607431768211455' as uhugeint) as uhugeint_value
"""


class RelationWithoutFetchMany:
    """Expose the relation API present on the affected DuckDB 2.0 build."""

    def __init__(self, relation):
        self.relation = relation

    @property
    def columns(self):
        return self.relation.columns

    def sql_query(self):
        return self.relation.sql_query()


def _legacy_fetch_json(relation, chunk_size):
    """Copy the pre-DuckDB-2 implementation as an independent parity oracle."""
    while True:
        batch = relation.fetchmany(chunk_size)
        if not batch:
            break
        yield [dict(zip(relation.columns, row)) for row in batch]


def _write_csv(tmp_path):
    path = tmp_path / "people.csv"
    path.write_text("id,name\n1,Alice\n2,Bob\n", encoding="utf-8")
    return path


def _read(path, fragment=""):
    return list(LocalFilesystemSource().dlt_source(f"file://{path}{fragment}", ""))


def test_resource_for_reader_builds_csv_duckdb_transformer(tmp_path):
    path = _write_csv(tmp_path)
    reference = FilesystemReference(
        fs=ArrowFSWrapper(LocalFileSystem()),
        bucket_url=str(tmp_path),
        file_glob=path.name,
        reader_name="read_csv_duckdb",
    )

    resource = resource_for_reader(reference)

    assert resource.name == "read_csv_duckdb"
    assert resource.table_name == "read_csv_duckdb"
    assert resource._parent.name == "filesystem"


def test_use_pyarrow_true_hint_reaches_csv_duckdb_reader(tmp_path):
    batches = _read(_write_csv(tmp_path), "#csv_duckdb&use_pyarrow=true")

    assert len(batches) == 1
    assert isinstance(batches[0], RecordBatch)
    assert batches[0].num_rows == 2


def test_use_pyarrow_false_hint_keeps_json_rows(tmp_path):
    rows = _read(_write_csv(tmp_path), "#csv_duckdb&use_pyarrow=false")

    assert rows == [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"},
    ]


def test_empty_use_pyarrow_hint_keeps_json_rows(tmp_path):
    rows = _read(_write_csv(tmp_path), "#csv_duckdb&use_pyarrow=")

    assert rows == [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"},
    ]


def test_csv_duckdb_matches_default_csv_row_count(tmp_path):
    path = _write_csv(tmp_path)

    assert len(_read(path, "#csv_duckdb")) == len(_read(path)) == 2


def test_fetch_json_fallback_preserves_python_values():
    batches = list(
        fetch_json(RelationWithoutFetchMany(duckdb.sql(COMPLEX_VALUES_QUERY)), 1)
    )

    assert batches == [
        [
            {
                "decimal_value": Decimal("12345678901234567890.123456789"),
                "timestamp_value": datetime(
                    2026, 10, 20, 23, 34, 56, 123456, tzinfo=timezone.utc
                ),
                "null_value": None,
                "list_value": [1, None, 3],
                "struct_value": {"name": "Alice", "score": Decimal("12.34")},
                "map_value": {"a": 1, "b": 2},
                "hugeint_value": 1267650600228229401496703205376,
                "uhugeint_value": 340282366920938463463374607431768211455,
            }
        ]
    ]
    row = batches[0][0]
    assert type(row["decimal_value"]) is Decimal
    assert type(row["timestamp_value"]) is datetime
    assert row["timestamp_value"].tzinfo is not None
    assert type(row["struct_value"]["score"]) is Decimal
    assert type(row["map_value"]) is dict
    assert type(row["hugeint_value"]) is int
    assert type(row["uhugeint_value"]) is int


def test_fetch_json_fallback_matches_legacy_relation_conversion():
    relation = duckdb.sql("select 1")
    if not hasattr(relation, "fetchmany"):
        pytest.skip("DuckDB removed the relation method used by the legacy oracle")

    expected = list(_legacy_fetch_json(duckdb.sql(COMPLEX_VALUES_QUERY), 1))
    actual = list(
        fetch_json(RelationWithoutFetchMany(duckdb.sql(COMPLEX_VALUES_QUERY)), 1)
    )

    assert actual == expected


def test_fetch_json_fallback_reexecutes_file_like_relation_on_default_connection():
    timezone_row = duckdb.execute("select current_setting('TimeZone')").fetchone()
    assert timezone_row is not None
    original_timezone = timezone_row[0]
    duckdb.execute("set TimeZone = ?", ["UTC"])
    try:
        relation = duckdb.from_csv_auto(
            BytesIO(b"timestamp\n2026-10-21 12:34:56\n"),
            columns={"timestamp": "TIMESTAMPTZ"},
            header=True,
        )

        batches = list(fetch_json(RelationWithoutFetchMany(relation), 1))

        assert batches == [
            [{"timestamp": datetime(2026, 10, 21, 12, 34, 56, tzinfo=timezone.utc)}]
        ]
        assert type(batches[0][0]["timestamp"]) is datetime
        assert batches[0][0]["timestamp"].tzinfo is not None
    finally:
        duckdb.execute("set TimeZone = ?", [original_timezone])

    restored_timezone_row = duckdb.execute(
        "select current_setting('TimeZone')"
    ).fetchone()
    assert restored_timezone_row is not None
    assert restored_timezone_row[0] == original_timezone


@pytest.mark.parametrize(
    ("row_count", "expected_batch_lengths"),
    [
        pytest.param(4, [2, 2], id="exact-multiple"),
        pytest.param(5, [2, 2, 1], id="partial-final-batch"),
        pytest.param(0, [], id="empty"),
    ],
)
def test_fetch_json_fallback_preserves_batch_boundaries(
    row_count, expected_batch_lengths
):
    batches = list(
        fetch_json(
            RelationWithoutFetchMany(
                duckdb.sql(f"select range as id from range({row_count})")
            ),
            2,
        )
    )

    assert [len(batch) for batch in batches] == expected_batch_lengths
    assert [row["id"] for batch in batches for row in batch] == list(range(row_count))
