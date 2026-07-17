"""``--incremental-strategy scd2`` on the sources where SCD2 means something.

SCD2 is the one ``IncrementalStrategy`` value that is not a dlt write disposition but a
merge *strategy*, so it needs unflattening onto dlt's two axes before the run. The only
existing coverage was the filesystem reject path (``filesystem/test_incremental_strategy``),
which is why a bare ``write_disposition="scd2"`` reached dlt unnoticed; these tests cover
the sources it was never exercised on.

Mock-only unit lane (no Docker, no credentials): a real ``sqlite://`` source loads into a
real embedded duckdb, and behaviour is proven from the validity columns dlt materializes.
The source that manages its own incrementality is a small fake, mirroring the SaaS shape.
"""

import sqlite3

import dlt
import duckdb
import pyarrow as pa
import pyarrow.ipc as ipc
import pytest

from omniload import ValidationError, run_ingest
from omniload.core.factory import SourceDestinationFactory


def _make_sqlite(path, rows):
    """(Re)create a `people` table holding `rows`, standing in for a changing source."""
    conn = sqlite3.connect(str(path))
    conn.execute("DROP TABLE IF EXISTS people")
    conn.execute("CREATE TABLE people (id INTEGER, name TEXT, city TEXT)")
    conn.executemany("INSERT INTO people VALUES (?, ?, ?)", rows)
    conn.commit()
    conn.close()


def _load(src, dest, **kwargs):
    run_ingest(
        source_uri=f"sqlite:///{src}",
        dest_uri=f"duckdb:///{dest}",
        source_table="main.people",
        dest_table="out.people",
        incremental_strategy="scd2",
        progress="log",
        **kwargs,
    )


def _records(dest):
    """Every row as (name, city, is_active), oldest first."""
    con = duckdb.connect(str(dest))
    try:
        return con.sql(
            "select name, city, _dlt_valid_to is null from out.people "
            "order by name, _dlt_valid_from"
        ).fetchall()
    finally:
        con.close()


ALICE_BOB = [(1, "Alice", "Auckland"), (2, "Bob", "Wellington")]
BOB_MOVED = [(1, "Alice", "Auckland"), (2, "Bob", "Christchurch")]


def test_scd2_tracks_a_changed_record(tmp_path):
    """The point of SCD2: a changed row is retired and its successor opened, so the
    destination keeps the history rather than overwriting it."""
    src, dest = tmp_path / "src.db", tmp_path / "wh.duckdb"

    _make_sqlite(src, ALICE_BOB)
    _load(src, dest, primary_key=["id"])
    assert _records(dest) == [
        ("Alice", "Auckland", True),
        ("Bob", "Wellington", True),
    ]

    _make_sqlite(src, BOB_MOVED)
    _load(src, dest, primary_key=["id"])
    assert _records(dest) == [
        ("Alice", "Auckland", True),  # untouched: still the active record
        ("Bob", "Wellington", False),  # retired, but retained
        ("Bob", "Christchurch", True),  # the new active record
    ]


def test_scd2_leaves_unchanged_records_alone(tmp_path):
    """A re-run over identical data must be a no-op, not a retire-and-reinsert of every
    row. dlt decides this from a hash of the row it stores in `_dlt_id`, so this fails
    the moment scd2 runs on data whose `_dlt_id` is a fresh random value per load."""
    src, dest = tmp_path / "src.db", tmp_path / "wh.duckdb"
    _make_sqlite(src, ALICE_BOB)

    for _ in range(2):
        _load(src, dest, primary_key=["id"])

    assert _records(dest) == [
        ("Alice", "Auckland", True),
        ("Bob", "Wellington", True),
    ]


def test_scd2_tracks_changes_without_a_primary_key(tmp_path):
    """A primary key is not required: dlt hashes the whole row to spot a change, so scd2
    is just as meaningful without one."""
    src, dest = tmp_path / "src.db", tmp_path / "wh.duckdb"

    _make_sqlite(src, ALICE_BOB)
    _load(src, dest)
    _make_sqlite(src, BOB_MOVED)
    _load(src, dest)

    assert _records(dest) == [
        ("Alice", "Auckland", True),
        ("Bob", "Wellington", False),
        ("Bob", "Christchurch", True),
    ]


@pytest.mark.parametrize(
    ("option", "kwargs"),
    [
        ("--incremental-key", {"incremental_key": "city"}),
        ("--sql-limit", {"sql_limit": 1}),
        ("--yield-limit", {"yield_limit": 1}),
    ],
)
def test_scd2_rejects_a_partial_read(tmp_path, option, kwargs):
    """scd2 reads absence as deletion, so a run that reads only part of the table would
    retire every record it leaves out while the source still holds it, unchanged. Refuse
    the combination rather than quietly rewrite history."""
    src, dest = tmp_path / "src.db", tmp_path / "wh.duckdb"
    _make_sqlite(src, ALICE_BOB)

    with pytest.raises(ValidationError, match=f"cannot be combined with '{option}'"):
        _load(src, dest, **kwargs)


@pytest.mark.parametrize("algorithm", ["date_shift", "noise", "random", "uuid"])
def test_scd2_rejects_a_non_deterministic_mask(tmp_path, algorithm):
    """A mask that draws a fresh value per run rewrites the rows scd2 compares, so every
    untouched record would be retired and re-inserted on every load."""
    src, dest = tmp_path / "src.db", tmp_path / "wh.duckdb"
    _make_sqlite(src, ALICE_BOB)

    with pytest.raises(ValidationError, match="draws a fresh value on every run"):
        _load(src, dest, mask=[f"city:{algorithm}"])


def test_scd2_allows_a_deterministic_mask(tmp_path):
    """A deterministic mask holds its value across runs, so an unchanged row still reads as
    unchanged and the history stays put."""
    src, dest = tmp_path / "src.db", tmp_path / "wh.duckdb"
    _make_sqlite(src, ALICE_BOB)

    for _ in range(2):
        _load(src, dest, mask=["city:sha256"])

    con = duckdb.connect(str(dest))
    try:
        assert con.sql("select count(*) from out.people").fetchall()[0][0] == 2
    finally:
        con.close()


@pytest.mark.parametrize("backend", ["pyarrow", "connectorx"])
def test_scd2_rejects_the_arrow_backends(tmp_path, backend):
    """The Arrow-yielding backends can't carry scd2's row hash, so an explicit one is
    refused up front rather than failing deep inside dlt's normalize step."""
    src, dest = tmp_path / "src.db", tmp_path / "wh.duckdb"
    _make_sqlite(src, ALICE_BOB)

    with pytest.raises(ValidationError, match="cannot use the '" + backend + "' SQL"):
        _load(src, dest, primary_key=["id"], sql_backend=backend)


def test_scd2_rejects_the_arrow_backends_on_a_dry_run(tmp_path):
    """The refusal is a property of the request, so `--dry-run` reports it rather than
    passing a run that could never load."""
    src, dest = tmp_path / "src.db", tmp_path / "wh.duckdb"
    _make_sqlite(src, ALICE_BOB)

    with pytest.raises(ValidationError, match="cannot use the 'pyarrow' SQL"):
        _load(src, dest, primary_key=["id"], sql_backend="pyarrow", dry_run=True)


def test_scd2_ignores_the_backend_for_a_source_that_has_none(tmp_path, monkeypatch):
    """The Arrow-backend rule is about how a SQL source is read, so it must not fire for a
    source that yields dicts and never consults `--sql-backend`."""
    dest = tmp_path / "wh.duckdb"

    class _DictSource:
        def handles_incrementality(self) -> bool:
            return False

        def dlt_source(self, uri, table, **kwargs):
            @dlt.resource(name="rows")
            def rows():
                yield from [{"id": 1, "city": "Auckland"}]

            return rows()

    monkeypatch.setattr(
        SourceDestinationFactory, "get_source", lambda self: _DictSource()
    )
    run_ingest(
        source_uri="mongodb://placeholder/db",
        dest_uri=f"duckdb:///{dest}",
        source_table="rows",
        dest_table="out.rows",
        incremental_strategy="scd2",
        sql_backend="pyarrow",
        progress="log",
    )

    con = duckdb.connect(str(dest))
    try:
        columns = [c[0] for c in con.sql("describe out.rows").fetchall()]
        assert "_dlt_valid_from" in columns
    finally:
        con.close()


def test_scd2_allows_an_arrow_backend_for_a_custom_query(tmp_path):
    """A `query:` table is read with sqlalchemy whichever backend is named, so scd2 has no
    reason to refuse one. A dry run pins the check on the request without loading, because
    reading a custom query swaps a function out of dlt for the rest of the process."""
    src, dest = tmp_path / "src.db", tmp_path / "wh.duckdb"
    _make_sqlite(src, ALICE_BOB)

    # Raises if the Arrow-backend rule wrongly fires for a custom query.
    run_ingest(
        source_uri=f"sqlite:///{src}",
        dest_uri=f"duckdb:///{dest}",
        source_table="query:SELECT id, name, city FROM people",
        dest_table="out.people",
        incremental_strategy="scd2",
        sql_backend="pyarrow",
        dry_run=True,
        progress="log",
    )


def test_scd2_rejects_an_arrow_source(tmp_path):
    """`mmap://` yields Arrow whatever the backend says, so scd2 is refused for it outright
    rather than reaching the same unpopulated-`_dlt_id` failure inside dlt."""
    src = tmp_path / "people.arrow"
    table = pa.table({"id": [1, 2], "city": ["Auckland", "Wellington"]})
    with pa.OSFile(str(src), "wb") as sink:
        with ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)

    with pytest.raises(ValidationError, match="cannot read from 'mmap://'"):
        run_ingest(
            source_uri=f"mmap://{src}",
            dest_uri=f"duckdb:///{tmp_path / 'wh.duckdb'}",
            source_table="arrow_mmap",
            dest_table="out.people",
            incremental_strategy="scd2",
            progress="log",
        )


def test_scd2_runs_on_the_default_backend(tmp_path):
    """scd2 works without naming a backend: the default resolves to pyarrow, which can't
    carry the row hash, so scd2 selects sqlalchemy for itself."""
    src, dest = tmp_path / "src.db", tmp_path / "wh.duckdb"
    _make_sqlite(src, ALICE_BOB)

    _load(src, dest, primary_key=["id"], sql_backend="default")

    assert _records(dest) == [
        ("Alice", "Auckland", True),
        ("Bob", "Wellington", True),
    ]


class _ManagedSource:
    """A ``handles_incrementality`` source that sets its own resource-level disposition,
    like the SaaS/streaming sources. It omits ``honours_run_disposition``, so a run-level
    disposition must not reach it (getattr defaults to False)."""

    def handles_incrementality(self) -> bool:
        return True

    def dlt_source(self, uri, table, **kwargs):
        @dlt.resource(name="rows")
        def rows():
            yield from [{"id": 1}, {"id": 2}, {"id": 3}]

        return rows()


def test_managed_source_still_ignores_scd2(tmp_path, monkeypatch):
    """A source that owns its disposition keeps ignoring scd2 rather than being switched
    to it: the resource-level append stands, so rows accumulate across runs."""
    dest = tmp_path / "wh.duckdb"
    monkeypatch.setattr(
        SourceDestinationFactory, "get_source", lambda self: _ManagedSource()
    )
    for _ in range(2):
        run_ingest(
            source_uri="file://placeholder.csv",
            dest_uri=f"duckdb:///{dest}",
            source_table="rows",
            dest_table="out.rows",
            incremental_strategy="scd2",
            primary_key=["id"],
            progress="log",
        )

    con = duckdb.connect(str(dest))
    try:
        assert con.sql("select count(*) from out.rows").fetchall()[0][0] == 6
        # No SCD2 machinery on a table that was never loaded with it.
        columns = [c[0] for c in con.sql("describe out.rows").fetchall()]
        assert "_dlt_valid_from" not in columns
    finally:
        con.close()
