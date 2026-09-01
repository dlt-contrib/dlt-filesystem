"""``--reshape`` over a SQL source, which is where the item shape bites.

SQL defaults to the ``pyarrow`` backend (``api.py`` promotes ``SqlBackend.default``),
so the extract pipe carries ``pyarrow.Table`` items rather than dicts. dlt's
``MapItem`` enumerates a Python list and passes anything else through whole, so a
per-row reshape wired with ``add_map`` would receive one Table where its contract
says one document, and a jq reshape would then fail inside ``_to_jsonish``.

Mock-only unit lane (no Docker, no credentials): a real ``sqlite://`` source loads
into a real embedded duckdb, and the assertion is on the rows that land.
"""

import sqlite3

import duckdb
import pytest

from omniload import ValidationError, run_ingest

RESHAPE_CALLS: list[str] = []


def rename_and_widen(doc: dict) -> dict:
    """A per-row reshape, recording the type it was actually handed."""
    RESHAPE_CALLS.append(type(doc).__name__)
    return {
        "id": doc["id"],
        "full_name": f"{doc['first']} {doc['last']}",
    }


@pytest.fixture
def people_db(tmp_path):
    path = tmp_path / "src.db"
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE people (id INTEGER, first TEXT, last TEXT)")
    conn.executemany(
        "INSERT INTO people VALUES (?, ?, ?)",
        [(1, "Ada", "Lovelace"), (2, "Grace", "Hopper")],
    )
    conn.commit()
    conn.close()
    return path


def test_per_row_reshape_receives_documents_from_an_arrow_backed_source(
    people_db, tmp_path
):
    RESHAPE_CALLS.clear()
    dest = tmp_path / "warehouse.duckdb"

    run_ingest(
        source_uri=f"sqlite:///{people_db}",
        dest_uri=f"duckdb:///{dest}",
        source_table="main.people",
        dest_table="out.people",
        reshape=f"python:{__name__}:rename_and_widen",
        progress="log",
    )

    # One call per row, each a document. A Table would show up as a single
    # "Table" entry, which is the regression this pins.
    assert RESHAPE_CALLS == ["dict", "dict"]

    con = duckdb.connect(str(dest))
    try:
        rows = con.sql("select id, full_name from out.people order by id").fetchall()
    finally:
        con.close()
    assert rows == [(1, "Ada Lovelace"), (2, "Grace Hopper")]


def test_dry_run_rejects_an_unusable_reshape_spec(people_db, tmp_path):
    """`--dry-run` has to resolve the reshape, or it validates everything but.

    The reshape used to be built after the dry-run return, so a bad spec, a
    missing extra or a batch engine over a source that cannot feed it all
    reported success and failed only on the real run.
    """
    with pytest.raises(ValidationError):
        run_ingest(
            source_uri=f"sqlite:///{people_db}",
            dest_uri=f"duckdb:///{tmp_path / 'warehouse.duckdb'}",
            source_table="main.people",
            dest_table="out.people",
            reshape="nope:anything",
            dry_run=True,
            progress="log",
        )


def test_dry_run_rejects_a_batch_engine_the_source_cannot_feed(people_db, tmp_path):
    """The Arrow requirement is a spec-level fact, so a dry run knows it."""
    pytest.importorskip("macropipe")
    pytest.importorskip("polars")

    with pytest.raises(ValidationError, match="Arrow-yielding"):
        run_ingest(
            source_uri=f"sqlite:///{people_db}",
            dest_uri=f"duckdb:///{tmp_path / 'warehouse.duckdb'}",
            source_table="main.people",
            dest_table="out.people",
            reshape="polars:select:id",
            dry_run=True,
            progress="log",
        )
