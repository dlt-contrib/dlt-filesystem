"""Read a CSV whose first line is data, not a header.

The `csv_headless` format was covered only by routing assertions -- that a
`#csv_headless` selection resolves to `read_csv_headless` -- so the reader itself
had never run in a test. These read files, because the whole point of the format is
which values end up in which column.
"""

import duckdb

from omniload import run_ingest

ROWS = "Alice,30\nBob,41\n"


def load(tmp_path, **options):
    source = tmp_path / "people.dat"
    source.write_text(ROWS)
    destination = tmp_path / "warehouse.duckdb"

    run_ingest(
        source_uri=f"file://{source}#csv_headless",
        dest_uri=f"duckdb:///{destination}",
        source_table="",
        dest_table="out.people",
        progress="log",
        **options,
    )
    return destination


def rows(destination, statement):
    connection = duckdb.connect(str(destination))
    try:
        return connection.sql(statement).fetchall()
    finally:
        connection.close()


def test_named_columns_carry_the_first_line_as_data(tmp_path):
    """Names supplied with `--columns` name the columns; no row is consumed."""
    destination = load(tmp_path, columns=["name:text,age:bigint"])

    assert rows(destination, "select name, age from out.people order by name") == [
        ("Alice", 30),
        ("Bob", 41),
    ]


def test_unnamed_columns_get_generated_names(tmp_path):
    """Without names, every column still arrives, under a generated name."""
    destination = load(tmp_path)

    assert rows(
        destination,
        "select unknown_col_0, unknown_col_1 from out.people order by unknown_col_0",
    ) == [("Alice", 30), ("Bob", 41)]
