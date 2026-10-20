# Copyright 2022-2025 ScaleVector
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helpers for the filesystem resource."""

from typing import Any, Dict, Generator, Iterable, List, Optional, Sequence

from dlt.common.typing import TDataItem

from dlt_filesystem.source.format.settings import DEFAULT_CHUNK_SIZE


def add_columns(
    columns: List[str], rows: Iterable[Sequence[Any]]
) -> List[Dict[str, Any]]:
    """Adds column names to the given rows.

    Args:
        columns (List[str]): The column names.
        rows (Iterable[Sequence[Any]]): The rows.

    Returns:
        List[Dict[str, Any]]: The rows with column names.
    """
    result = []
    for row in rows:
        result.append(dict(zip(columns, row)))

    return result


def fetch_arrow(file_data, chunk_size: Optional[int] = None) -> Iterable[TDataItem]:
    """Fetches data from the given CSV file.

    Args:
        file_data (DuckDBPyRelation): The CSV file data.
        chunk_size (int): The number of rows to read at once.

    Yields:
        Iterable[TDataItem]: Data items, read from the given CSV file.
    """
    chunk_size = chunk_size or DEFAULT_CHUNK_SIZE
    batcher = file_data.fetch_arrow_reader(batch_size=chunk_size)
    yield from batcher


def fetch_json(
    file_data, chunk_size: Optional[int] = None
) -> Generator[List[Dict[str, Any]], None, None]:
    """Fetches data from the given CSV file.

    Args:
        file_data (DuckDBPyRelation): The CSV file data.
        chunk_size (int): The number of rows to read at once.

    Yields:
        Iterable[TDataItem]: Data items, read from the given CSV file.
    """
    chunk_size = chunk_size or DEFAULT_CHUNK_SIZE
    columns = file_data.columns
    fetchmany = getattr(file_data, "fetchmany", None)
    connection = None
    try:
        if fetchmany is None:
            # DuckDB 2.0 development builds removed fetchmany() from relations, but the
            # DB-API connection still exposes it. Use a duplicate cursor because executing
            # two results on the module connection would make them overwrite each other.
            import duckdb

            settings_query = "SELECT name, value FROM duckdb_settings()"
            source_settings = dict(duckdb.sql(settings_query).fetchall())
            if source_settings.get("lock_configuration") == "true":
                # A locked connection forbids replaying its local settings on a duplicate.
                # Page the original relation instead, which keeps those settings without
                # holding shared DB-API result state between yields.
                offset = 0
                while True:
                    batch = file_data.limit(chunk_size, offset).fetchall()
                    if not batch:
                        return
                    yield add_columns(columns, batch)
                    offset += len(batch)

            connection = duckdb.cursor()
            duplicate_settings = dict(connection.execute(settings_query).fetchall())
            for name, value in source_settings.items():
                if duplicate_settings.get(name) == value:
                    continue
                quoted_name = '"' + name.replace('"', '""') + '"'
                if value is None:
                    connection.execute(f"RESET {quoted_name}")
                else:
                    connection.execute(f"SET {quoted_name} = ?", [value])

            connection.execute(file_data.sql_query())
            fetchmany = connection.fetchmany

        while True:
            batch = fetchmany(chunk_size)
            if not batch:
                break

            yield add_columns(columns, batch)
    finally:
        if connection is not None:
            connection.close()
