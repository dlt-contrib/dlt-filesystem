"""Read back the loader files dlt wrote, in whatever format it wrote them.

The format is taken from the file name, which dlt always supplies: it writes
``{base}.{file_extension}[.gz]`` and appends ``.gz`` exactly when compression is on
(``dlt.common.data_writers.buffered``). Reading it back is the mirror of that, down to
choosing between ``gzip.open`` and ``open`` on the same fact the writer used.

Sniffing the bytes instead does not answer the question, because compression hides the
format: dlt gzips CSV as well as JSONL by default, so both arrive as the same magic
number. The name is the one place the format survives compression.
"""

import csv
import gzip
import os
from contextlib import contextmanager
from typing import Generator

from dlt.common import json
from pyarrow.parquet import ParquetFile

PARQUET_BATCH_SIZE = 64


class UnsupportedLoaderFileFormat(Exception):
    pass


def load_dlt_file(filepath: str) -> Generator:
    """
    load_dlt_file reads dlt loader files. It handles different loader file formats
    automatically. It returns a generator that yield data items as a python dict
    """
    with factory(filepath) as reader:
        yield from reader


def factory(filepath: str):
    """Return a reader for ``filepath``, chosen by the extension dlt gave it.

    ``insert_values`` is a supported dlt loader file format that this reader has no
    row-shaped reading for, so it lands here as an unsupported format rather than being
    misread as something else.
    """
    name = os.path.basename(filepath)

    compressed = name.endswith(".gz")
    if compressed:
        name = name[: -len(".gz")]

    _, dot, file_format = name.rpartition(".")
    if not dot:
        raise UnsupportedLoaderFileFormat(
            f"{os.path.basename(filepath)}: no format extension, "
            f"so this is not a file dlt wrote"
        )

    if file_format == "jsonl":
        return jsonlfile(filepath, compressed)
    elif file_format == "csv":
        return csvfile(filepath, compressed)
    elif file_format == "parquet":
        return parquetfile(filepath, compressed)
    else:
        raise UnsupportedLoaderFileFormat(file_format)


@contextmanager
def jsonlfile(filepath: str, compressed: bool = False):
    def reader(fd):
        for line in fd:
            yield json.loads(line.decode().strip())

    with (gzip.open if compressed else open)(filepath, "rb") as fd:
        yield reader(fd)


@contextmanager
def csvfile(filepath: str, compressed: bool = False):
    # Read the dialect dlt wrote with rather than assuming the default one. A configured
    # `data_writer.delimiter` otherwise parses every row into a single composite column,
    # which reads back as a successful load of unusable data.
    from dlt.common.configuration import resolve_configuration
    from dlt.common.destination.configuration import CsvFormatConfiguration

    # The section matters. These files are written in the normalize stage, so a
    # `[normalize.data_writer]` setting applies to them; resolving without the section
    # sees only the unscoped `[data_writer]` spelling and silently reads the default
    # dialect against a file written with another one.
    csv_format = resolve_configuration(
        CsvFormatConfiguration(), sections=("normalize",)
    )
    # csv ends a record on its own newline handling, which is the three spellings of a
    # line ending and nothing else; `lineterminator` governs writing, not reading, so
    # there is nothing to pass it. Splitting the text on any other terminator would have
    # to know where the quoted fields are to be correct, and a value containing the
    # terminator would be truncated without a word. Refusing says what happened instead.
    if csv_format.lineterminator not in ("\n", "\r\n", "\r"):
        raise UnsupportedLoaderFileFormat(
            f"csv written with the line terminator {csv_format.lineterminator!r}: "
            f"only the line endings csv itself ends a record on can be read back"
        )
    if not csv_format.include_header:
        raise UnsupportedLoaderFileFormat(
            "csv written without a header: the column names are in the dlt schema "
            "rather than the file, so the rows cannot be named from it alone"
        )

    # newline="" is what the csv module needs to handle quoted fields spanning lines.
    with (gzip.open if compressed else open)(
        filepath, "rt", newline="", encoding=csv_format.encoding
    ) as fd:
        yield csv.DictReader(fd, delimiter=csv_format.delimiter)


@contextmanager
def parquetfile(filepath: str, compressed: bool = False):
    def reader(pf: ParquetFile):
        for batch in pf.iter_batches(PARQUET_BATCH_SIZE):
            yield from batch.to_pylist()

    with (gzip.open if compressed else open)(filepath, "rb") as fd:
        yield reader(ParquetFile(fd))
