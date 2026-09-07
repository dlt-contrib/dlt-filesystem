"""Read-back tests for the dlt loader files ``dlt_filesystem.util.loader`` consumes.

The fixtures are written by dlt itself rather than by hand, because the thing under test
is agreement with dlt's own naming: which extension it gives each loader file format, and
when it appends ``.gz``. Hand-rolled fixtures would agree with whatever this module
believes and could not catch a change in dlt.
"""

import os
import pathlib
import tempfile

import pytest
from dlt.common.data_writers.writers import TLoaderFileFormat

from dlt_filesystem.util.loader import (
    UnsupportedLoaderFileFormat,
    factory,
    load_dlt_file,
)

TESTDATA = [
    {"name": "Jhon", "email": "jhon@acme.com"},
    {"name": "Lisa", "email": "lisa@acme.com"},
]

#: label -> (loader_file_format, compression disabled, expected file name suffix).
#: Both compressed cases matter: dlt gzips CSV as well as JSONL by default, so the
#: compressed names are the pair a content sniffer cannot tell apart.
LOADER_FILES: dict[str, tuple[TLoaderFileFormat, bool, str]] = {
    "jsonl": ("jsonl", False, ".jsonl.gz"),
    "csv": ("csv", False, ".csv.gz"),
    "parquet": ("parquet", False, ".parquet"),
    "jsonl-uncompressed": ("jsonl", True, ".jsonl"),
    "csv-uncompressed": ("csv", True, ".csv"),
}


def _write_with_dlt(
    loader_file_format: TLoaderFileFormat, disable_compression: bool
) -> str:
    """Run one dlt load into a temp bucket and return its single data file."""
    import dlt

    if disable_compression:
        os.environ["DATA_WRITER__DISABLE_COMPRESSION"] = "true"
    else:
        os.environ.pop("DATA_WRITER__DISABLE_COMPRESSION", None)

    bucket = tempfile.mkdtemp()
    pipeline = dlt.pipeline(
        pipeline_name=f"loader_fixture_{loader_file_format}_{disable_compression}",
        destination=dlt.destinations.filesystem(
            bucket_url=pathlib.Path(bucket).as_uri()
        ),
        dataset_name="public",
        pipelines_dir=tempfile.mkdtemp(),
    )
    pipeline.run(TESTDATA, table_name="people", loader_file_format=loader_file_format)

    table_dir = os.path.join(bucket, "public", "people")
    names = sorted(os.listdir(table_dir))
    assert len(names) == 1, f"expected one data file, got {names}"
    return os.path.join(table_dir, names[0])


@pytest.fixture(scope="session")
def loader_files():
    """One dlt-written data file per format, keyed by the labels in ``LOADER_FILES``."""
    files = {}
    try:
        for label, (fmt, disable_compression, _) in LOADER_FILES.items():
            files[label] = _write_with_dlt(fmt, disable_compression)
        yield files
    finally:
        os.environ.pop("DATA_WRITER__DISABLE_COMPRESSION", None)


def _payload(rows):
    """Rows without dlt's bookkeeping columns, which the caller strips itself."""
    return [{k: v for k, v in row.items() if not k.startswith("_dlt_")} for row in rows]


@pytest.mark.parametrize("label", list(LOADER_FILES))
def test_loader_reads_every_format_dlt_writes(loader_files, label):
    """Both compressed formats included, which is what the previous ``file``-based
    routing could not do: it sent every gzip file to the JSONL reader, so a CSV load
    failed with a raw JSON decode error."""
    path = loader_files[label]
    assert path.endswith(LOADER_FILES[label][2]), path

    assert _payload(load_dlt_file(path)) == TESTDATA


def test_loader_needs_no_external_file_command(loader_files, monkeypatch):
    """The loader must not shell out: ``file`` is absent on Windows and its output
    varies by version. An emptied ``PATH`` makes a ``file`` subprocess raise
    ``FileNotFoundError``, which is how this asserts something. Emptied, not deleted:
    with ``PATH`` unset, the subprocess falls back to a default search path and finds
    the command anyway, so the test would pass against a loader that still shells out.
    """
    monkeypatch.setenv("PATH", "")

    for label in LOADER_FILES:
        assert _payload(load_dlt_file(loader_files[label])) == TESTDATA


def test_loader_refuses_a_format_it_cannot_read(loader_files):
    """``insert_values`` is a dlt loader file format with no row-shaped reading here.
    It used to be gzipped, matched the gzip branch and was misread as JSONL, so it
    failed with a JSON decode error naming a column position. Now it is named."""
    unreadable = str(
        pathlib.Path(loader_files["jsonl"]).with_name("x.insert_values.gz")
    )

    with pytest.raises(UnsupportedLoaderFileFormat, match="insert_values"):
        factory(unreadable)


def test_loader_refuses_a_file_with_no_extension(loader_files):
    """dlt names every loader file with its format, so a name without one is not a dlt
    loader file. Refusing beats guessing: the two compressed formats are identical
    bytes, so a guess would silently pick the wrong reader for one of them."""
    nameless = str(pathlib.Path(loader_files["jsonl"]).with_name("no-extension-here"))

    with pytest.raises(UnsupportedLoaderFileFormat, match="no-extension-here"):
        factory(nameless)


def test_loader_never_invokes_a_subprocess(loader_files, monkeypatch):
    """Stronger than the PATH check: an emptied PATH proves the loader does not depend
    on the command succeeding, not that it never runs one (an absolute path bypasses
    PATH, and a caught FileNotFoundError could hide a fallback). This proves it."""
    import subprocess

    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: calls.append(a))
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: calls.append(a))

    for label in LOADER_FILES:
        assert _payload(load_dlt_file(loader_files[label])) == TESTDATA

    assert calls == []


def test_loader_reads_the_csv_dialect_dlt_wrote(monkeypatch):
    """dlt's CSV writer takes its delimiter from configuration. Reading with a hardcoded
    comma parses every row into one composite column named by the whole header line, so
    the load reports success and the rows are unusable."""
    monkeypatch.setenv("DATA_WRITER__DELIMITER", "|")

    path = _write_with_dlt("csv", disable_compression=False)

    assert _payload(load_dlt_file(path)) == TESTDATA
