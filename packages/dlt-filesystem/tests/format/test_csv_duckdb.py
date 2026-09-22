from fsspec.implementations.arrow import ArrowFSWrapper
from pyarrow import RecordBatch
from pyarrow.fs import LocalFileSystem

from dlt_filesystem.source.core import resource_for_reader
from dlt_filesystem.source.fsspec.local import LocalFilesystemSource
from dlt_filesystem.source.model import FilesystemReference


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
