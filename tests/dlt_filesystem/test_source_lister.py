import datetime as dt

import pytest
from dlt.common.storages.fsspec_filesystem import MTIME_DISPATCH
from fsspec import AbstractFileSystem
from fsspec.implementations.arrow import ArrowFSWrapper
from pyarrow.fs import FileInfo, FileSelector, FileType

from dlt_filesystem.source.lister import glob_files, resolve_modification_date

MODIFIED = dt.datetime(2026, 8, 4, 9, 30, tzinfo=dt.timezone.utc)
EPOCH_SECONDS = MODIFIED.timestamp()


class StubFilesystem(AbstractFileSystem):
    """A filesystem client that lists one file with a caller-supplied shape."""

    def __init__(self, file_info: dict):
        super().__init__()
        self.file_info = file_info

    def _strip_protocol(self, url: str) -> str:  # ty: ignore[invalid-method-override]
        return url.split("://", 1)[-1].rstrip("/")

    def glob(self, path, maxdepth=None, **kwargs) -> dict:
        return {"bucket/data/report.csv": self.file_info}


def stub_filesystem(protocol: str, file_info: dict) -> StubFilesystem:
    """Return a stub client that reports `protocol` as its fsspec protocol."""
    return type("StubFilesystem", (StubFilesystem,), {"protocol": protocol})(file_info)


def listing(**overrides) -> dict:
    """One entry of a filesystem listing, minus its modification-date key."""
    return {"name": "bucket/data/report.csv", "size": 12, "type": "file", **overrides}


@pytest.mark.parametrize(
    "scheme,key,raw",
    [
        ("s3", "LastModified", MODIFIED),
        ("s3a", "LastModified", MODIFIED),
        ("gs", "updated", MODIFIED),
        ("gcs", "updated", MODIFIED),
        ("az", "last_modified", MODIFIED),
        ("abfss", "last_modified", MODIFIED),
        ("file", "mtime", EPOCH_SECONDS),
        ("gdrive", "modifiedTime", MODIFIED),
    ],
)
def test_known_scheme_keeps_dlt_extractor(scheme, key, raw):
    """Schemes dlt knows resolve through dlt's own per-scheme extractor."""
    assert scheme in MTIME_DISPATCH
    assert resolve_modification_date(scheme, listing(**{key: raw})) == MODIFIED


@pytest.mark.parametrize(
    "scheme,key,raw",
    [
        # Each value is the shape the backend's own `modified()` reads.
        ("r2", "LastModified", MODIFIED),  # s3fs
        ("oss", "LastModified", MODIFIED),  # ossfs
        ("hdfs", "mtime", MODIFIED),  # pyarrow.fs via ArrowFSWrapper
        ("smb", "mtime", EPOCH_SECONDS),  # os.stat_result.st_mtime
        ("ftp", "modify", "20260804093000"),  # RFC 3659 MLSD fact
        ("dbfs", "modified", MODIFIED),  # fsspec-databricks
        ("oci", "timeModified", MODIFIED),  # ocifs
        ("webhdfs", "modificationTime", int(EPOCH_SECONDS * 1000)),  # epoch millis
    ],
)
def test_scheme_dlt_does_not_know_resolves_from_the_listing(scheme, key, raw):
    """Schemes absent from dlt's table resolve from the key their backend emits."""
    assert scheme not in MTIME_DISPATCH
    assert resolve_modification_date(scheme, listing(**{key: raw})) == MODIFIED


def test_known_scheme_with_foreign_key_falls_back():
    """A pyarrow.fs client addressed as `s3://` reports `mtime`, not `LastModified`."""
    assert resolve_modification_date("s3", listing(mtime=MODIFIED)) == MODIFIED


def test_access_time_is_not_read_as_a_modification_date():
    """SMB carries both `mtime` and `time`, where `time` is the access time."""
    accessed = dt.datetime(2026, 8, 4, 18, 0, tzinfo=dt.timezone.utc)
    info = listing(mtime=EPOCH_SECONDS, time=accessed.timestamp())

    assert resolve_modification_date("smb", info) == MODIFIED


def test_listing_without_a_modification_date_names_what_it_saw():
    with pytest.raises(ValueError) as excinfo:
        resolve_modification_date("acme", listing())
    message = str(excinfo.value)
    assert "'acme'" in message
    assert "'name', 'size', 'type'" in message


def test_year_less_ftp_timestamp_is_rejected_with_its_value():
    """fsspec parses `dir` output into a year-less `modify` on servers without MLSD."""
    with pytest.raises(ValueError) as excinfo:
        resolve_modification_date("ftp", listing(modify="Aug 4 09:30"))
    message = str(excinfo.value)
    assert "no usable modification date" in message
    assert "modify='Aug 4 09:30'" in message


@pytest.mark.parametrize("scheme", ["s3", "r2", "oss"])
def test_glob_files_lists_s3_compatible_schemes(scheme):
    """Listing an S3-compatible bucket yields files whatever scheme addresses it."""
    fs = stub_filesystem(scheme, listing(LastModified=MODIFIED))

    files = list(glob_files(fs, f"{scheme}://bucket/data", "*.csv"))

    assert len(files) == 1
    assert files[0]["file_name"] == "report.csv"
    assert files[0]["relative_path"] == "report.csv"
    assert files[0]["file_url"] == f"{scheme}://bucket/data/report.csv"
    assert files[0]["modification_date"] == MODIFIED
    assert files[0]["size_in_bytes"] == 12


def test_glob_files_lists_arrow_backed_clients():
    """A pyarrow.fs-backed client lists through the same path as its fsspec peer."""
    fs = stub_filesystem("s3", listing(mtime=MODIFIED))

    files = list(glob_files(fs, "s3://bucket/data", "*.csv"))

    assert [file["modification_date"] for file in files] == [MODIFIED]


class RecordingArrowFilesystem:
    """Return a fixed recursive listing and record each native Arrow request."""

    type_name = "s3"

    def __init__(self):
        self.calls = []

    def get_file_info(self, selector):
        self.calls.append(selector)
        return [
            FileInfo("bucket/data", FileType.Directory),
            FileInfo("bucket/data/part-1.csv", FileType.File, mtime=MODIFIED, size=12),
            FileInfo(
                "bucket/data/nested/part-2.csv",
                FileType.File,
                mtime=MODIFIED,
                size=13,
            ),
            FileInfo(
                "bucket/data/ignored.jsonl",
                FileType.File,
                mtime=MODIFIED,
                size=14,
            ),
        ]


def test_arrow_glob_uses_one_recursive_native_listing_request():
    """A recursive Arrow glob does not inherit fsspec's level-by-level walk."""
    arrow_fs = RecordingArrowFilesystem()
    fs = ArrowFSWrapper(arrow_fs)

    files = list(glob_files(fs, "s3://bucket", "data/**/*.csv"))

    assert [file["relative_path"] for file in files] == [
        "data/nested/part-2.csv",
        "data/part-1.csv",
    ]
    assert len(arrow_fs.calls) == 1
    selector = arrow_fs.calls[0]
    assert isinstance(selector, FileSelector)
    assert selector.base_dir == "bucket/data"
    assert selector.recursive is True
    assert selector.allow_not_found is True
