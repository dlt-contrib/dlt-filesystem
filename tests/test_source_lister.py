import datetime as dt
import time

import pytest
from dlt.common.pendulum import pendulum
from dlt.common.storages.fsspec_filesystem import MTIME_DISPATCH
from fsspec import AbstractFileSystem
from fsspec.implementations.arrow import ArrowFSWrapper
from pyarrow.fs import FileInfo, FileSelector, FileType

from dlt_filesystem.source.lister import glob_files, resolve_modification_date
from dlt_filesystem.util.time import ensure_datetime_utc

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

    def modified(self, path):
        self.modified_path = path
        return MODIFIED


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


def test_http_header_wins_before_dlts_synthesized_fallback(monkeypatch):
    """A concrete HTTP file keeps its header without a second metadata request."""
    fs = stub_filesystem("http", listing())
    monkeypatch.setitem(MTIME_DISPATCH, "http", lambda _: "dlt-now")

    resolved = resolve_modification_date(
        "http",
        listing(**{"Last-Modified": "Thu, 20 Aug 2026 22:39:23 GMT"}),
        fs,
    )

    assert resolved == dt.datetime(2026, 8, 20, 22, 39, 23, tzinfo=dt.timezone.utc)
    assert not hasattr(fs, "modified_path")


def test_http_missing_listing_header_uses_filesystem_not_dlt(monkeypatch):
    """An index entry is enriched instead of receiving dlt's current time."""
    fs = stub_filesystem("http", listing())
    monkeypatch.setitem(MTIME_DISPATCH, "http", lambda _: "dlt-now")
    info = listing(name="http://example.test/report.csv")

    resolved = resolve_modification_date("http", info, fs)

    assert resolved == MODIFIED
    assert fs.modified_path == "http://example.test/report.csv"


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
        entries = [
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
        if isinstance(selector, list):
            return [entry for entry in entries if entry.path in selector]
        if selector.recursive:
            return entries
        return [
            entry for entry in entries if "/" not in entry.path[len("bucket/data/") :]
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


def test_arrow_shallow_glob_does_not_enumerate_nested_prefixes():
    """A one-directory glob stays one native request without scanning descendants."""
    arrow_fs = RecordingArrowFilesystem()
    fs = ArrowFSWrapper(arrow_fs)

    files = list(glob_files(fs, "s3://bucket", "data/*.csv"))

    assert [file["relative_path"] for file in files] == ["data/part-1.csv"]
    assert len(arrow_fs.calls) == 1
    selector = arrow_fs.calls[0]
    assert selector.base_dir == "bucket/data"
    assert selector.recursive is False


def test_arrow_concrete_path_uses_one_native_metadata_request():
    arrow_fs = RecordingArrowFilesystem()
    fs = ArrowFSWrapper(arrow_fs)

    files = list(glob_files(fs, "s3://bucket", "data/part-1.csv"))

    assert [file["relative_path"] for file in files] == ["data/part-1.csv"]
    assert arrow_fs.calls == [["bucket/data/part-1.csv"]]


@pytest.mark.parametrize(
    "value,expected",
    [
        (MODIFIED.replace(tzinfo=None), MODIFIED),
        (MODIFIED.date(), MODIFIED.replace(hour=0, minute=0)),
        (MODIFIED.astimezone(dt.timezone(dt.timedelta(hours=5, minutes=30))), MODIFIED),
        (MODIFIED.astimezone(dt.timezone(dt.timedelta(hours=-7))), MODIFIED),
        ("2026-08-04T15:00:00+05:30", MODIFIED),
        ("2026-08-04T02:30:00-07:00", MODIFIED),
        (EPOCH_SECONDS, MODIFIED),
        (EPOCH_SECONDS + 0.125, MODIFIED.replace(microsecond=125000)),
        ("2026-08-04T09:30:00.123456Z", MODIFIED.replace(microsecond=123456)),
    ],
)
def test_utc_coercion(value, expected):
    result = ensure_datetime_utc(value)
    assert isinstance(result, pendulum.DateTime)
    assert result == expected
    assert result.utcoffset() == dt.timedelta(0)


@pytest.mark.parametrize("value", [None, [], {}, object(), "not-a-date"])
def test_utc_coercion_rejects_unsupported_values(value):
    with pytest.raises((TypeError, ValueError)):
        ensure_datetime_utc(value)
    with pytest.raises(ValueError, match="no usable modification date"):
        resolve_modification_date("acme", listing(mtime=value))


@pytest.mark.parametrize(
    "value",
    [
        MODIFIED.replace(tzinfo=None),
        MODIFIED.date(),
        EPOCH_SECONDS,
        "2026-08-04T09:30:00",
    ],
)
def test_utc_coercion_ignores_dlt_timezone_context(value):
    timezone_module = pytest.importorskip(
        "dlt.common.configuration.specs.timezone_context"
    )
    from dlt.common.configuration.container import Container

    with Container().injectable_context(
        timezone_module.TimezoneContext("America/New_York")
    ):
        result = ensure_datetime_utc(value)
    expected = (
        MODIFIED.replace(hour=0, minute=0) if type(value) is dt.date else MODIFIED
    )
    assert result == expected
    assert result.utcoffset() == dt.timedelta(0)


@pytest.mark.parametrize(
    "scheme,key,value",
    [
        ("smb", "mtime", EPOCH_SECONDS + 0.125),
        ("webhdfs", "modificationTime", EPOCH_SECONDS * 1000 + 125),
        ("gdrive", "modifiedTime", "2026-08-04T15:00:00.125+05:30"),
        ("oci", "timeModified", "2026-08-04T02:30:00.125-07:00"),
        ("dbfs", "modified", "2026-08-04T09:30:00.125Z"),
    ],
)
def test_backend_timestamps_are_utc(scheme, key, value):
    result = resolve_modification_date(scheme, listing(**{key: value}))
    assert result == MODIFIED.replace(microsecond=125000)
    assert result.utcoffset() == dt.timedelta(0)


@pytest.mark.skipif(
    not hasattr(time, "tzset"), reason="requires process timezone support"
)
def test_webhdfs_timestamp_is_independent_of_host_timezone(monkeypatch):
    try:
        with monkeypatch.context() as timezone_patch:
            timezone_patch.setenv("TZ", "EST5EDT")
            time.tzset()
            assert dt.datetime.fromtimestamp(EPOCH_SECONDS).hour != MODIFIED.hour
            result = resolve_modification_date(
                "webhdfs", listing(modificationTime=EPOCH_SECONDS * 1000 + 125)
            )
            assert result == MODIFIED.replace(microsecond=125000)
            assert result.utcoffset() == dt.timedelta(0)
    finally:
        time.tzset()


@pytest.mark.parametrize(
    "value",
    ["Tue, 04 Aug 2026 15:00:00 +0530", "Tue, 04 Aug 2026 02:30:00 -0700"],
)
def test_http_listing_normalizes_timezone(value):
    result = resolve_modification_date("http", listing(**{"Last-Modified": value}))
    assert result == MODIFIED
    assert result.utcoffset() == dt.timedelta(0)
