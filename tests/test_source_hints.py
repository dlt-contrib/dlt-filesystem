"""Source-level unit tests: named `#key=value` hints thread into FilesystemReference.

Mock-only (no Docker, no credentials): `resource_for_reader` is patched to capture
the constructed `FilesystemReference`, and each source's fs constructor is patched so
nothing connects. The warehouse S3/GCS matrix keeps the end-to-end regression cover;
these prove the hint plumbing per source in isolation.
"""

from unittest.mock import MagicMock, patch

import fsspec
import pytest

from dlt_filesystem.source.core import resource_for_reader
from dlt_filesystem.source.format.registry import READER_REGISTRATIONS
from dlt_filesystem.source.fsspec.local import LocalFilesystemSource
from dlt_filesystem.source.model import FilesystemReference
from dlt_filesystem.source.impl.remote import (
    AzureSource,
    GCSSource,
    S3Source,
    SFTPSource,
)

RESOURCE_FOR_READER = "dlt_filesystem.source.core.resource_for_reader"

GCS_URI = "gs://?credentials_base64=e30K"  # base64 for "{}"
S3_URI = "s3://?access_key_id=KEY&secret_access_key=SECRET"


def _captured_ref(rfr: MagicMock):
    assert rfr.call_count == 1
    return rfr.call_args.args[0]


@pytest.mark.parametrize(
    ("uri", "expected_hints"),
    [
        # no fragment -> hints default to {}
        ("file://book.csv", {}),
        # a named hint on a plain resolvable file
        ("file://book.xlsx#sheet_name=foo", {"sheet_name": "foo"}),
        # format hint (#xlsx resolves the reader) and named hint coexist
        ("file://feed.dat#xlsx&sheet_name=foo", {"sheet_name": "foo"}),
    ],
)
def test_local_threads_hints(uri: str, expected_hints: dict[str, str]):
    with (
        patch(RESOURCE_FOR_READER) as rfr,
        patch("fsspec.filesystem"),
    ):
        LocalFilesystemSource().dlt_source(uri, "")
    assert _captured_ref(rfr).hints == expected_hints


@pytest.mark.parametrize(
    ("table", "expected_hints"),
    [
        ("bucket/book.csv", {}),
        ("bucket/book.xlsx#sheet_name=foo", {"sheet_name": "foo"}),
        ("bucket/feed.dat#xlsx&sheet_name=foo", {"sheet_name": "foo"}),
    ],
)
def test_gcs_threads_hints(table: str, expected_hints: dict[str, str]):
    with (
        patch(RESOURCE_FOR_READER) as rfr,
        patch("gcsfs.GCSFileSystem"),
    ):
        GCSSource().dlt_source(GCS_URI, table)
    assert _captured_ref(rfr).hints == expected_hints


@pytest.mark.parametrize(
    ("table", "expected_hints"),
    [
        ("bucket/book.csv", {}),
        ("bucket/book.xlsx#sheet_name=foo", {"sheet_name": "foo"}),
        ("bucket/feed.dat#xlsx&sheet_name=foo", {"sheet_name": "foo"}),
    ],
)
def test_s3_threads_hints(table: str, expected_hints: dict[str, str]):
    with (
        patch(RESOURCE_FOR_READER) as rfr,
        patch("pyarrow.fs.S3FileSystem"),
    ):
        S3Source().dlt_source(S3_URI, table)
    assert _captured_ref(rfr).hints == expected_hints


def test_s3_threads_hints_from_uri_path_form():
    """The `s3://bucket/path#frag` form carries the fragment in
    URI-path's `fragment`; blob_hints reconstructs it so hints still thread."""
    with (
        patch(RESOURCE_FOR_READER) as rfr,
        patch("pyarrow.fs.S3FileSystem"),
    ):
        S3Source().dlt_source(
            "s3://bucket/book.xlsx?access_key_id=KEY&secret_access_key=SECRET#sheet_name=foo",
            "",
        )
    assert _captured_ref(rfr).hints == {"sheet_name": "foo"}


def test_s3_blob_hints_track_the_loaded_file_when_both_forms_given():
    """When a URI path and a --source-table are both supplied (a contradictory,
    deprecated shape), parse_uri loads the URI-path file, so blob_hints must read
    the URI fragment, not the ignored table's."""
    with (
        patch(RESOURCE_FOR_READER) as rfr,
        patch("pyarrow.fs.S3FileSystem"),
    ):
        S3Source().dlt_source(
            "s3://bucket/loaded.xlsx?access_key_id=KEY&secret_access_key=SECRET#sheet_name=uri",
            "ignored/other.xlsx#sheet_name=table",
        )
    assert _captured_ref(rfr).hints == {"sheet_name": "uri"}


@pytest.mark.parametrize(
    ("table", "expected_hints"),
    [
        ("/book.csv", {}),
        ("/book.xlsx#sheet_name=foo", {"sheet_name": "foo"}),
        ("/feed.dat#xlsx&sheet_name=foo", {"sheet_name": "foo"}),
    ],
)
def test_sftp_threads_hints(table: str, expected_hints: dict[str, str]):
    with (
        patch(RESOURCE_FOR_READER) as rfr,
        patch("fsspec.filesystem"),
    ):
        SFTPSource().dlt_source("sftp://user:pass@host:22", table)
    assert _captured_ref(rfr).hints == expected_hints


<<<<<<< HEAD
@pytest.mark.parametrize(
    "reader_name", [entry.reader_name for entry in READER_REGISTRATIONS]
)
def test_unknown_hint_names_reader_and_accepted_hints(tmp_path, reader_name):
    ref = FilesystemReference(
        fs=fsspec.filesystem("file"),
        bucket_url=str(tmp_path),
        file_glob="*.none",
        reader_name=reader_name,
        hints={"not_a_hint": "1"},
    )
    with pytest.raises(TypeError) as exc:
        resource_for_reader(ref)
    message = str(exc.value)
    assert reader_name in message
    assert "not_a_hint" in message
    assert "Accepted hints:" in message
    assert "items" not in message
    assert "worksheet_names" not in message


@pytest.mark.parametrize(
    "reader_name, hint",
    [
        ("read_csv", "source"),
        ("read_csv_duckdb", "path_or_buffer"),
        ("read_excel", "items"),
        ("read_excel", "worksheet_names"),
    ],
)
def test_internal_reader_arguments_are_not_hints(tmp_path, reader_name, hint):
    ref = FilesystemReference(
        fs=fsspec.filesystem("file"),
        bucket_url=str(tmp_path),
        file_glob="*.none",
        reader_name=reader_name,
        hints={hint: "ignored"},
    )
    with pytest.raises(TypeError, match="Accepted hints:"):
        resource_for_reader(ref)


@pytest.mark.parametrize(
    "fragment",
    [
        '#csv_headless&column_names=["a","b"]',
        '#csv_headless&new_columns=["a","b"]',
    ],
)
def test_headerless_column_names_hint_has_no_duplicate_binding(tmp_path, fragment):
    path = tmp_path / "rows.csv"
    path.write_text("1;2\n3;4\n")
    result = list(
        LocalFilesystemSource().dlt_source(f"file://{path}{fragment}&separator=;", "")
    )
    assert result == [{"a": 1, "b": 2}, {"a": 3, "b": 4}]


def test_headerless_column_types_take_precedence_over_naming_hints(tmp_path):
    path = tmp_path / "rows.csv"
    path.write_text("1;2\n3;4\n")
    ref = FilesystemReference(
        fs=fsspec.filesystem("file"),
        bucket_url=str(tmp_path),
        file_glob=path.name,
        reader_name="read_csv_headless",
        column_types={"run_a": "int", "run_b": "int"},
        hints={
            "column_names": '["hint_a","hint_b"]',
            "new_columns": '["polars_a","polars_b"]',
            "separator": ";",
        },
    )
    assert list(resource_for_reader(ref)) == [
        {"run_a": 1, "run_b": 2},
        {"run_a": 3, "run_b": 4},
    ]
=======
AZURE_QUERY = "account_name=acct&account_key=a2V5"


@pytest.mark.parametrize(
    ("source_cls", "constructor", "uri"),
    [
        (
            S3Source,
            "pyarrow.fs.S3FileSystem",
            "s3://bucket/data.txt?access_key_id=KEY&secret_access_key=SECRET#csv",
        ),
        (
            GCSSource,
            "gcsfs.GCSFileSystem",
            "gs://bucket/data.txt?credentials_base64=e30K#csv",
        ),
        (
            AzureSource,
            "pyarrow.fs.AzureFileSystem",
            f"az://container/data.txt?{AZURE_QUERY}#csv",
        ),
        (SFTPSource, "fsspec.filesystem", "sftp://user:pass@host:22/data.txt#csv"),
    ],
)
def test_remote_sources_honour_a_uri_format_hint(source_cls, constructor, uri):
    """A `#csv` on the URI names the reader on every remote transport, as on HTTP and local."""
    with patch(RESOURCE_FOR_READER) as rfr, patch(constructor):
        source_cls().dlt_source(uri, "")
    ref = _captured_ref(rfr)
    assert (ref.file_glob, ref.reader_name) == ("data.txt", "read_csv")


def test_s3_reader_follows_the_listed_file_not_the_ignored_table():
    """The table is ignored for the path, so its extension must not pick the reader."""
    with patch(RESOURCE_FOR_READER) as rfr, patch("pyarrow.fs.S3FileSystem"):
        S3Source().dlt_source(
            "s3://bucket/a.csv?access_key_id=KEY&secret_access_key=SECRET", "b.parquet"
        )
    ref = _captured_ref(rfr)
    assert (ref.file_glob, ref.reader_name) == ("a.csv", "read_csv")


def test_s3_table_directives_apply_beside_a_uri_path():
    """A table fragment is a directive, not a path, so it still applies where the URI has none."""
    with patch(RESOURCE_FOR_READER) as rfr, patch("pyarrow.fs.S3FileSystem"):
        S3Source().dlt_source(
            "s3://bucket/feed.dat?access_key_id=KEY&secret_access_key=SECRET",
            "ignored#xlsx&sheet_name=foo",
        )
    ref = _captured_ref(rfr)
    assert (ref.file_glob, ref.reader_name) == ("feed.dat", "read_excel")
    assert ref.hints == {"sheet_name": "foo"}


def test_s3_uri_directive_applies_on_the_split_form():
    """A URI fragment applies whichever carrier names the path."""
    with patch(RESOURCE_FOR_READER) as rfr, patch("pyarrow.fs.S3FileSystem"):
        S3Source().dlt_source(
            "s3://bucket?access_key_id=KEY&secret_access_key=SECRET#csv", "data.txt"
        )
    ref = _captured_ref(rfr)
    assert (ref.bucket_url, ref.file_glob, ref.reader_name) == (
        "s3://bucket/",
        "data.txt",
        "read_csv",
    )


def test_s3_scheme_qualified_table_drops_userinfo():
    """Keeping the bucket's case must not keep a `user:pass@` in front of it."""
    with patch(RESOURCE_FOR_READER) as rfr, patch("pyarrow.fs.S3FileSystem"):
        S3Source().dlt_source(
            "s3://?access_key_id=KEY&secret_access_key=SECRET",
            "s3://user:pass@bucket/a.csv",
        )
    assert _captured_ref(rfr).bucket_url == "s3://bucket/"


def test_local_table_format_hint_applies_beside_a_uri_path():
    with patch(RESOURCE_FOR_READER) as rfr:
        LocalFilesystemSource().dlt_source("file://data.dat", "other.csv#csv")
    ref = _captured_ref(rfr)
    assert ref.reader_name == "read_csv"
    assert ref.file_glob == "data.dat"
>>>>>>> 92bb971 (fix(source): read the path, reader and directives off one carrier)
