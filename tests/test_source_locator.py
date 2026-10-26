"""Locator-based sources: which carrier names the files, and what that resolves to.

Mock-only (no services, no credentials). Each source's filesystem constructor is
patched, and `resource_for_reader` is patched to capture the `FilesystemReference`
the source builds, so these pin the addressing itself: `bucket_url`, `file_glob`,
the reader, and the incremental state key. Where a filesystem's own path handling
decides whether a listing can work, its real `_strip_protocol` is applied to the
captured reference.
"""

import posixpath
from contextlib import ExitStack
from unittest.mock import patch

import pytest
from fsspec.implementations.memory import MemoryFileSystem

from dlt_filesystem.source.fsspec.databricks import DatabricksSource
from dlt_filesystem.source.fsspec.dropbox import DropboxSource
from dlt_filesystem.source.fsspec.ftp import FTPSource
from dlt_filesystem.source.fsspec.gdrive import GoogleDriveSource
from dlt_filesystem.source.fsspec.hdfs import HDFSSource
from dlt_filesystem.source.fsspec.http import HttpFilesystemSource
from dlt_filesystem.source.fsspec.oci import OCISource
from dlt_filesystem.source.fsspec.onedrive import OneDriveSource
from dlt_filesystem.source.fsspec.oss import OSSSource
from dlt_filesystem.source.fsspec.sharepoint import SharePointSource
from dlt_filesystem.source.fsspec.smb import SMBSource
from dlt_filesystem.source.fsspec.webdav import WebdavSource
from dlt_filesystem.source.fsspec.webhdfs import WebHDFSSource
from dlt_filesystem.source.lister import file_url_for
from dlt_filesystem.source.model import FilesystemReference

RESOURCE_FOR_READER = "dlt_filesystem.source.core.resource_for_reader"


def build_reference(source_cls, uri: str, table: str) -> FilesystemReference:
    """Capture the reference a locator-based source builds, constructing nothing."""
    captured: dict = {}

    def fake_reader(ref: FilesystemReference):
        captured["ref"] = ref
        return "SENTINEL"

    with ExitStack() as stack:
        stack.enter_context(patch(RESOURCE_FOR_READER, fake_reader))
        if source_cls is FTPSource:
            # The FTP source names its filesystem class inline, not as a property.
            stack.enter_context(patch("fsspec.implementations.ftp.FTPFileSystem"))
        else:
            stack.enter_context(
                patch.object(
                    source_cls,
                    "fs_class",
                    property(lambda self: lambda *args, **kwargs: MemoryFileSystem()),
                )
            )
        assert source_cls().dlt_source(uri, table) == "SENTINEL"
    return captured["ref"]


def databricks_path(ref: FilesystemReference) -> str:
    """The path a Databricks listing asks for, by the client's own stripping."""
    from fsspec_databricks import DatabricksFileSystem

    return posixpath.join(
        DatabricksFileSystem._strip_protocol(ref.bucket_url), ref.file_glob
    )


# Each pair addresses one location twice: the path on the URI, and the path on the
# table beside a URI that names none. Both must resolve to the same files.
SPLIT_EQUALS_URI_FORM = [
    pytest.param(
        DropboxSource,
        ("dropbox://path/to/data.csv?token=t", ""),
        ("dropbox://?token=t", "path/to/data.csv"),
        id="dropbox",
    ),
    pytest.param(
        SharePointSource,
        ("sharepoint://site/drive/data.csv?client_id=c", ""),
        ("sharepoint://?client_id=c", "site/drive/data.csv"),
        id="sharepoint",
    ),
    pytest.param(
        OneDriveSource,
        ("onedrive://drive/data.csv?client_id=c", ""),
        ("onedrive://?client_id=c", "drive/data.csv"),
        id="onedrive",
    ),
    pytest.param(
        GoogleDriveSource,
        ("gdrive://path/to/data.parquet?token=anon", ""),
        ("gdrive://?token=anon", "path/to/data.parquet"),
        id="gdrive",
    ),
    pytest.param(
        GoogleDriveSource,
        ("gdrive://MixedFolder/data.csv?token=anon", ""),
        ("gdrive://?token=anon", "gdrive://MixedFolder/data.csv"),
        id="gdrive-scheme-qualified-table",
    ),
    pytest.param(
        OSSSource,
        ("oss://bucket/path/to/data.parquet?endpoint=http://oss.example.com/", ""),
        ("oss://?endpoint=http://oss.example.com/", "bucket/path/to/data.parquet"),
        id="oss",
    ),
    pytest.param(
        OCISource,
        ("oci://bucket@ns/prefix/data.parquet", ""),
        ("oci://bucket@ns", "/prefix/data.parquet"),
        id="oci",
    ),
    pytest.param(
        OCISource,
        ("oci://Bucket@NameSpace/prefix/data.parquet", ""),
        ("oci://", "oci://Bucket@NameSpace/prefix/data.parquet"),
        id="oci-scheme-qualified-table",
    ),
    pytest.param(
        HDFSSource,
        ("hdfs://example.com:8020/path/to/data.parquet?user=test", ""),
        ("hdfs://example.com:8020/?user=test", "path/to/data.parquet"),
        id="hdfs-root-path",
    ),
    pytest.param(
        FTPSource,
        ("ftp://user:pw@ftp.example.com/path/to/data.csv", ""),
        ("ftp://user:pw@ftp.example.com", "/path/to/data.csv"),
        id="ftp",
    ),
    pytest.param(
        SMBSource,
        ("smb://user:pw@server.example.com:445/share/data.parquet", ""),
        ("smb://user:pw@server.example.com:445", "share/data.parquet"),
        id="smb",
    ),
    pytest.param(
        WebHDFSSource,
        ("webhdfs://host:9870/path/to/data.parquet", ""),
        ("webhdfs://host:9870", "path/to/data.parquet"),
        id="webhdfs",
    ),
    pytest.param(
        WebdavSource,
        ("https+webdav://user:pw@dav.example.com/path/data.csv", ""),
        ("https+webdav://user:pw@dav.example.com", "path/data.csv"),
        id="webdav",
    ),
    pytest.param(
        HttpFilesystemSource,
        ("https://example.com/path/data.csv", ""),
        ("https://example.com", "path/data.csv"),
        id="http",
    ),
    pytest.param(
        DatabricksSource,
        ("dbfs:/Volumes/c/s/v/data.csv?host=https://adb.example.com&token=one", ""),
        ("dbfs://?host=https://adb.example.com&token=two", "/Volumes/c/s/v/data.csv"),
        id="databricks",
    ),
]


@pytest.mark.parametrize(
    ("source_cls", "uri_form", "split_form"), SPLIT_EQUALS_URI_FORM
)
def test_split_form_addresses_what_the_uri_form_does(source_cls, uri_form, split_form):
    """Which carrier names the path must not change the files, the reader or the state."""
    by_uri = build_reference(source_cls, *uri_form)
    by_table = build_reference(source_cls, *split_form)

    assert (by_table.bucket_url, by_table.file_glob, by_table.reader_name) == (
        by_uri.bucket_url,
        by_uri.file_glob,
        by_uri.reader_name,
    )
    assert by_table.incremental_resource_name == by_uri.incremental_resource_name


@pytest.mark.parametrize(
    ("source_cls", "uri", "table"),
    [
        (DropboxSource, "dropbox://?token=t", "path/to/data.csv"),
        (SharePointSource, "sharepoint://?client_id=c", "site/drive/data.csv"),
        (OneDriveSource, "onedrive://?client_id=c", "drive/data.csv"),
        (GoogleDriveSource, "gdrive://?token=anon", "path/to/data.parquet"),
    ],
)
def test_split_form_takes_the_bucket_from_the_table(source_cls, uri, table):
    """A URI without a host no longer fails to name a `bucket_url`."""
    ref = build_reference(source_cls, uri, table)

    bucket, _, glob = table.partition("/")
    assert ref.bucket_url == f"{uri.split('://')[0]}://{bucket}"
    assert ref.file_glob == glob


def test_a_selected_bucket_wins_over_the_endpoint_surrogate():
    """`endpoint` stands in for a host only when the selection names no bucket.

    Taking it first addressed `Actual/data.csv` and dropped `Placeholder` from the
    selection, which for OSS, whose `endpoint` is the service URL, could not work.
    """
    ref = build_reference(
        GoogleDriveSource, "gdrive://?endpoint=Actual", "Placeholder/data.csv"
    )

    assert (ref.bucket_url, ref.file_glob) == ("gdrive://Placeholder", "data.csv")


def test_root_only_uri_path_defers_to_the_table():
    """`hdfs://host:8020/` names no path, so the documented split form reads the table."""
    ref = build_reference(
        HDFSSource, "hdfs://example.com:8020/?user=test", "path/to/data.parquet"
    )

    assert ref.bucket_url == "hdfs://example.com:8020"
    assert ref.file_glob == "path/to/data.parquet"
    assert ref.reader_name == "read_parquet"


def test_uri_path_beside_a_table_reads_the_uri_path_by_its_own_extension():
    """The reader comes from the file that is listed, never from the ignored table."""
    with pytest.raises(ValueError, match="only supports file formats"):
        build_reference(
            WebHDFSSource, "webhdfs://host:9870/path", "path/to/data.parquet"
        )


@pytest.mark.parametrize(
    ("uri", "table"),
    [
        ("dbfs:/Volumes/c/s/v/data.csv", ""),
        ("dbfs://?token=t", "/Volumes/c/s/v/data.csv"),
        ("dbfs://?token=t", "Volumes/c/s/v/data.csv"),
    ],
)
def test_databricks_lists_the_selected_path_from_the_root(uri, table):
    """The URI was the `bucket_url`, so the listing doubled the path."""
    ref = build_reference(DatabricksSource, uri, table)

    assert ref.bucket_url == "dbfs:/"
    assert databricks_path(ref) == "/Volumes/c/s/v/data.csv"


def test_databricks_state_key_survives_token_rotation():
    old = build_reference(
        DatabricksSource, "dbfs:/Volumes/c/s/v/data.csv?token=OLD", ""
    )
    new = build_reference(
        DatabricksSource, "dbfs:/Volumes/c/s/v/data.csv?token=NEW", ""
    )

    assert old.incremental_resource_name == new.incremental_resource_name
    assert "OLD" not in old.bucket_url


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("host=https://prod.example.com", "host=https://staging.example.com"),
        ("host=https://prod.example.com", "profile=prod.example.com"),
        ("profile=prod", "profile=staging"),
        ("profile=prod", ""),
    ],
)
def test_databricks_state_key_separates_workspaces(first, second):
    """With the URI out of `bucket_url`, the workspace has to be named elsewhere."""
    path = "dbfs:/Volumes/c/s/v/data.csv"
    one = build_reference(DatabricksSource, f"{path}?{first}", "")
    two = build_reference(DatabricksSource, f"{path}?{second}" if second else path, "")

    assert one.incremental_resource_name != two.incremental_resource_name


def test_databricks_open_url_is_one_the_client_accepts():
    """dlt's fallback composes `dbfs:///Volumes/...`, which the client rejects."""
    from fsspec_databricks import DatabricksFileSystem

    url = file_url_for("dbfs", "/Volumes/c/s/v/data.csv", "dbfs:/")

    assert url == "dbfs:/Volumes/c/s/v/data.csv"
    assert DatabricksFileSystem._strip_protocol(url) == "/Volumes/c/s/v/data.csv"


def test_databricks_reads_rows_end_to_end():
    """List and open through a memory filesystem that strips paths as Databricks does."""
    from fsspec_databricks import DatabricksFileSystem

    class DatabricksShapedMemory(MemoryFileSystem):
        protocol = ("dbfs",)
        cachable = False

        @classmethod
        def _strip_protocol(cls, path):
            return DatabricksFileSystem._strip_protocol(path)

        def ls(self, path, detail=True, **kwargs):
            # The Databricks client reports `modified`; the memory one only `created`.
            entries = super().ls(path, detail=detail, **kwargs)
            if detail:
                for entry in entries:
                    entry.setdefault("modified", entry["created"])
            return entries

        def info(self, path, **kwargs):
            entry = super().info(path, **kwargs)
            entry.setdefault("modified", entry.get("created"))
            return entry

    fs = DatabricksShapedMemory()
    fs.pipe("/Volumes/c/s/v/data.csv", b"id,name\n1,alpha\n2,bravo\n")

    with patch.object(
        DatabricksSource, "fs_class", property(lambda self: lambda **kwargs: fs)
    ):
        resource = DatabricksSource().dlt_source("dbfs:/Volumes/c/s/v/data.csv", "")
        rows = list(resource)

    assert [(row["id"], row["name"]) for row in rows] == [(1, "alpha"), (2, "bravo")]


@pytest.mark.parametrize(
    ("uri", "table"),
    [
        ("oci://bucket@ns/prefix/x.parquet", ""),
        ("oci://bucket@ns?iam_type=api_key", "/prefix/x.parquet"),
    ],
)
def test_oci_keeps_the_bucket_in_the_listing_path(uri, table):
    """`bucket@namespace` is the bucket; reducing it to the host lists the namespace."""
    from ocifs import OCIFileSystem

    ref = build_reference(OCISource, uri, table)

    assert ref.bucket_url == "oci://bucket@ns"
    listing = posixpath.join(
        OCIFileSystem._strip_protocol(ref.bucket_url), ref.file_glob
    )
    assert listing == "bucket@ns/prefix/x.parquet"
