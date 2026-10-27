"""Dropbox paths are validated by the SDK without contacting the service."""

import io
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from dropbox import files
from dropboxdrivefs import DropboxDriveFileSystem

from dlt_filesystem.source.fsspec.dropbox import DropboxSource


@pytest.fixture
def dropbox_client():
    client = Mock(_oauth2_access_token="test-token")  # noqa: S106
    metadata = files.FileMetadata(
        name="data.csv",
        id="id:test-file",
        client_modified=datetime(2026, 1, 1),
        server_modified=datetime(2026, 1, 1),
        rev="0123456789",
        size=4,
        path_display="/path/to/data.csv",
        path_lower="/path/to/data.csv",
    )

    def get_metadata(path):
        files.GetMetadataArg(path=path)
        return metadata

    def list_folder(path, **kwargs):
        files.ListFolderArg(path=path, **kwargs)
        return SimpleNamespace(entries=[metadata], has_more=False)

    def temporary_link(path):
        files.GetTemporaryLinkArg(path=path)
        return SimpleNamespace(link="https://example.test/download")

    client.files_get_metadata.side_effect = get_metadata
    client.files_list_folder.side_effect = list_folder
    client.files_get_temporary_link.side_effect = temporary_link
    return client


@pytest.mark.parametrize(
    ("uri", "table"),
    [
        ("dropbox://path/to/data.csv", ""),
        ("dropbox://", "path/to/data.csv"),
        ("dropbox://", "dropbox://path/to/data.csv"),
    ],
)
def test_dropbox_discovery_and_open_use_absolute_paths(dropbox_client, uri, table):
    with patch("dlt_filesystem.source.core.resource_for_reader", lambda ref: ref):
        ref = DropboxSource().dlt_source(uri, table, client=dropbox_client)

    assert isinstance(ref.fs, DropboxDriveFileSystem)
    root = ref.fs._strip_protocol(ref.bucket_url)
    listing = ref.fs.glob(f"{root}/{ref.file_glob}", detail=True)
    assert list(listing) == ["/path/to/data.csv"]
    assert listing["/path/to/data.csv"]["size"] == 4
    assert dropbox_client.files_get_metadata.call_args.args == ("/path/to/data.csv",)

    with patch(
        "dropboxdrivefs.core.webhdfs.WebHDFile", return_value=io.BytesIO(b"x\n1\n")
    ):
        with ref.fs.open("dropbox://path/to/data.csv", "rb") as stream:
            assert stream.read() == b"x\n1\n"
    dropbox_client.files_get_temporary_link.assert_called_once_with("/path/to/data.csv")


@pytest.mark.parametrize("path", ["dropbox://path/to", "path/to", "/path/to"])
def test_dropbox_info_and_directory_listing_strip_the_protocol(dropbox_client, path):
    fs = DropboxSource().fs_class(client=dropbox_client, skip_instance_cache=True)

    assert fs.info(path)["name"] == "/path/to/data.csv"
    assert fs.ls(path)[0]["name"] == "/path/to/data.csv"
    dropbox_client.files_get_metadata.assert_called_once_with("/path/to")
    assert dropbox_client.files_list_folder.call_args.args == ("/path/to",)


@pytest.mark.parametrize("path", ["dropbox://", "/", ""])
def test_dropbox_root_listing_uses_the_sdk_root(dropbox_client, path):
    fs = DropboxSource().fs_class(client=dropbox_client, skip_instance_cache=True)

    assert fs.ls(path, detail=False) == ["/path/to/data.csv"]
    assert dropbox_client.files_list_folder.call_args.args == ("",)
