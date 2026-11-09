"""WebHDFS discovery and reads against an in-process fake of the namenode's REST API."""

from unittest.mock import patch
from urllib.parse import urlsplit

import pytest

from dlt_filesystem.source.fsspec.webhdfs import WebHDFSFileSystem, WebHDFSSource

PREFIX = "/webhdfs/v1"
FILES = {"/path/to/data.csv": b"x\n1\n"}
DIRECTORIES = {"/", "/path", "/path/to"}


class FakeResponse:
    def __init__(self, status_code, body=None, content=b""):
        self.status_code = status_code
        self._body = body
        self.content = content
        self.headers = {}

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _status(suffix, kind, length=0):
    return {
        "pathSuffix": suffix,
        "type": kind,
        "length": length,
        "modificationTime": 1_767_225_600_000,
    }


def _not_found(path):
    return FakeResponse(
        404,
        {
            "RemoteException": {
                "exception": "FileNotFoundException",
                "message": f"File does not exist: {path}",
            }
        },
    )


@pytest.fixture
def namenode():
    """Answer WebHDFS operations the way a namenode does, recording each path."""
    paths = []

    def request(session, method, url, params: dict, **kwargs):
        url_path = urlsplit(url).path
        paths.append(url_path)
        # The namenode reads everything after `/webhdfs/v1` as the HDFS path, and
        # an HDFS path is absolute.
        path = url_path.removeprefix(PREFIX)
        if not path.startswith("/"):
            return _not_found(path)
        path = path.rstrip("/") or "/"
        op = params["op"]
        if op == "GETFILESTATUS":
            if path in FILES:
                return FakeResponse(
                    200, {"FileStatus": _status("", "FILE", len(FILES[path]))}
                )
            if path in DIRECTORIES:
                return FakeResponse(200, {"FileStatus": _status("", "DIRECTORY")})
            return _not_found(path)
        if op == "LISTSTATUS":
            parent = "" if path == "/" else path
            children = [
                _status(name.rsplit("/", 1)[1], "FILE", len(data))
                for name, data in FILES.items()
                if name.rsplit("/", 1)[0] == parent
            ] + [
                _status(name.rsplit("/", 1)[1], "DIRECTORY")
                for name in DIRECTORIES
                if name != "/" and name.rsplit("/", 1)[0] == parent
            ]
            return FakeResponse(200, {"FileStatuses": {"FileStatus": children}})
        if op == "OPEN":
            start = int(params["offset"])
            return FakeResponse(
                200, content=FILES[path][start : start + int(params["length"])]
            )
        raise AssertionError(f"unexpected WebHDFS operation {op}")

    with patch("requests.Session.request", request):
        yield paths


@pytest.mark.parametrize("selection", ["path/to/data.csv", "path/to/*.csv"])
@pytest.mark.parametrize("split", [False, True], ids=["uri-form", "split-form"])
def test_webhdfs_discovers_and_reads_from_the_namespace_root(
    namenode, split, selection
):
    """A host-only URI used to reach the namenode as `/webhdfs/v1path/to/...`."""
    uri = "webhdfs://host:9870" + ("" if split else "/" + selection)
    table = selection if split else ""

    resource = WebHDFSSource().dlt_source(uri, table, skip_instance_cache=True)

    assert list(resource) == [{"x": 1}]
    assert namenode
    assert all(path.startswith(PREFIX + "/") for path in namenode), namenode


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("webhdfs://host:9870", "/"),
        ("webhdfs://host:9870/", "/"),
        ("webhdfs://host:9870/path/to", "/path/to"),
        ("path/to", "/path/to"),
        ("/path/to", "/path/to"),
        ("", "/"),
    ],
)
def test_webhdfs_paths_are_rooted(path, expected):
    assert WebHDFSFileSystem._strip_protocol(path) == expected
