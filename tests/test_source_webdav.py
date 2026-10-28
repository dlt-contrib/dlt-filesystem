"""Exercise the real WebDAV filesystem with an offline HTTP transport."""

from datetime import datetime, timezone
from unittest.mock import patch
from urllib.parse import quote

import httpx
import pytest

from dlt_filesystem.source.fsspec.webdav import WebdavSource
from dlt_filesystem.source.lister import glob_files


@pytest.mark.parametrize(
    "scheme", ["http+webdav", "https+webdav", "http+dav", "https+dav"]
)
@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("known_size", [False, True])
@pytest.mark.parametrize("wildcard", ["none", "all", "prefix", "directories"])
@pytest.mark.parametrize(
    "filename",
    [
        "x.csv",
        "report#1.csv",
        "report?1.csv",
        "report%23.csv",
        "report%2a.csv",
        "report%3f.csv",
        "report%5b.csv",
        "report%5d.csv",
        "space here.csv",
        "x+1.csv",
        "café.csv",
        "report[1].csv",
    ],
)
def test_webdav_requests_the_selection_once(
    scheme, split, wildcard, filename, known_size
):
    requests = []
    content = b"x\n1\n"
    encoded_path = "/p/" + quote(filename, safe="")
    if wildcard == "directories":
        selection = "p/*/"
    elif wildcard == "all":
        selection = "p/*.csv"
    elif wildcard == "prefix":
        prefix = filename[:-4]
        for char in "%#?[]":
            prefix = prefix.replace(char, quote(char, safe=""))
        selection = "p/" + prefix + "*.csv"
    else:
        selection = encoded_path.lstrip("/")

    def respond(request):
        requests.append(request)
        assert request.url.raw_path.decode() in {"/p", "/p/", encoded_path}
        if request.method == "PROPFIND":
            body = """<?xml version="1.0"?>
<D:multistatus xmlns:D="DAV:">
  <D:response>
    <D:href>/p/x.csv</D:href>
    <D:propstat>
      <D:prop>
        <D:resourcetype/>
        <D:getetag>test-etag</D:getetag>
        <D:creationdate>2024-01-01T00:00:00Z</D:creationdate>
        <D:getlastmodified>Wed, 01 Jan 2025 00:00:00 GMT</D:getlastmodified>
      </D:prop>
      <D:status>HTTP/1.1 200 OK</D:status>
    </D:propstat>
  </D:response>
</D:multistatus>"""
            body = body.replace("/p/x.csv", encoded_path)
            if known_size:
                body = body.replace(
                    "<D:resourcetype/>",
                    "<D:resourcetype/><D:getcontentlength>4</D:getcontentlength>",
                )
            if request.url.path.rstrip("/") == "/p":
                directory = """<D:response><D:href>/p/</D:href><D:propstat>
<D:prop><D:resourcetype><D:collection/></D:resourcetype></D:prop>
<D:status>HTTP/1.1 200 OK</D:status></D:propstat></D:response>"""
                body = body.replace("</D:multistatus>", directory + "</D:multistatus>")
            return httpx.Response(207, text=body)
        if request.url.raw_path.decode() == encoded_path:
            return httpx.Response(200, content=content)
        # A truncated path can be a different valid file, not a missing one.
        return httpx.Response(200, content=b"x\n999\n")

    origin = f"{scheme}://user:pass@example.test:8443"
    uri = origin if split else origin + "/" + selection
    # Query and fragment are connector and reader options, not part of base_url.
    uri += "?verify=false#csv"
    table = selection if split else ""
    with patch("dlt_filesystem.source.core.resource_for_reader", lambda ref: ref):
        ref = WebdavSource().dlt_source(
            uri, table, transport=httpx.MockTransport(respond)
        )

    assert ref.reader_name == "read_csv"
    assert ref.fs.client.base_url.path == "/"
    assert not ref.fs.client.base_url.query
    assert not ref.fs.client.base_url.fragment
    items = list(
        glob_files(ref.fs, ref.bucket_url, ref.file_glob, fetch_file_info=True)
    )
    if wildcard == "directories":
        assert items == []
        assert ref.fs.glob("/p/*/") == []
        return
    assert len(items) == 1
    assert ref.fs.info(encoded_path)["name"] == encoded_path
    assert ref.fs.modified(encoded_path) == datetime(2025, 1, 1, tzinfo=timezone.utc)
    assert ref.fs.created(encoded_path) == datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert ref.fs.size(encoded_path) == (4 if known_size else None)
    assert ref.fs.checksum(encoded_path) == "test-etag"
    assert items[0]["relative_path"] == encoded_path.lstrip("/")
    with ref.fs.open(items[0]["file_url"], "rb", block_size=0) as stream:
        assert stream.read() == content

    assert ref.fs.ls("p", detail=False) == [encoded_path]
    listed = ref.fs.ls("p", detail=True)[0]
    assert "size" in listed
    assert listed["size"] == (4 if known_size else None)
    # Exercise a single-character glob directly: '?' on a source URI is a query.
    one_character = "/p/" + quote(filename[:-5], safe="") + "?.csv"
    assert encoded_path in ref.fs.glob(one_character)
    assert {request.method for request in requests} == {"PROPFIND", "GET"}
    for request in requests:
        if request.method == "GET":
            assert request.url.path == "/p/" + filename
        else:
            assert request.url.path.rstrip("/") in {"/p", "/p/" + filename}
        assert request.url.host == "example.test"
        assert request.url.port == 8443
        assert request.url.scheme == scheme.split("+")[0]
        assert not request.url.query
        assert request.headers["Authorization"] == "Basic dXNlcjpwYXNz"


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize(
    ("selection", "expected"),
    [
        ("p/report%2A1.csv", {"/p/report%2A1.csv"}),
        ("p/report%3F1.csv", {"/p/report%3F1.csv"}),
        (
            "p/report*1.csv",
            {"/p/report%2A1.csv", "/p/report%3F1.csv", "/p/reportX1.csv"},
        ),
    ],
)
def test_webdav_escaped_glob_characters_are_literals(split, selection, expected):
    filenames = ["report*1.csv", "report?1.csv", "reportX1.csv"]
    directory = """<D:response><D:href>/p/</D:href><D:propstat>
<D:prop><D:resourcetype><D:collection/></D:resourcetype></D:prop>
<D:status>HTTP/1.1 200 OK</D:status></D:propstat></D:response>"""

    def respond(request):
        entries = [directory]
        for filename in filenames:
            href = quote("/p/" + filename, safe="/")
            entries.append(f"""<D:response><D:href>{href}</D:href><D:propstat>
<D:prop><D:resourcetype/><D:getcontentlength>4</D:getcontentlength>
<D:getlastmodified>Wed, 01 Jan 2025 00:00:00 GMT</D:getlastmodified></D:prop>
<D:status>HTTP/1.1 200 OK</D:status></D:propstat></D:response>""")
        body = '<D:multistatus xmlns:D="DAV:">' + "".join(entries) + "</D:multistatus>"
        return httpx.Response(207, text=body)

    origin = "https+webdav://example.test"
    uri = origin if split else origin + "/" + selection
    with patch("dlt_filesystem.source.core.resource_for_reader", lambda ref: ref):
        ref = WebdavSource().dlt_source(
            uri + "#csv",
            selection if split else "",
            transport=httpx.MockTransport(respond),
        )
    items = list(glob_files(ref.fs, ref.bucket_url, ref.file_glob))
    assert {"/" + item["relative_path"] for item in items} == expected
    assert set(ref.fs.glob("/p/report?1.csv")) == {
        quote("/p/" + name, safe="/") for name in filenames
    }


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("directory", ["trailing ", "trailing\t", "trailing\n"])
def test_webdav_directory_whitespace_is_preserved(split, directory):
    encoded_directory = "/p/" + quote(directory, safe="")
    encoded_file = encoded_directory + "/x.csv"

    def respond(request):
        path = request.url.raw_path.decode()
        if path.rstrip("/") != encoded_directory:
            return httpx.Response(404)
        body = f"""<D:multistatus xmlns:D="DAV:">
<D:response><D:href>{encoded_directory}/</D:href><D:propstat>
<D:prop><D:resourcetype><D:collection/></D:resourcetype></D:prop>
<D:status>HTTP/1.1 200 OK</D:status></D:propstat></D:response>
<D:response><D:href>{encoded_file}</D:href><D:propstat>
<D:prop><D:resourcetype/><D:getcontentlength>4</D:getcontentlength>
<D:getlastmodified>Wed, 01 Jan 2025 00:00:00 GMT</D:getlastmodified></D:prop>
<D:status>HTTP/1.1 200 OK</D:status></D:propstat></D:response>
</D:multistatus>"""
        return httpx.Response(207, text=body)

    origin = "https+webdav://example.test"
    selection = encoded_directory.lstrip("/") + "/*.csv"
    uri = origin if split else origin + "/" + selection
    with patch("dlt_filesystem.source.core.resource_for_reader", lambda ref: ref):
        ref = WebdavSource().dlt_source(
            uri + "#csv",
            selection if split else "",
            transport=httpx.MockTransport(respond),
        )
    items = list(glob_files(ref.fs, ref.bucket_url, ref.file_glob))
    assert ["/" + item["relative_path"] for item in items] == [encoded_file]
    assert items[0]["size_in_bytes"] == 4
    assert ref.fs.ls(encoded_directory, detail=False) == [encoded_file]
    with pytest.raises(FileNotFoundError):
        ref.fs.ls("/p/missing")
