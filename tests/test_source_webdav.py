"""Exercise the real WebDAV filesystem with an offline HTTP transport."""

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
@pytest.mark.parametrize("wildcard", [False, True])
@pytest.mark.parametrize(
    "filename",
    ["x.csv", "report#1.csv", "report?1.csv", "report%23.csv", "space here.csv"],
)
def test_webdav_requests_the_selection_once(scheme, split, wildcard, filename):
    requests = []
    content = b"x\n1\n"
    encoded_path = "/p/" + quote(filename, safe="")
    selection = "p/*.csv" if wildcard else encoded_path.lstrip("/")

    def respond(request):
        requests.append(request)
        if request.method == "PROPFIND":
            body = """<?xml version="1.0"?>
<D:multistatus xmlns:D="DAV:">
  <D:response>
    <D:href>/p/x.csv</D:href>
    <D:propstat>
      <D:prop>
        <D:resourcetype/>
        <D:getcontentlength>4</D:getcontentlength>
        <D:getlastmodified>Wed, 01 Jan 2025 00:00:00 GMT</D:getlastmodified>
      </D:prop>
      <D:status>HTTP/1.1 200 OK</D:status>
    </D:propstat>
  </D:response>
</D:multistatus>"""
            body = body.replace("/p/x.csv", encoded_path)
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
    items = list(glob_files(ref.fs, ref.bucket_url, ref.file_glob))
    assert len(items) == 1
    assert items[0]["relative_path"] == encoded_path.lstrip("/")
    with ref.fs.open(items[0]["file_url"], "rb", block_size=0) as stream:
        assert stream.read() == content

    assert ref.fs.ls("p", detail=False) == [encoded_path]
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
