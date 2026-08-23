"""The fixture servers behave the way the HTTP source tests assume they do.

Every claim the HTTP matrix rests on is a claim about these servers: that one
honours `Range` with `206`, that another ignores it, that a third reports no size
at all, and that the request log records what actually arrived. If a fixture
quietly stopped modelling its behaviour, the tests built on it would keep passing
while proving nothing, so the behaviours are pinned here at the protocol level.

Asserted with `urllib.request` rather than fsspec: the point is what the server
sends, independent of the client the source happens to use.
"""

import ssl
import urllib.error
import urllib.request

import pytest

from tests.dlt_filesystem.http_server import (
    AUTH_PASSWORD,
    AUTH_USERNAME,
    HttpFixture,
)

DOCUMENT = "people.csv"


def fetch(url: str, *, headers: dict | None = None, context=None):
    """Return `(status, headers, body)` for one request, without a client library."""
    request = urllib.request.Request(url, headers=headers or {})  # noqa: S310 - fixture URL
    with urllib.request.urlopen(request, context=context, timeout=30) as response:  # noqa: S310
        return response.status, response.headers, response.read()


def test_range_server_reports_a_size(range_server: HttpFixture):
    status, headers, _ = fetch(range_server.url(DOCUMENT))
    body = range_server.body(DOCUMENT)

    assert status == 200
    assert headers["Content-Length"] == str(len(body))
    assert headers["Accept-Ranges"] == "bytes"
    assert headers["Last-Modified"]


def test_range_server_answers_a_range_with_partial_content(range_server: HttpFixture):
    """The 206 path: the requested bytes only, and the total in `Content-Range`."""
    body = range_server.body(DOCUMENT)

    status, headers, served = fetch(
        range_server.url(DOCUMENT), headers={"Range": "bytes=2-5"}
    )

    assert status == 206
    assert served == body[2:6]
    assert headers["Content-Range"] == f"bytes 2-5/{len(body)}"
    assert headers["Content-Length"] == "4"


def test_range_server_answers_an_open_ended_range(range_server: HttpFixture):
    body = range_server.body(DOCUMENT)

    status, headers, served = fetch(
        range_server.url(DOCUMENT), headers={"Range": "bytes=4-"}
    )

    assert status == 206
    assert served == body[4:]
    assert headers["Content-Range"] == f"bytes 4-{len(body) - 1}/{len(body)}"


def test_range_server_answers_a_suffix_range(range_server: HttpFixture):
    """Nothing under test sends this form, but a server that broke on it would lie."""
    body = range_server.body(DOCUMENT)

    status, _, served = fetch(range_server.url(DOCUMENT), headers={"Range": "bytes=-6"})

    assert status == 206
    assert served == body[-6:]


def test_range_server_refuses_an_unsatisfiable_range(range_server: HttpFixture):
    """A `416` is load-bearing: fsspec reads it as "past the end", not as an error."""
    body = range_server.body(DOCUMENT)

    with pytest.raises(urllib.error.HTTPError) as exception:
        fetch(range_server.url(DOCUMENT), headers={"Range": f"bytes={len(body)}-"})

    assert exception.value.code == 416
    assert exception.value.headers["Content-Range"] == f"bytes */{len(body)}"


@pytest.mark.parametrize("header", ["bytes=-", "bytes=abc-1", "rows=1-2", "bytes=9-4"])
def test_range_server_handles_a_range_it_cannot_use(range_server: HttpFixture, header):
    """An unparseable range is ignored, per RFC 9110; a reversed one gets a 416.

    Neither may take the handler thread down, which is what an uncaught parse
    error would do, leaving the client with a connection reset.
    """
    try:
        status, _, served = fetch(range_server.url(DOCUMENT), headers={"Range": header})
    except urllib.error.HTTPError as error:
        assert error.code == 416
        return

    assert status == 200
    assert served == range_server.body(DOCUMENT)


def test_no_range_server_ignores_a_range(no_range_server: HttpFixture):
    """The behaviour fsspec raises on: a ranged request answered with everything."""
    body = no_range_server.body(DOCUMENT)

    status, headers, served = fetch(
        no_range_server.url(DOCUMENT), headers={"Range": "bytes=2-5"}
    )

    assert status == 200
    assert served == body
    assert "Accept-Ranges" not in headers
    assert "Content-Range" not in headers


def test_chunked_server_reports_no_length(chunked_server: HttpFixture):
    """Neither HEAD nor GET carries a `Content-Length`, so no size is knowable."""
    url = chunked_server.url(DOCUMENT)

    head = urllib.request.Request(url, method="HEAD")  # noqa: S310 - fixture URL
    with urllib.request.urlopen(head, timeout=30) as response:  # noqa: S310
        assert response.headers["Transfer-Encoding"] == "chunked"
        assert response.headers.get("Content-Length") is None
        assert response.headers["Last-Modified"]

    status, headers, served = fetch(url)

    assert status == 200
    assert headers.get("Content-Length") is None
    assert served == chunked_server.body(DOCUMENT)


def test_chunked_server_looks_sizeless_to_the_client(chunked_server: HttpFixture):
    """What the reader stack actually sees: `size` is `None`, not zero or missing."""
    from fsspec.implementations.http import HTTPFileSystem

    info = HTTPFileSystem().info(chunked_server.url(DOCUMENT))

    assert info["size"] is None
    assert info["type"] == "file"


def test_auth_server_refuses_a_request_without_credentials(auth_server: HttpFixture):
    with pytest.raises(urllib.error.HTTPError) as exception:
        fetch(auth_server.url(DOCUMENT))

    assert exception.value.code == 401
    assert exception.value.headers["WWW-Authenticate"].startswith("Basic ")


def test_index_server_renders_relative_links(index_server: HttpFixture):
    status, headers, served = fetch(f"{index_server.base_url}/")

    assert status == 200
    assert headers.get_content_type() == "text/html"
    assert b'<a href="alpha.csv">alpha.csv</a>' in served
    assert b'<a href="sub/">sub/</a>' in served


def test_headerless_server_omits_last_modified(
    no_last_modified_server: HttpFixture,
):
    status, headers, _ = fetch(no_last_modified_server.url(DOCUMENT))

    assert status == 200
    assert headers.get("Last-Modified") is None


def test_auth_server_accepts_the_decoded_credentials(auth_server: HttpFixture):
    """The credentials that must arrive decoded, not as the percent-encoded form."""
    import base64

    token = base64.b64encode(f"{AUTH_USERNAME}:{AUTH_PASSWORD}".encode()).decode()

    status, _, served = fetch(
        auth_server.url(DOCUMENT), headers={"Authorization": f"Basic {token}"}
    )

    assert status == 200
    assert served == auth_server.body(DOCUMENT)


def test_tls_server_serves_over_https(tls_server: HttpFixture, http_certificate):
    """The certificate is verified, not skipped, so `https://` is really exercised."""
    context = ssl.create_default_context(cafile=str(http_certificate[0]))

    status, _, served = fetch(tls_server.url(DOCUMENT), context=context)

    assert tls_server.url(DOCUMENT).startswith("https://")
    assert status == 200
    assert served == tls_server.body(DOCUMENT)


def test_request_log_records_the_query_exactly_as_it_arrived(range_server: HttpFixture):
    """The log is the only witness to signature preservation, so it must not decode."""
    query = "X-Amz-Signature=abc%2Fdef&plus=a+b"

    fetch(range_server.url(DOCUMENT, query=query))

    assert [request.query for request in range_server.requests] == [query]
    assert range_server.requests[0].path == f"/{DOCUMENT}"


def test_request_log_separates_ranged_requests_from_probes(range_server: HttpFixture):
    """`ranged()` is what lets a test ignore the unranged listing probe."""
    fetch(range_server.url(DOCUMENT))
    fetch(range_server.url(DOCUMENT), headers={"Range": "bytes=0-3"})

    assert len(range_server.requests) == 2
    assert [request.range_header for request in range_server.ranged()] == ["bytes=0-3"]
    assert range_server.bytes_served(ranged_only=True) == 4
    assert range_server.bytes_served() == len(range_server.body(DOCUMENT)) + 4


def test_log_is_cleared_between_tests(range_server: HttpFixture):
    """Each test gets a fresh log, so one test cannot read another's requests."""
    assert range_server.requests == []
