"""Fixtures for the local HTTP servers the HTTP source is tested against.

One server per behaviour is started once per session and shared, because binding
a socket and generating the document root is the expensive part. The request log
is per test: each function-scoped fixture clears it before handing the server
over, so an assertion about what reached the wire cannot read another test's
requests.
"""

from pathlib import Path
from typing import Iterator

import pytest

from tests.dlt_filesystem.http_server import (
    AUTH_PASSWORD,
    AUTH_USERNAME,
    HttpFixture,
    ServerMode,
    build_document_root,
    self_signed_certificate,
    serve,
)


@pytest.fixture(scope="session")
def http_document_root(tmp_path_factory) -> Path:
    return build_document_root(tmp_path_factory.mktemp("http-root"))


@pytest.fixture(scope="session")
def http_certificate(tmp_path_factory) -> tuple[Path, Path]:
    return self_signed_certificate(tmp_path_factory.mktemp("http-tls"))


@pytest.fixture(scope="session")
def _range_server(http_document_root) -> Iterator[HttpFixture]:
    with serve(http_document_root, ServerMode.RANGE) as fixture:
        yield fixture


@pytest.fixture(scope="session")
def _no_range_server(http_document_root) -> Iterator[HttpFixture]:
    with serve(http_document_root, ServerMode.NO_RANGE) as fixture:
        yield fixture


@pytest.fixture(scope="session")
def _chunked_server(http_document_root) -> Iterator[HttpFixture]:
    with serve(http_document_root, ServerMode.CHUNKED) as fixture:
        yield fixture


@pytest.fixture(scope="session")
def _auth_server(http_document_root) -> Iterator[HttpFixture]:
    with serve(
        http_document_root, ServerMode.RANGE, auth=(AUTH_USERNAME, AUTH_PASSWORD)
    ) as fixture:
        yield fixture


@pytest.fixture(scope="session")
def _tls_server(http_document_root, http_certificate) -> Iterator[HttpFixture]:
    with serve(
        http_document_root, ServerMode.RANGE, certificate=http_certificate
    ) as fixture:
        yield fixture


def _fresh(fixture: HttpFixture) -> HttpFixture:
    fixture.clear()
    return fixture


@pytest.fixture
def range_server(_range_server) -> HttpFixture:
    """A server that honours `Range` with `206`, as a CDN or object store does."""
    return _fresh(_range_server)


@pytest.fixture
def no_range_server(_no_range_server) -> HttpFixture:
    """A server that ignores `Range` and always answers with the whole body."""
    return _fresh(_no_range_server)


@pytest.fixture
def chunked_server(_chunked_server) -> HttpFixture:
    """A server that answers chunked, so no `Content-Length` is ever reported."""
    return _fresh(_chunked_server)


@pytest.fixture
def auth_server(_auth_server) -> HttpFixture:
    """A `Range`-serving server that demands basic authentication."""
    return _fresh(_auth_server)


@pytest.fixture
def tls_server(_tls_server) -> HttpFixture:
    """A `Range`-serving server behind TLS with a self-signed certificate."""
    return _fresh(_tls_server)
