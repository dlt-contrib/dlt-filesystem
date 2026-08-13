"""The contract omniload offers for an `http://` or `https://` source URL.

Driven through the registered scheme (`run_ingest` and `SourceDestinationFactory`)
rather than a named class, so the matrix reads the same before and after the
connector behind that scheme is replaced. What it pins is what reaches the wire
and what lands in the destination, not one implementation of it.

Three server behaviours are covered because they are three different code paths,
and each is a way the previous connector's whole-body download quietly worked:
a server that honours `Range`, one that ignores it, and one that answers chunked
so no size is reported at all. See `http_server.py`.

Every server is a local `http.server` thread: no Docker, no credentials, no
network.
"""

import duckdb
import pytest

from omniload import run_ingest
from tests.dlt_filesystem.http_server import (
    AUTH_PASSWORD_ENCODED,
    AUTH_USERNAME,
    EVENT_COUNT,
    HttpFixture,
)

#: The rows every document in the fixture root carries, in query order.
EXPECTED = [("Alice", 30), ("Bob", 41), ("Charlie", 25)]

#: A presigned-URL shape. `%2F` must survive to the wire byte for byte, because a
#: signature is computed over the encoded form; `%7E` normalizes to `~`, which is
#: what `requests` did and what every SigV4 canonicalization treats as equal.
SIGNED_QUERY = "X-Amz-Signature=abc%2Fdef%7Eghi&X-Amz-Expires=900"
SIGNED_QUERY_ON_THE_WIRE = "X-Amz-Signature=abc%2Fdef~ghi&X-Amz-Expires=900"


def load(
    server: HttpFixture,
    document: str,
    tmp_path,
    *,
    table: str = "",
    query: str = "",
    fragment: str = "",
    **options,
):
    """Ingest one document from the fixture server into a fresh duckdb file."""
    destination = tmp_path / "warehouse.duckdb"
    run_ingest(
        source_uri=server.url(document, query=query, fragment=fragment),
        dest_uri=f"duckdb:///{destination}",
        source_table=table,
        dest_table="out.people",
        progress="log",
        **options,
    )
    return destination


def rows(
    destination, statement: str = "select name, age from out.people order by name"
):
    connection = duckdb.connect(str(destination))
    try:
        return connection.sql(statement).fetchall()
    finally:
        connection.close()


DOCUMENTS = [
    pytest.param("people.csv", "", id="csv"),
    pytest.param("people.jsonl", "", id="jsonl"),
    pytest.param("people.json", "", id="json"),
    pytest.param("people.parquet", "", id="parquet"),
    pytest.param(
        "people-no-header.csv", "people-no-header.csv#csv_headless", id="csv_headless"
    ),
    pytest.param("feeds/events.csv", "", id="nested-path"),
]


@pytest.mark.parametrize(("document", "table"), DOCUMENTS)
def test_document_formats_load(range_server, tmp_path, document, table):
    """Every format the connector claims, read over plaintext `http://`.

    This is also the plaintext-`http` regression test: it proves the scheme by
    reading rows, not by listing files. dlt composes a file's URL through
    `dlt.common.storages.configuration.MAKE_URI_DISPATCH`, which registers
    `https` and not `http`, and the URL it composes for `http` is wrong in a way
    listing cannot see: it only fails when a reader opens the file.
    """
    destination = load(
        range_server,
        document,
        tmp_path,
        table=table,
        columns=["name:text,age:bigint"],
    )

    assert rows(destination) == EXPECTED


def test_signed_url_query_reaches_the_wire(range_server, tmp_path):
    """A signed URL loads, and its signature arrives at the server intact.

    Asserted from the fixture's own request log rather than from the row count:
    rows would also come back from a server that ignores the query, so only the
    recorded request can show the signature was neither dropped nor rewritten.
    """
    destination = load(range_server, "people.csv", tmp_path, query=SIGNED_QUERY)

    assert rows(destination) == EXPECTED
    assert range_server.queries(), "no request reached the server"
    assert set(range_server.queries()) == {SIGNED_QUERY_ON_THE_WIRE}


def test_server_that_ignores_range_loads_csv(no_range_server, tmp_path):
    """A server that answers every request with the whole body still loads."""
    destination = load(no_range_server, "people.csv", tmp_path)

    assert rows(destination) == EXPECTED


def test_server_that_ignores_range_loads_parquet(no_range_server, tmp_path):
    """The worst case for a range-less server: a reader that seeks.

    pyarrow reads a parquet footer before anything else, so this fails on a
    plain fsspec range read (fsspec raises rather than falling back) where the
    previous connector's whole-body download did not.
    """
    destination = load(no_range_server, "people.parquet", tmp_path)

    assert rows(destination) == EXPECTED


def test_chunked_response_without_content_length_loads_csv(chunked_server, tmp_path):
    """A chunked response reports no size, which must not be fatal.

    This is ordinary `Transfer-Encoding: chunked`, not an exotic case: the file
    is concrete and named, and the client simply cannot say how long it is.
    """
    destination = load(chunked_server, "people.csv", tmp_path)

    assert rows(destination) == EXPECTED


def test_chunked_response_without_content_length_loads_parquet(
    chunked_server, tmp_path
):
    """Unknown size plus a seeking reader: no size means no seekable file."""
    destination = load(chunked_server, "people.parquet", tmp_path)

    assert rows(destination) == EXPECTED


def test_percent_encoded_password_authenticates(auth_server, tmp_path):
    """Userinfo is percent-decoded before it is sent, as `requests` decoded it.

    The password here needs encoding to survive a URI at all (`@` would end the
    userinfo, `/` the netloc), so a connector that forwards the raw substring
    authenticates as the wrong user.
    """
    url = auth_server.url("people.csv")
    scheme, _, remainder = url.partition("://")
    credentialed = f"{scheme}://{AUTH_USERNAME}:{AUTH_PASSWORD_ENCODED}@{remainder}"

    destination = tmp_path / "warehouse.duckdb"
    run_ingest(
        source_uri=credentialed,
        dest_uri=f"duckdb:///{destination}",
        source_table="",
        dest_table="out.people",
        progress="log",
    )

    assert rows(destination) == EXPECTED
    assert [request.status for request in auth_server.requests].count(401) <= 2


def test_reader_pulls_the_whole_body_in_one_unranged_request(range_server, tmp_path):
    """How much of a document a read costs, on a server that can serve ranges.

    Baseline, and the thing the migration changes: the connector behind `http://`
    fetches the entire body with a single unranged `GET` and buffers it, so a
    range-serving server buys nothing and a large document is read in full even
    when the reader only needs part of it. Pinned rather than described, because
    "it streams now" is otherwise unfalsifiable: rows come back either way.
    """
    body = range_server.body("events.parquet")

    destination = tmp_path / "warehouse.duckdb"
    run_ingest(
        source_uri=range_server.url("events.parquet"),
        dest_uri=f"duckdb:///{destination}",
        source_table="",
        dest_table="out.events",
        progress="log",
    )

    assert rows(destination, "select count(*) from out.events") == [(EVENT_COUNT,)]
    assert range_server.ranged() == []
    assert range_server.bytes_served() >= len(body)


def test_missing_document_fails_loudly(range_server, tmp_path):
    """A URL that names nothing must raise, not load nothing and call it a success.

    Kept apart from the redaction case below on purpose: folding the two together
    lets a regression to a silent empty load satisfy that test's `xfail`, which is
    the louder of the two failures.
    """
    with pytest.raises(Exception) as exception:  # noqa: PT011 - tightened in the swap
        load(range_server, "absent.csv", tmp_path, query=SIGNED_QUERY)

    assert 404 in [request.status for request in range_server.requests]
    assert str(exception.value)


@pytest.mark.xfail(
    reason="the connector behind http:// reports a 404 by interpolating the whole "
    "requested URL, signature included; moving the failure into the family's own "
    "error is what redacts it",
    strict=True,
)
def test_missing_document_does_not_leak_the_query(range_server, tmp_path):
    """The message for an absent document must not carry the signature.

    Deliberately does not require an exception: whether one is raised at all is
    the previous test's contract, so this one reads whatever message came back
    (none, if nothing raised) and only judges what is in it.
    """
    try:
        load(range_server, "absent.csv", tmp_path, query=SIGNED_QUERY)
    except Exception as error:  # noqa: BLE001 - the message is the subject
        message = str(error)
    else:
        message = ""

    assert "X-Amz-Signature" not in message
    assert "abc%2Fdef" not in message
