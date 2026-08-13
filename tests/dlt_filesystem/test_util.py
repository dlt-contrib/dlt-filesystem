import pytest

from dlt_filesystem.source.impl.util import strip_protocol_suffix
from dlt_filesystem.util.python import (
    apply_alias,
    asbool,
    cast_to_bool,
    cast_to_float,
    cast_to_list,
)


def test_asbool_success():
    """Validate the `asbool` utility function."""
    assert asbool("true") is True
    assert asbool("yes") is True
    assert asbool("false") is False
    assert asbool("no") is False


def test_asbool_failure():
    """Validate failing the `asbool` utility function."""
    with pytest.raises(ValueError) as exc_info:
        asbool("unknown")
    assert exc_info.match("Cannot cast value to bool: 'unknown'")

    with pytest.raises(ValueError) as exc_info:
        asbool("2")
    assert exc_info.match("Cannot cast value to bool: '2'")

    with pytest.raises(ValueError) as exc_info:
        asbool("")
    assert exc_info.match("Cannot cast value to bool: ''")


def test_cast_to_bool():
    """Validate the `cast_to_bool` utility function."""
    data = {"foo": "1"}
    cast_to_bool(data, ["foo"])
    assert data["foo"] is True


def test_cast_to_float():
    """Validate the `cast_to_float` utility function."""
    data = {"foo": "42.42"}
    cast_to_float(data, ["foo"])
    assert data["foo"] == 42.42


def test_cast_to_list_success():
    """Validate the `cast_to_list` utility function."""
    data = {"foo": '["bar"]'}
    cast_to_list(data, ["foo"])
    assert data["foo"] == ["bar"]


def test_cast_to_list_failure():
    """Validate failing the `cast_to_list` utility function."""
    data = {"foo": None}
    with pytest.raises(ValueError) as exc_info:
        cast_to_list(data, ["foo"])
    assert exc_info.match(
        "Cannot cast value to list: None. "
        "Error: the JSON object must be str, bytes or bytearray, not NoneType"
    )


def test_apply_alias_success():
    """Validate the `apply_alias` utility function."""
    data = {"foo": "bar"}
    apply_alias(data, "foo", "effective")
    assert data["effective"] == "bar"


def test_apply_alias_collision():
    """Validate failing the `apply_alias` utility function."""
    data = {"foo": "bar"}
    with pytest.raises(ValueError) as exc_info:
        apply_alias(data, "foo", "foo")
    assert exc_info.match("use only one")


def test_strip_protocol_suffix_success():
    """Validate the `strip_protocol_suffix` utility function."""
    assert (
        strip_protocol_suffix("http+dav://foo:1234/bar/?baz=+dav", "webdav", "dav")
        == "http://foo:1234/bar/?baz=+dav"
    )
    assert (
        strip_protocol_suffix(
            "http+webdav://foo:1234/bar/?baz=+webdav", "webdav", "dav"
        )
        == "http://foo:1234/bar/?baz=+webdav"
    )


def test_strip_protocol_suffix_no_suffix():
    """Validate the `strip_protocol_suffix` utility function."""
    assert strip_protocol_suffix("http+dav://") == "http+dav://"


#: URLs that exercise every branch of the escaping rule, including the malformed
#: ones. The expected values are what `requests.utils.requote_uri` produces, since
#: reproducing that rule byte for byte is the point of the function: an HTTP URL has
#: to keep meaning what it meant to the connector that used to send it.
REQUOTE_CASES = [
    # A meaning-carrying escape survives: `%2F` is not the same as `/` inside a
    # signature, and decoding it would void one.
    ("http://h/f.csv?sig=a%2Fb&x=1", "http://h/f.csv?sig=a%2Fb&x=1"),
    # An escape of an unreserved character is decoded, because the two forms are
    # defined to be equal.
    ("http://h/f.csv?sig=a%7Eb", "http://h/f.csv?sig=a~b"),
    # What a URL cannot carry literally gets escaped.
    ("http://h/a b.csv", "http://h/a%20b.csv"),
    ("http://h/café.csv", "http://h/caf%C3%A9.csv"),
    # Already percent-encoded UTF-8: left exactly as it is. Escaping it again would
    # ask the server for a file whose name literally contains "%C3%A9".
    ("http://h/caf%C3%A9.csv", "http://h/caf%C3%A9.csv"),
    # A `+` is left alone: in a query it already means something.
    ("http://h/f.csv?q=a+b", "http://h/f.csv?q=a+b"),
    # Userinfo is escaped-as-found, not decoded.
    ("http://user:p%40ss%2Fword@h/f.csv", "http://user:p%40ss%2Fword@h/f.csv"),
    # Malformed: `zz` is not hex, so nothing can be unescaped and the stray `%`
    # is escaped as one.
    ("http://h/f.csv?sig=%zz", "http://h/f.csv?sig=%25zz"),
    # Not an escape at all: the `%` is literal and stays literal.
    ("http://h/f.csv?x=100%", "http://h/f.csv?x=100%"),
    # Valid hex that is not an unreserved character: the escape is kept as written.
    ("http://h/f.csv?sig=%80", "http://h/f.csv?sig=%80"),
]


@pytest.mark.parametrize(("uri", "expected"), REQUOTE_CASES)
def test_requote_uri(uri, expected):
    from dlt_filesystem.util.web import requote_uri

    assert requote_uri(uri) == expected


@pytest.mark.parametrize(("uri", "expected"), REQUOTE_CASES)
def test_requote_uri_matches_requests(uri, expected):
    """The claim is parity with `requests`, so it is asserted against `requests`.

    Skips rather than fails if `requests` is ever dropped as a dependency: the
    literal expectations above still pin the behaviour on their own.
    """
    requests_utils = pytest.importorskip("requests.utils")
    from dlt_filesystem.util.web import requote_uri

    assert requote_uri(uri) == requests_utils.requote_uri(uri) == expected
