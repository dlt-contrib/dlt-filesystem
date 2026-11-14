"""The fork of `fsspec.utils.infer_storage_options` in `dlt_filesystem.util.fsspec`."""

# Every password here is a parse fixture, and the upstream tests stay verbatim.
# ruff: noqa: S105

import fsspec.utils
import pytest

from dlt_filesystem.util.fsspec import infer_storage_options


# Ported verbatim from fsspec (fsspec/filesystem_spec#2114).
def test_infer_options_percent_encoded_userinfo():
    # Percent-encoded characters in the userinfo component must be decoded
    # so that backends (ftp, sftp, smb, ...) receive the real credentials
    # rather than the encoded form that appeared in the URL.
    so = infer_storage_options(
        "sftp://user%40corp:p%23ass%2Fword%20!@example.com:22/path"
    )
    assert so["username"] == "user@corp"
    assert so["password"] == "p#ass/word !"
    assert so["host"] == "example.com"
    assert so["port"] == 22
    assert so["path"] == "/path"

    # Unencoded credentials pass through unchanged.
    so = infer_storage_options("ftp://plainuser:plainpw@example.com/f")
    assert so["username"] == "plainuser"
    assert so["password"] == "plainpw"

    # Backwards-compat: URLs whose password contains a bare ``%`` that is
    # not a valid percent-escape must be preserved.  ``50%off`` (no hex
    # after the %) and ``pass%`` (trailing %) already round-trip through
    # ``urllib.parse.unquote``; ``pass%ab`` decodes to the byte 0xAB which
    # is not valid UTF-8, so the loose default ``errors='replace'`` would
    # silently corrupt it to U+FFFD.  We fall back to the raw form in that
    # case so the caller still sees the original password.
    for pw in ("50%off", "pass%", "pass%ab"):
        so = infer_storage_options(f"sftp://user:{pw}@example.com/f")
        assert so["password"] == pw, pw


# Ported verbatim from fsspec (fsspec/filesystem_spec#2170).
def test_infer_ipv6_host():
    # The address itself contains colons, so only a colon after the closing
    # bracket separates the port.
    out = infer_storage_options("hdfs://[::1]/mnt/test.csv")
    assert out["host"] == "[::1]"
    assert out["path"] == "/mnt/test.csv"
    assert "port" not in out

    out = infer_storage_options("hdfs://[2001:db8::1]/mnt/test.csv")
    assert out["host"] == "[2001:db8::1]"

    # A port, a user and a password are still picked up alongside the address.
    out = infer_storage_options("hdfs://[::1]:8020/mnt/test.csv")
    assert out["host"] == "[::1]"
    assert out["port"] == 8020

    out = infer_storage_options("hdfs://user:pwd@[2001:db8::1]:8020/mnt/test.csv")
    assert out["host"] == "[2001:db8::1]"
    assert out["port"] == 8020
    assert out["username"] == "user"
    assert out["password"] == "pwd"


def test_https_userinfo_is_decoded():
    """The fork parses http/https, so their userinfo is decoded too."""
    assert infer_storage_options("https://u:p%40ss@host/x")["password"] == "p@ss"


def test_http_ipv6_host_with_port():
    out = infer_storage_options("http://[::1]:8080/x")
    assert out["host"] == "[::1]"
    assert out["port"] == 8080


def test_userinfo_is_decoded_exactly_once():
    """`%2540` is an encoded `%40`: one decode gives `%40`, two would give `@`."""
    assert infer_storage_options("ftp://u:p%2540@host/x")["password"] == "p%40"


def test_plus_in_userinfo_stays_a_plus():
    """Userinfo is not form-encoded, so `+` is not a space."""
    assert infer_storage_options("ftp://al+ce:pw@host/x")["username"] == "al+ce"


def test_percent_without_hex_digits_is_kept():
    out = infer_storage_options("ftp://a%zzb:p%zz@host/x")
    assert out["username"] == "a%zzb"
    assert out["password"] == "p%zz"


def test_invalid_utf8_keeps_the_whole_value_raw():
    """Decoding is all-or-nothing: no escape is decoded, none is replaced."""
    out = infer_storage_options("ftp://a%40%ffb:p%40%ff@host/x")
    assert out["username"] == "a%40%ffb"
    assert out["password"] == "p%40%ff"


def _upstream_decodes_userinfo_and_keeps_ipv6_hosts() -> bool:
    upstream = fsspec.utils.infer_storage_options
    return (
        upstream("ftp://u:p%40@h/x").get("password") == "p@"
        and upstream("hdfs://[::1]/p").get("host") == "[::1]"
    )


#: The non-http forms the fork and fsspec are expected to agree on. The http and
#: https forms are left out: the fork parses them on purpose and fsspec does not.
UPSTREAM_PARITY_URIS = [
    "webdav://user:p%40ss@host/x",
    "ftp://us%3Aer:p%2Fw@host:21/x",
    "hdfs://[2001:db8::1]/p",
    "hdfs://[2001:db8::1]:8020/p",
    "smb://u:p%40@[::1]:445/share/x",
    "ftp://u:%ff@host/x",
    "ftp://u:a+b@host/x",
    "ftp://u:p%2540@host/x",
    "s3://bucket/key",
    "C:\\x\\y",
    "relative/path",
    "oci://bucket@ns/x",
]


@pytest.mark.skipif(
    not _upstream_decodes_userinfo_and_keeps_ipv6_hosts(),
    reason="the installed fsspec carries neither or only one of fsspec#2114 and #2170",
)
@pytest.mark.parametrize("uri", UPSTREAM_PARITY_URIS)
def test_fork_matches_upstream_once_upstream_carries_both_fixes(uri):
    """Turns on with the first fsspec release carrying both ported fixes.

    Guards this matrix only. Once it passes on the project's fsspec floor, the
    userinfo and IPv6 parts of the fork can go.
    """
    assert infer_storage_options(uri) == fsspec.utils.infer_storage_options(uri)
