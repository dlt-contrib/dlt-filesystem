from dataclasses import dataclass
from unittest.mock import patch
from urllib.parse import quote, urlparse

import pytest

from dlt_filesystem.source.error import UnsupportedEndpointError
from dlt_filesystem.source.format.registry import supported_file_format_message
from dlt_filesystem.source.fsspec.r2 import R2Source
from dlt_filesystem.source.impl.remote import (
    AzureSource,
    S3Source,
    _R2ArrowFSWrapper,
    _S3CompatibleArrowFSWrapper,
)
from dlt_filesystem.source.router import (
    determine_endpoint,
    parse_endpoint,
    parse_fragment,
    parse_uri,
    source_selects_single_file,
    split_format_hint,
)
from dlt_filesystem.util.auth import parse_azure_blob_auth


@dataclass
class URITestCase:
    uri: str
    table: str
    expect_bucket: str
    expect_glob: str


test_cases: list[URITestCase] = [
    URITestCase("s3://", "bucket/file", "bucket", "file"),
    URITestCase("s3://bucket", "file", "bucket", "file"),
    URITestCase("s3://bucket/file", "", "bucket", "file"),
    URITestCase("s3://primary", "s3://secondary/file", "primary", "file"),
    URITestCase(
        "s3://primary", "s3://secondary/path/to/file", "primary", "path/to/file"
    ),
    URITestCase("s3://primary", "path/to/file", "primary", "path/to/file"),
    URITestCase("s3://", "s3://secondary/path/to/file", "secondary", "path/to/file"),
    URITestCase("s3://", "s3://bucket/file", "bucket", "file"),
]


@pytest.mark.parametrize("test_case", test_cases, ids=[case.uri for case in test_cases])
def test_parse_uri(test_case: URITestCase):
    """Parsing a source URI splits it into the expected bucket and file glob."""
    uri = urlparse(test_case.uri)
    (bucket, glob) = parse_uri(uri, test_case.table)
    assert bucket == test_case.expect_bucket
    assert glob == test_case.expect_glob


@pytest.mark.parametrize(
    ("uri", "table", "expected"),
    [
        ("", "path/file.csv", True),
        ("", "path/*.csv", False),
        ("", "path/file?.csv", False),
        ("", "path/da?ta/y=1/*.csv", False),
        ("", "path/[ab].csv", False),
        ("", "path/**/*.csv", False),
        ("", "", False),
        ("", None, False),
        ("", "path/{a,b}.csv", True),
        ("", "path/no-extension#csv", True),
        ("", "path/file.csv#sheet_name=*", True),
        ("", "path/vendor#1/data.csv", True),
        ("", "path/vendor#1/*.csv", False),
        ("gs://", "bucket/file.csv", True),
        ("gs://primary", "gs://secondary/file.csv", True),
        ("gs://", "gs://bucket/*.csv", False),
        ("gs://", "gs://bucket/file.csv?token=secret", True),
        ("gs://bucket/file.csv", "", True),
        ("gs://bucket", "file.csv", True),
        ("s3://bucket/bar/baz?.csv", "ignored.csv", False),
        ("s3://bucket/bar/baz?.csv?token=secret", "ignored.csv", False),
        ("s3://bucket/file.csv?token=secret", "ignored.csv", True),
        ("s3://bucket/vendor#1/*.csv", "ignored.csv", False),
    ],
)
def test_source_selects_single_file(uri: str, table: str | None, expected: bool):
    """Classification uses the unparsed carrier and strips valid directives."""
    assert source_selects_single_file(uri, table) is expected


@pytest.mark.parametrize(
    ("path", "endpoint"),
    [
        ("data.csv", "read_csv"),
        ("data.csv.gz", "read_csv"),
        ("data.jsonl", "read_jsonl"),
        ("data.jsonl.gz", "read_jsonl"),
        ("data.parquet", "read_parquet"),
        ("data.bson", "read_bson"),
        ("data.bson.gz", "read_bson"),
    ],
)
def test_parse_endpoint(path: str, endpoint: str):
    """A file extension maps to the expected reader name."""
    assert parse_endpoint(path) == endpoint


@pytest.mark.parametrize(
    ("table", "path", "endpoint"),
    [
        ("bucket/path/no-extension#csv", "path/no-extension", "read_csv"),
        (
            "bucket/path/no-extension#csv_headless",
            "path/no-extension",
            "read_csv_headless",
        ),
        ("bucket/path/no-extension#jsonl", "path/no-extension", "read_jsonl"),
        ("bucket/path/no-extension#parquet", "path/no-extension", "read_parquet"),
        ("bucket/path/no-extension#bson", "path/no-extension", "read_bson"),
    ],
)
def test_determine_endpoint_format_hint(table: str, path: str, endpoint: str):
    """An explicit `#format` hint selects the reader, overriding the extension."""
    assert determine_endpoint(table, path) == endpoint


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        ("path/file.csv#csv", ("path/file.csv", "csv")),
        ("path/no-extension#csv_headless", ("path/no-extension", "csv_headless")),
        # literal '#' in the path: trailing segment is not a known format
        ("path/vendor#1/data.csv", ("path/vendor#1/data.csv", None)),
        ("path/data#unknown", ("path/data#unknown", None)),
        ("path/file.csv", ("path/file.csv", None)),
    ],
)
def test_split_format_hint(table: str, expected: tuple[str, str | None]):
    """Splitting a table spec yields the path and any trailing format hint."""
    assert split_format_hint(table) == expected


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        # no fragment
        ("file.csv", ("file.csv", None, {})),
        # bare format hint preserved (unchanged from split_format_hint)
        ("file#csv", ("file", "csv", {})),
        ("path/no-extension#csv_headless", ("path/no-extension", "csv_headless", {})),
        # a single named hint
        ("book.xlsx#sheet_name=foo", ("book.xlsx", None, {"sheet_name": "foo"})),
        # multiple named hints
        (
            "book.xlsx#sheet_name=foo&header=0",
            ("book.xlsx", None, {"sheet_name": "foo", "header": "0"}),
        ),
        # format hint and named hint coexist in one fragment
        ("feed.dat#xlsx&sheet_name=foo", ("feed.dat", "xlsx", {"sheet_name": "foo"})),
        # empty value is kept (reader decides if "" means unset)
        ("book.xlsx#sheet_name=", ("book.xlsx", None, {"sheet_name": ""})),
        # '=' in the value: parse_qsl partitions on the first '=', not split
        ("book.xlsx#x=a=b", ("book.xlsx", None, {"x": "a=b"})),
        # percent-decoding of values
        (
            "book.xlsx#sheet_name=My%20Sheet",
            ("book.xlsx", None, {"sheet_name": "My Sheet"}),
        ),
        ("book.xlsx#sheet_name=R%26D", ("book.xlsx", None, {"sheet_name": "R&D"})),
        # duplicate key: last wins
        (
            "book.xlsx#sheet_name=foo&sheet_name=bar",
            ("book.xlsx", None, {"sheet_name": "bar"}),
        ),
        # trailing '&' is harmless separator noise
        ("book.xlsx#sheet_name=foo&", ("book.xlsx", None, {"sheet_name": "foo"})),
        # mixed valid hint + invalid bare token -> whole '#...' stays literal
        ("book.xlsx#sheet_name=foo&bad", ("book.xlsx#sheet_name=foo&bad", None, {})),
        # duplicate/conflicting bare formats -> literal
        ("feed.dat#csv&parquet", ("feed.dat#csv&parquet", None, {})),
        # literal '#' in a path (trailing segment is neither hint nor format)
        ("/feeds/vendor#1/data.csv", ("/feeds/vendor#1/data.csv", None, {})),
        ("path/data#unknown", ("path/data#unknown", None, {})),
        # a bare trailing '#' interprets to nothing -> kept literal
        ("file.csv#", ("file.csv#", None, {})),
        # %23 forces a literal '#' that would otherwise look like a hint fragment
        ("book.xlsx%23sheet_name=foo", ("book.xlsx%23sheet_name=foo", None, {})),
    ],
)
def test_parse_fragment(spec: str, expected: tuple[str, str | None, dict[str, str]]):
    """Parsing a spec fragment yields the path, format hint, and named hints."""
    assert parse_fragment(spec) == expected


@pytest.mark.parametrize(
    "spec",
    [
        "path/file.csv#csv",
        "path/no-extension#csv_headless",
        "path/vendor#1/data.csv",
        "path/data#unknown",
        "path/file.csv",
    ],
)
def test_split_format_hint_matches_parse_fragment(spec: str):
    """split_format_hint stays a faithful (path, format) projection of parse_fragment."""
    path, fmt, _ = parse_fragment(spec)
    assert split_format_hint(spec) == (path, fmt)


@pytest.mark.parametrize(
    ("table", "path", "endpoint"),
    [
        # a literal '#' in the path must not be mistaken for a format hint;
        # the extension drives the reader instead
        ("bucket/vendor#1/data.csv", "vendor#1/data.csv", "read_csv"),
        ("bucket/weird#thing.jsonl", "weird#thing.jsonl", "read_jsonl"),
    ],
)
def test_determine_endpoint_literal_hash_in_path(table: str, path: str, endpoint: str):
    """A literal `#` in a path is not treated as a format hint."""
    assert determine_endpoint(table, path) == endpoint


def test_parse_endpoint_rejects_unsupported_format():
    """An unknown extension raises UnsupportedEndpointError."""
    with pytest.raises(UnsupportedEndpointError, match="Unsupported file format: bin"):
        parse_endpoint("data.bin")


def test_supported_file_format_message():
    """The supported-formats message lists the base formats in order."""
    # The base formats are always advertised, in order. Iterable-extra formats (msgpack, ...)
    # are appended only when their decoder is installed, so assert the stable base prefix.
    assert "S3 Source only supports file formats:" in supported_file_format_message(
        "S3"
    )


class FakeArrowS3Filesystem:
    """Enough of a native Arrow filesystem for construction-only source tests."""

    type_name = "s3"


@pytest.mark.parametrize(
    ("source", "scheme"),
    [(S3Source(), "s3"), (R2Source(), "r2")],
)
def test_s3_compatible_sources_map_uri_options_to_arrow(source, scheme):
    captured_kwargs = []

    def build_filesystem(**kwargs):
        captured_kwargs.append(kwargs)
        return FakeArrowS3Filesystem()

    with (
        patch("pyarrow.fs.S3FileSystem", side_effect=build_filesystem),
        patch("dlt_filesystem.source.core.resource_for_reader") as build_resource,
    ):
        source.dlt_source(
            f"{scheme}://bucket/data.csv?access_key_id=KEY"
            "&secret_access_key=SECRET&endpoint_url=http://localhost:9000"
            "&region=us-west-1",
            "",
        )

    assert captured_kwargs == [
        {
            "access_key": "KEY",
            "secret_key": "SECRET",
            "scheme": "http",
            "endpoint_override": "localhost:9000",
            "region": "us-west-1",
        }
    ]
    reference = build_resource.call_args.args[0]
    assert reference.bucket_url == f"{scheme}://bucket/"
    assert reference.fs.protocol == scheme
    assert reference.fs._strip_protocol(f"{scheme}://bucket/data.csv") == (
        "bucket/data.csv"
    )


def test_s3_source_rejects_endpoint_without_http_scheme():
    with pytest.raises(ValueError, match="Invalid endpoint_url"):
        S3Source().dlt_source(
            "s3://bucket/data.csv?access_key_id=KEY&secret_access_key=SECRET"
            "&endpoint_url=localhost:9000",
            "",
        )


def test_s3_source_rejects_endpoint_with_path_prefix():
    with pytest.raises(ValueError, match="must not include a path"):
        S3Source().dlt_source(
            "s3://bucket/data.csv?access_key_id=KEY&secret_access_key=SECRET"
            "&endpoint_url=https://gateway.example.com/s3/",
            "",
        )


@pytest.mark.parametrize(
    ("wrapper_class", "scheme"),
    [(_S3CompatibleArrowFSWrapper, "s3"), (_R2ArrowFSWrapper, "r2")],
)
@pytest.mark.parametrize("key", ["vendor#1/data.csv", "odd?name.csv"])
def test_s3_compatible_wrapper_preserves_url_delimiters_in_keys(
    wrapper_class, scheme, key
):
    fs = wrapper_class(FakeArrowS3Filesystem())

    assert fs._strip_protocol(f"{scheme}://bucket/{key}") == f"bucket/{key}"


@pytest.mark.parametrize(
    ("connection_string", "expected_account", "expected_host"),
    [
        (
            "AccountName=keyaccount;AccountKey=a2V5",
            "keyaccount",
            "https://keyaccount.blob.core.windows.net/",
        ),
        (
            "DefaultEndpointsProtocol=https;AccountName=sasaccount;"
            "SharedAccessSignature=sv=2023-01-03&ss=b&sig=a%2Fb%2Bc%3D;"
            "EndpointSuffix=core.windows.net",
            "sasaccount",
            "https://sasaccount.blob.core.windows.net/",
        ),
        (
            "AccountName=customaccount;AccountKey=a2V5;"
            "BlobEndpoint=http://127.0.0.1:10000/customaccount;",
            "customaccount",
            "http://127.0.0.1:10000/customaccount/",
        ),
        (
            "AccountName=customsas;SharedAccessSignature=sv=2023-01-03&sig=SECRET;"
            "BlobEndpoint=http://127.0.0.1:10000/customsas;",
            "customsas",
            "http://127.0.0.1:10000/customsas/",
        ),
        (
            "SharedAccessSignature=sv=2023-01-03&sig=SECRET;"
            "BlobEndpoint=https://custom.example/accounts/anonymous;",
            None,
            "https://custom.example/accounts/anonymous/",
        ),
        (
            "UseDevelopmentStorage=true",
            "devstoreaccount1",
            "http://127.0.0.1:10000/devstoreaccount1/",
        ),
        (
            "defaultendpointsprotocol=https;accountname=lowercase;"
            "accountkey=a2V5;endpointsuffix=core.windows.net;",
            "lowercase",
            "https://lowercase.blob.core.windows.net/",
        ),
    ],
)
def test_azure_connection_string_resolves_storage_identity(
    connection_string, expected_account, expected_host
):
    auth = parse_azure_blob_auth({"connection_string": [connection_string]})

    assert auth.account_name == expected_account
    assert auth.account_host == expected_host
    assert auth.connection_string == connection_string
    assert auth.account_host is not None
    assert "SECRET" not in auth.account_host
    assert "SECRET" not in repr(auth)


@pytest.mark.parametrize(
    "connection_string",
    [
        "AccountName=account;AccountKey=SECRET;Malformed",
        "AccountName=first;accountname=second;AccountKey=SECRET",
    ],
)
def test_azure_connection_string_rejects_malformed_or_conflicting_fields(
    connection_string,
):
    with pytest.raises(ValueError, match="Invalid Azure connection_string") as exc_info:
        parse_azure_blob_auth({"connection_string": [connection_string]})

    assert "SECRET" not in str(exc_info.value)


def _azure_connection_string_reference(connection_string):
    with (
        patch("adlfs.AzureBlobFileSystem") as filesystem,
        patch("dlt_filesystem.source.core.resource_for_reader") as build_resource,
    ):
        AzureSource().dlt_source(
            f"az://?connection_string={quote(connection_string, safe='')}",
            "container/*.csv",
            filesystem_incremental=True,
        )

    assert filesystem.call_args.kwargs == {
        "account_name": None,
        "connection_string": connection_string,
    }
    return build_resource.call_args.args[0]


def test_azure_connection_string_accounts_use_distinct_incremental_resources():
    first = _azure_connection_string_reference(
        "AccountName=first;AccountKey=FIRST_SECRET"
    )
    second = _azure_connection_string_reference(
        "AccountName=second;AccountKey=SECOND_SECRET"
    )

    assert first.storage_namespace == "azure:first:first.blob.core.windows.net"
    assert second.storage_namespace == "azure:second:second.blob.core.windows.net"
    assert first.incremental_resource_name != second.incremental_resource_name
    assert first.storage_namespace != "azure::azure-public"
    assert "SECRET" not in first.storage_namespace
    assert "SECRET" not in second.storage_namespace
