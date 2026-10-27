"""Tests for the S3, GCS and Azure blob destinations in ``target/remote.py``.

None of these need a cloud account or a container. The write-through tests swap the
protocol's fsspec backend for an in-memory filesystem that records the arguments
dlt constructs it with, so a real pipeline lands real rows while the test checks
that each URI option reached the backend under the name it expects.
"""

import base64
import inspect
import io
import json
from pathlib import Path
from urllib.parse import parse_qs, quote

import dlt
import pytest
from dlt.common.configuration.specs import (
    AwsCredentials,
    AzureCredentials,
    AzureServicePrincipalCredentials,
)
from dlt.destinations.impl.filesystem.filesystem import FilesystemClient
from fsspec.implementations.memory import MemoryFileSystem
from fsspec.registry import _registry
from pyarrow import parquet

from dlt_filesystem.error import MissingConnectorOption
from dlt_filesystem.target.remote import (
    AzureDestination,
    BlobFS,
    BlobFSClient,
    GCSDestination,
    S3Destination,
    blob_destination_options,
)

ROWS = [{"id": 1, "name": "Ada"}, {"id": 2, "name": "Bob"}, {"id": 3, "name": "Cy"}]

S3_AUTH = "access_key_id=KEY&secret_access_key=SECRET"
AZURE_AUTH = "account_name=acct&account_key=a2V5"
SERVICE_ACCOUNT = {
    "type": "service_account",
    "project_id": "proj",
    "private_key": "-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----\n",
    "client_email": "loader@proj.iam.gserviceaccount.com",
}
GCS_AUTH = "credentials_base64=" + quote(
    base64.b64encode(json.dumps(SERVICE_ACCOUNT).encode()).decode(), safe=""
)


def pipeline(destination, pipelines_dir: Path):
    return dlt.pipeline(
        pipeline_name="remote_destination",
        destination=destination,
        dataset_name="dataset",
        pipelines_dir=str(pipelines_dir),
    )


def load(destination, tmp_path: Path, table: str) -> None:
    """Run ``ROWS`` into ``table`` through a real dlt pipeline."""
    pipeline(destination, tmp_path / "pipelines").run(
        ROWS, table_name=table, loader_file_format="parquet"
    )


def read_parquet(files: dict[str, bytes]) -> list[dict]:
    rows: list[dict] = []
    for name in sorted(files):
        rows.extend(parquet.read_table(io.BytesIO(files[name])).to_pylist())
    return sorted(rows, key=lambda row: (row["id"], row["_dlt_load_id"]))


def user_columns(rows: list[dict]) -> list[dict]:
    return [{"id": row["id"], "name": row["name"]} for row in rows]


@pytest.fixture
def capturing_backend(monkeypatch):
    """Serve one protocol from a fresh in-memory filesystem and record its kwargs.

    Each call builds its own class, with its own store, so no file and no cached
    instance leaks between tests. dlt resolves the backend through fsspec's
    registry, which the monkeypatch restores afterwards.
    """

    def install(protocol: str) -> tuple[type[MemoryFileSystem], list[dict]]:
        captured: list[dict] = []

        class CapturingFileSystem(MemoryFileSystem):
            store: dict = {}
            pseudo_dirs = [""]

            def __init__(self, *args, **kwargs):
                captured.append(kwargs)
                super().__init__()

        CapturingFileSystem.protocol = protocol
        monkeypatch.setitem(_registry, protocol, CapturingFileSystem)
        return CapturingFileSystem, captured

    return install


def write_through(capturing_backend, tmp_path, destination, protocol, query, table):
    """Load ``ROWS`` through ``destination.dlt_dest`` into a capturing backend.

    Returns the parquet files the load wrote, as bytes by store key, and every
    kwargs dict the backend was constructed with.
    """
    backend, captured = capturing_backend(protocol)
    dest = destination.dlt_dest(f"{protocol}://?{query}", dest_table=table)
    load(dest, tmp_path, table.split("/")[-1])
    written = {
        key: value.getvalue()
        for key, value in backend.store.items()
        if key.endswith(".parquet")
    }
    return written, captured


# --- BlobFSClient.dataset_path ---------------------------------------------------


def test_dataset_path_override_still_overrides_dlt():
    """``BlobFSClient`` overrides dlt's internal ``FilesystemClient.dataset_path``.

    dlt reads that property wherever it lays out a dataset, so if dlt renames it or
    turns it into a method, the override goes dead without an error and tables land
    one directory deeper. This fails at the rename rather than at the layout.
    """
    upstream = inspect.getattr_static(FilesystemClient, "dataset_path")
    assert isinstance(upstream, property)
    override = inspect.getattr_static(BlobFSClient, "dataset_path")
    assert isinstance(override, property)
    assert override is not upstream


def test_blob_fs_writes_tables_at_the_bucket_root(tmp_path):
    """A table lands directly under the bucket, with no dataset directory between.

    The second load runs from a fresh local pipeline, so dlt first restores the
    first load's state from the bucket, and reads go through the overridden path
    as well as writes.
    """
    bucket = tmp_path / "bucket"

    def destination():
        return BlobFS(
            bucket_url=bucket.as_uri(),
            enable_dataset_name_normalization=False,
        )

    @dlt.resource(name="people")
    def people(marker):
        dlt.current.resource_state()["marker"] = marker
        yield ROWS

    first = pipeline(destination(), tmp_path / "first")
    first.run(people("first"), loader_file_format="parquet")

    second = pipeline(destination(), tmp_path / "second")
    second.sync_destination()
    resources = second.state["sources"]["remote_destination"]["resources"]
    assert resources["people"]["marker"] == "first"
    second.run(people("second"), loader_file_format="parquet")

    assert not (bucket / "dataset").exists()
    files = {path.name: path.read_bytes() for path in bucket.glob("people/*.parquet")}
    assert user_columns(read_parquet(files)) == user_columns(
        sorted(ROWS + ROWS, key=lambda row: row["id"])
    )
    assert (bucket / "_dlt_loads").is_dir()


# --- dlt_dest: table path and options --------------------------------------------


@pytest.mark.parametrize(
    ("uri", "table", "bucket_url"),
    [
        # no host: the first table segment is the bucket
        (f"s3://?{S3_AUTH}", "bucket/table", "s3://bucket"),
        (f"s3://?{S3_AUTH}", "bucket/a/b/table", "s3://bucket/a/b"),
        (f"s3://?{S3_AUTH}", "/bucket/table", "s3://bucket"),
        # a host names the bucket, and a URI path prefixes the table
        (f"s3://bucket?{S3_AUTH}", "table", "s3://bucket"),
        (f"s3://bucket/pre/fix?{S3_AUTH}", "a/table", "s3://bucket/pre/fix/a"),
        (f"gs://?{GCS_AUTH}", "bucket/a/table", "gs://bucket/a"),
        (f"az://?{AZURE_AUTH}", "container/a/table", "az://container/a"),
    ],
)
def test_dlt_dest_bucket_url(uri, table, bucket_url):
    destination = {"s3": S3Destination, "gs": GCSDestination, "az": AzureDestination}[
        uri.split(":", 1)[0]
    ]()
    dest = destination.dlt_dest(uri, dest_table=table)
    assert isinstance(dest, BlobFS)
    assert dest.config_params["bucket_url"] == bucket_url
    assert dest.config_params["enable_dataset_name_normalization"] is False
    assert destination.dlt_run_params(uri, table) == {"table_name": "table"}


@pytest.mark.parametrize("table", ["table", "/table/"])
def test_dlt_dest_without_a_host_needs_a_bucket_in_the_table(table):
    with pytest.raises(ValueError, match="format {bucket-name}/{path}"):
        S3Destination().dlt_dest(f"s3://?{S3_AUTH}", dest_table=table)


@pytest.mark.parametrize(
    ("uri", "table"),
    [
        (f"s3://?{S3_AUTH}", "bucket/table/"),
        (f"s3://bucket?{S3_AUTH}", "table/"),
        (f"s3://bucket?{S3_AUTH}", "/table"),
    ],
    ids=["no-host", "host", "host-leading"],
)
def test_a_slash_around_the_table_is_ignored(uri, table):
    dest = S3Destination().dlt_dest(uri, dest_table=table)
    assert dest.config_params["bucket_url"] == "s3://bucket"
    assert S3Destination().dlt_run_params(uri, table) == {"table_name": "table"}


def test_dlt_dest_passes_layout_through():
    layout = "{table_name}/{YYYY}/{load_id}.{file_id}.{ext}"
    dest = S3Destination().dlt_dest(
        f"s3://?{S3_AUTH}&layout={quote(layout)}", dest_table="bucket/table"
    )
    assert dest.config_params["layout"] == layout
    assert "kwargs" not in dest.config_params


# --- S3 ----------------------------------------------------------------------------


def test_s3_credentials():
    credentials = S3Destination().credentials(
        parse_qs(f"{S3_AUTH}&endpoint_url=http://localhost:9000")
    )
    assert isinstance(credentials, AwsCredentials)
    assert credentials.aws_access_key_id == "KEY"
    assert credentials.aws_secret_access_key == "SECRET"  # noqa: S105
    assert credentials.endpoint_url == "http://localhost:9000"


@pytest.mark.parametrize(
    ("query", "missing"),
    [
        ("secret_access_key=SECRET", "access_key_id"),
        ("access_key_id=KEY", "secret_access_key"),
    ],
)
def test_s3_missing_key_names_the_option(query, missing):
    with pytest.raises(MissingConnectorOption, match=f"^{missing} is required"):
        S3Destination().dlt_dest(f"s3://?{query}", dest_table="bucket/table")


@pytest.mark.parametrize("endpoint_url", ["localhost:9000", "http://", "/path"])
def test_s3_rejects_an_endpoint_without_scheme_and_host(endpoint_url):
    with pytest.raises(ValueError, match="Invalid endpoint_url"):
        S3Destination().dlt_dest(
            f"s3://?{S3_AUTH}&endpoint_url={endpoint_url}", dest_table="bucket/table"
        )


def test_s3_write_through(capturing_backend, tmp_path):
    written, captured = write_through(
        capturing_backend,
        tmp_path,
        S3Destination(),
        "s3",
        f"{S3_AUTH}&endpoint_url=http://localhost:9000",
        "bucket/out/people",
    )
    assert list(written) and all(
        key.startswith("s3://bucket/out/people/") for key in written
    )
    assert user_columns(read_parquet(written)) == ROWS
    assert {
        (kwargs.get("key"), kwargs.get("secret"), kwargs.get("endpoint_url"))
        for kwargs in captured
    } == {("KEY", "SECRET", "http://localhost:9000")}


# --- GCS ---------------------------------------------------------------------------


def test_gcs_credentials_from_base64():
    credentials = GCSDestination().credentials(parse_qs(GCS_AUTH))
    assert credentials == SERVICE_ACCOUNT


def test_gcs_credentials_from_path(tmp_path):
    path = tmp_path / "service-account.json"
    path.write_text(json.dumps(SERVICE_ACCOUNT))
    credentials = GCSDestination().credentials({"credentials_path": [str(path)]})
    assert credentials == SERVICE_ACCOUNT


def test_gcs_missing_credentials():
    with pytest.raises(
        MissingConnectorOption, match="credentials_path or credentials_base64"
    ):
        GCSDestination().dlt_dest("gs://?project=proj", dest_table="bucket/table")


def test_gcs_write_through(capturing_backend, tmp_path):
    written, captured = write_through(
        capturing_backend, tmp_path, GCSDestination(), "gs", GCS_AUTH, "bucket/people"
    )
    assert user_columns(read_parquet(written)) == ROWS
    assert captured
    assert all(kwargs["project"] == "proj" for kwargs in captured)
    for kwargs in captured:
        token = kwargs["token"]
        assert {name: token.get(name) for name in SERVICE_ACCOUNT} == SERVICE_ACCOUNT


# --- Azure -------------------------------------------------------------------------


def test_azure_account_key_credentials():
    credentials = AzureDestination().credentials(
        parse_qs(f"{AZURE_AUTH}&account_host=acct.blob.core.chinacloudapi.cn")
    )
    assert type(credentials) is AzureCredentials
    assert credentials.azure_storage_account_name == "acct"
    assert credentials.azure_storage_account_key == "a2V5"
    assert credentials.azure_account_host == "acct.blob.core.chinacloudapi.cn"


def test_azure_service_principal_credentials():
    credentials = AzureDestination().credentials(
        parse_qs("account_name=acct&tenant_id=t&client_id=c&client_secret=s")
    )
    assert isinstance(credentials, AzureServicePrincipalCredentials)
    assert credentials.azure_storage_account_name == "acct"
    assert credentials.azure_tenant_id == "t"
    assert credentials.azure_client_id == "c"
    assert credentials.azure_client_secret == "s"  # noqa: S105


# Each query is URL-encoded the way a caller has to: a base64 account key carries
# ``+ / =``, and a SAS token carries its own ``& =`` pairs, which ``parse_qs`` would
# otherwise turn into a space and into stray parameters.
azure_write_cases = [
    (
        "account_name=acct&account_key=a2V5%2Bx%2F%3D%3D",
        {"account_name": "acct", "account_key": "a2V5+x/=="},
    ),
    (
        "account_name=acct&sas_token=sv%3D2021%26sig%3Da%2Fb%2Bc%3D%3D",
        {"account_name": "acct", "sas_token": "sv=2021&sig=a/b+c=="},
    ),
    (
        "account_name=acct&tenant_id=t&client_id=c&client_secret=s"
        "&account_host=acct.blob.core.chinacloudapi.cn",
        {
            "tenant_id": "t",
            "client_id": "c",
            "client_secret": "s",
            "account_host": "acct.blob.core.chinacloudapi.cn",
        },
    ),
    (
        f"{AZURE_AUTH}&account_host=acct.blob.core.chinacloudapi.cn",
        {"account_host": "acct.blob.core.chinacloudapi.cn"},
    ),
]


@pytest.mark.parametrize(
    ("query", "expected"),
    azure_write_cases,
    ids=["encoded-account-key", "encoded-sas-token", "service-principal", "host"],
)
def test_azure_write_through(capturing_backend, tmp_path, query, expected):
    written, captured = write_through(
        capturing_backend, tmp_path, AzureDestination(), "az", query, "container/out"
    )
    assert user_columns(read_parquet(written)) == ROWS
    assert captured
    for kwargs in captured:
        assert {name: kwargs.get(name) for name in expected} == expected


def test_azure_connection_string_write_through(capturing_backend, tmp_path):
    """dlt has no connection-string credential, so the string bypasses it and
    reaches adlfs as a constructor argument. ``api_version`` stays behind, since
    adlfs drops it on that path anyway."""
    connection_string = (
        "DefaultEndpointsProtocol=http;AccountName=acct;AccountKey=a2V5;"
        "BlobEndpoint=http://127.0.0.1:10000/acct;"
    )
    query = f"connection_string={quote(connection_string, safe='')}"
    written, captured = write_through(
        capturing_backend,
        tmp_path,
        AzureDestination(),
        "az",
        f"{query}&api_version=2025-11-05",
        "container/out",
    )
    assert user_columns(read_parquet(written)) == ROWS
    assert captured
    for kwargs in captured:
        assert kwargs["connection_string"] == connection_string
        assert "api_version" not in kwargs
        assert "account_key" not in kwargs


@pytest.mark.parametrize(
    ("query", "error", "match"),
    [
        ("account_name=acct", MissingConnectorOption, "is required"),
        ("account_key=a2V5", MissingConnectorOption, "^account_name is required"),
        (
            "account_name=acct&tenant_id=t&client_id=c",
            MissingConnectorOption,
            "client_secret",
        ),
        (
            f"{AZURE_AUTH}&tenant_id=t&client_id=c&client_secret=s",
            ValueError,
            "Conflicting Azure credentials",
        ),
        (
            f"{AZURE_AUTH}&sas_token=sv%3Dx",
            ValueError,
            "Conflicting Azure credentials",
        ),
        (
            "connection_string=UseDevelopmentStorage%3Dtrue&account_key=a2V5",
            ValueError,
            "cannot be combined with account_key",
        ),
    ],
    ids=[
        "no-key",
        "no-account",
        "partial-service-principal",
        "key-and-service-principal",
        "key-and-sas",
        "connection-string-and-key",
    ],
)
def test_azure_rejects_incomplete_or_conflicting_auth(query, error, match):
    with pytest.raises(error, match=match):
        AzureDestination().dlt_dest(f"az://?{query}", dest_table="container/table")


# --- blob_destination_options -------------------------------------------------------


@pytest.mark.parametrize(
    ("protocol", "query", "credentials_type"),
    [
        ("s3", S3_AUTH, AwsCredentials),
        ("gs", GCS_AUTH, dict),
        ("az", AZURE_AUTH, AzureCredentials),
    ],
)
def test_blob_destination_options_builds_credentials(protocol, query, credentials_type):
    options = blob_destination_options(protocol, parse_qs(query))
    assert set(options) == {"credentials"}
    assert isinstance(options["credentials"], credentials_type)


@pytest.mark.parametrize("protocol", ["s3", "gs", "az"])
def test_blob_destination_options_leaves_an_empty_query_to_dlt(protocol):
    assert blob_destination_options(protocol, {}) == {}


def test_blob_destination_options_ignores_protocols_dlt_does_not_type():
    assert blob_destination_options("sftp", parse_qs("username=u")) == {}


def test_blob_destination_options_forwards_a_connection_string():
    connection_string = "UseDevelopmentStorage=true"
    options = blob_destination_options("az", {"connection_string": [connection_string]})
    assert options["kwargs"] == {"connection_string": connection_string}


def test_blob_destination_options_is_strict_once_a_query_is_given():
    with pytest.raises(MissingConnectorOption, match="^secret_access_key"):
        blob_destination_options("s3", parse_qs("access_key_id=KEY"))


@pytest.mark.parametrize(
    ("destination", "protocol"),
    [(S3Destination, "s3"), (GCSDestination, "gs"), (AzureDestination, "az")],
)
def test_dlt_dest_leaves_an_empty_query_to_dlt(destination, protocol):
    dest = destination().dlt_dest(f"{protocol}://", dest_table="bucket/table")
    assert "credentials" not in dest.config_params
    assert "kwargs" not in dest.config_params


def test_dlt_dest_leaves_a_layout_only_query_to_dlt():
    dest = S3Destination().dlt_dest("s3://?layout=x", dest_table="bucket/table")
    assert dest.config_params["layout"] == "x"
    assert "credentials" not in dest.config_params


def test_dlt_dest_is_strict_once_a_credential_is_given():
    with pytest.raises(MissingConnectorOption, match="^secret_access_key"):
        S3Destination().dlt_dest("s3://?access_key_id=KEY", dest_table="bucket/table")


@pytest.mark.parametrize(
    "uri",
    [
        "s3://?access_key_id=&secret_access_key=",
        "s3://?access_key_id=&layout=x",
        "gs://?credentials_base64=",
        "az://?connection_string=",
    ],
)
def test_dlt_dest_is_strict_when_a_credential_is_blank(uri):
    destination = {"s3": S3Destination, "gs": GCSDestination, "az": AzureDestination}[
        uri.split(":", 1)[0]
    ]()
    with pytest.raises(MissingConnectorOption):
        destination.dlt_dest(uri, dest_table="bucket/table")
