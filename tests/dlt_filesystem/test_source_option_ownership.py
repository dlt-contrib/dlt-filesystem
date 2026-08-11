"""Pin the ownership boundary between omniload's run options and connector options.

Every filesystem-family source is driven with the full run-parameter set that
`omniload.api` passes, and asserted on twice: the filesystem constructor must see
none of those names, and the two the package does consume must arrive on the
`FilesystemReference` instead.

The spy needs a per-connector patch map because there is no single construction
hook: most connectors resolve a `fs_class` property, FTP imports its class from
the fsspec module, SFTP goes through `fsspec.filesystem`, and the local source
wraps a pyarrow filesystem, whose constructor is recorded rather than replaced.
"""

import ast
import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from fsspec.implementations.memory import MemoryFileSystem

from dlt_filesystem.source.fsspec.ftp import FTPSource
from dlt_filesystem.source.fsspec.local import LocalFilesystemSource
from dlt_filesystem.source.impl.remote import SFTPSource
from dlt_filesystem.source.model import RUN_OPTION_KEYS, FilesystemReference
from omniload.core.factory import SourceDestinationFactory

ASSETS = Path(__file__).resolve().parents[1] / "assets"
PRIVATE_KEY_FILE = (ASSETS / "privatekey.pem").as_posix()
PRIVATE_KEY_FINGERPRINT = (ASSETS / "privatekey-fingerprint.txt").read_text().strip()
OCI_CONFIG = json.dumps(
    {
        "user": "ocid1.user.oc1..24g4uzg",
        "region": "us-ashburn-1",
        "tenancy": "ocid1.tenancy.oc1..23423r3",
        "key_file": PRIVATE_KEY_FILE,
        "fingerprint": PRIVATE_KEY_FINGERPRINT,
    },
    # Compact, because the value rides in a query string and a space would end it.
    separators=(",", ":"),
)
GRAPH_AUTH = (
    "client_id=1d2befad-2f22-4124-a779-b147dfeca342"
    "&tenant_id=6b337423-f504-4060-a91b-e9eaaf782609"
    "&client_secret=abc~xyz789EXAMPLE_foo"
)

#: The run parameters `omniload.api` hands to every `dlt_source`, with values that
#: exercise the two the filesystem package consumes. `incremental_key` stays `None`
#: because every connector rejects it up front (it manages incrementality itself).
RUN_OPTIONS: dict[str, Any] = {
    "incremental_key": None,
    "requested_incremental_key": None,
    "requested_primary_key": None,
    "merge_key": None,
    "interval_start": None,
    "interval_end": None,
    "sql_backend": "pyarrow",
    "page_size": 1000,
    "sql_reflection_level": "full",
    "sql_limit": None,
    "sql_exclude_columns": None,
    "extract_parallelism": 1,
    "column_types": {"name": {"data_type": "text"}},
    "filesystem_incremental": True,
}


@dataclass
class Case:
    """One registered filesystem-family scheme and a URI that reaches its constructor."""

    scheme: str
    uri: str
    table: str = ""
    #: Connection arguments this URI carries, which must survive the split.
    expect_kwargs: tuple[str, ...] = field(default_factory=tuple)


#: One case per filesystem-family scheme in `omniload.core.registry`.
CASES = [
    Case(
        "abfss",
        "abfss://schrott@acme.dfs.core.windows.net/path/to/data.parquet?account_name=acme&account_key=secret",
        expect_kwargs=("account_name", "account_key"),
    ),
    Case(
        "adls",
        "adls://schrott@acme.dfs.core.windows.net/path/to/data.parquet?account_name=acme&account_key=secret",
        expect_kwargs=("account_name", "account_key"),
    ),
    Case(
        "az",
        "az://schrott@acme.dfs.core.windows.net/path/to/data.parquet?account_name=acme&account_key=secret",
        expect_kwargs=("account_name", "account_key"),
    ),
    Case("dbfs", "dbfs:/Volumes/catalog/schema/volume/path/to/data.parquet"),
    Case(
        "dropbox",
        "dropbox://path/to/data.parquet?token=secret",
        expect_kwargs=("token",),
    ),
    Case("file", "file://__TMP__/data.parquet"),
    Case(
        "ftp",
        "ftp://username:password@intranet.example.org/path/to/data.parquet?tls=tls",
        expect_kwargs=("host", "username", "password"),
    ),
    Case(
        "gdrive",
        "gdrive://path/to/data.parquet?token=anon",
        expect_kwargs=("token",),
    ),
    Case(
        "gs",
        "gs://table-bucket-name/path/to/data.parquet?credentials_path=/path/to/service-account.json",
        expect_kwargs=("token",),
    ),
    Case(
        "hdfs",
        "hdfs://example.com:8020/path/to/data.parquet?user=test",
        expect_kwargs=("host", "port", "user"),
    ),
    Case(
        # WebDAV takes its URL positionally and folds any credentials into `auth`.
        "http+webdav",
        "http+webdav://public.example.org/path/to/data.parquet",
        expect_kwargs=("auth",),
    ),
    Case(
        "https+webdav",
        "https+webdav://username:password@cloud.example.org:4443/remote.php/webdav",
        table="path/to/data.parquet",
        expect_kwargs=("auth",),
    ),
    Case(
        "msgd",
        f"msgd://site_name/drive_name/path/to/data.parquet?{GRAPH_AUTH}",
        expect_kwargs=("client_id", "tenant_id", "client_secret"),
    ),
    Case(
        "oci",
        f"oci://bucket@namespace/prefix/path/to/data.parquet?iam_type=api_key&config={OCI_CONFIG}",
        expect_kwargs=("config",),
    ),
    Case(
        "onedrive",
        f"onedrive://drive_name/path/to/data.parquet?{GRAPH_AUTH}",
        expect_kwargs=("client_id", "tenant_id", "client_secret"),
    ),
    Case(
        "oss",
        "oss://bucket/path/to/data.parquet?endpoint=http://oss-cn-hangzhou.aliyuncs.com/&key=foo&secret=bar",
        expect_kwargs=("endpoint", "key", "secret"),
    ),
    Case(
        "r2",
        "r2://bucket/path/to/data.parquet?access_key_id=foo&secret_access_key=bar",
        expect_kwargs=("key", "secret"),
    ),
    Case(
        "s3",
        "s3://bucket/path/to/data.parquet?access_key_id=foo&secret_access_key=bar",
        expect_kwargs=("key", "secret"),
    ),
    Case(
        "sftp",
        "sftp://username:password@intranet.example.org:2222/path/to/data.parquet",
        expect_kwargs=("host", "port", "username", "password"),
    ),
    Case(
        "sharepoint",
        f"sharepoint://site_name/drive_name/path/to/data.parquet?{GRAPH_AUTH}",
        expect_kwargs=("client_id", "tenant_id", "client_secret"),
    ),
    Case(
        "smb",
        "smb://workgroup;user:password@server.example.org:445/path/to/data.parquet",
        expect_kwargs=("host", "port", "username", "password"),
    ),
    Case(
        "webhdfs",
        "webhdfs://host:9870/endpoint",
        table="path/to/data.parquet",
        expect_kwargs=("host", "port"),
    ),
]


def _skip_unsupported(scheme: str) -> None:
    """Mirror the platform limits the touch matrix already documents."""
    if scheme in {"dbfs", "hdfs"} and sys.version_info < (3, 11):
        pytest.skip(f"{scheme}:// needs Python 3.11+")
    if scheme == "oci" and sys.platform == "win32":
        pytest.skip("oci:// fails testing on Windows")


def _spy_class(calls: list[dict[str, Any]], protocol: str):
    """Build a filesystem class that records the keywords it was constructed with.

    `cachable` is off because fsspec keys its instance cache on the constructor
    arguments and skips `__init__` on a hit, which would silently record nothing
    for the second case that happens to construct an equal filesystem.
    """

    class SpyFileSystem(MemoryFileSystem):
        cachable = False
        # `ArrowFSWrapper` reads `type_name`, and `S3CompatibleSource` derives its
        # bucket URL from `protocol`.
        type_name = protocol

        def __init__(self, *args, **kwargs):
            calls.append(dict(kwargs))
            super().__init__()

    SpyFileSystem.protocol = protocol
    return SpyFileSystem


@contextmanager
def _spy_on_filesystem(source, scheme: str):
    """Patch whatever construction hook this connector actually uses."""
    calls: list[dict[str, Any]] = []
    spy = _spy_class(calls, scheme)

    if isinstance(source, FTPSource):
        # Imported into the function body from the fsspec module, not via `fs_class`.
        with mock.patch("fsspec.implementations.ftp.FTPFileSystem", spy):
            yield calls
    elif isinstance(source, SFTPSource):
        with mock.patch("fsspec.filesystem", lambda _protocol, **kwargs: spy(**kwargs)):
            yield calls
    elif isinstance(source, LocalFilesystemSource):
        # Wraps a pyarrow filesystem, so record that constructor instead of replacing
        # it: `ArrowFSWrapper` needs a real arrow filesystem underneath. Recording it
        # rather than asserting "no spy fired" is what makes the empty-kwargs claim
        # falsifiable, so forwarding a run option here would fail the same assertion
        # as everywhere else.
        import pyarrow.fs

        real_local = pyarrow.fs.LocalFileSystem

        def recording_local(*args, **kwargs):
            calls.append(dict(kwargs))
            return real_local(*args, **kwargs)

        with mock.patch("pyarrow.fs.LocalFileSystem", recording_local):
            yield calls
    else:
        with mock.patch.object(type(source), "fs_class", property(lambda self: spy)):
            yield calls


def _drive(case: Case, tmp_path, **overrides) -> tuple[FilesystemReference, list[dict]]:
    """Build the source for one scheme and capture both sides of the boundary."""
    # Substituted by hand rather than with `str.format`, because the OCI URI carries a
    # JSON object in its query string and its braces are not format fields.
    uri = case.uri.replace("__TMP__", tmp_path.as_posix())
    source = SourceDestinationFactory(uri, "file://").get_source()
    options = {**RUN_OPTIONS, **overrides}

    with (
        mock.patch("dlt_filesystem.source.core.resource_for_reader") as build,
        _spy_on_filesystem(source, case.scheme) as calls,
    ):
        source.dlt_source(uri=uri, table=case.table, **options)

    assert build.call_count == 1, f"{case.scheme}: reference was not built once"
    return build.call_args.args[0], calls


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.scheme)
def test_filesystem_constructor_receives_no_run_option(case, tmp_path):
    """The headline contract: no name omniload owns reaches an fsspec constructor."""
    _skip_unsupported(case.scheme)
    _, calls = _drive(case, tmp_path)

    assert calls, f"{case.scheme}: the filesystem spy was never constructed"
    for received in calls:
        leaked = sorted(set(received) & RUN_OPTION_KEYS)
        assert leaked == [], f"{case.scheme}: run options reached the constructor"


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.scheme)
def test_uri_connection_options_survive_the_split(case, tmp_path):
    """The guardrail: the split must not be tightened into dropping real kwargs."""
    _skip_unsupported(case.scheme)
    _, calls = _drive(case, tmp_path)

    received = calls[-1]
    missing = [key for key in case.expect_kwargs if key not in received]
    assert missing == [], f"{case.scheme}: connection arguments were dropped"
    if not case.expect_kwargs:
        # This URI addresses its object by path alone, so a clean constructor is the
        # whole assertion: anything present would have come from the run.
        assert received == {}


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.scheme)
def test_resource_options_reach_the_reference(case, tmp_path):
    """Both options omniload owns land on the reference, on every scheme."""
    _skip_unsupported(case.scheme)
    reference, _ = _drive(case, tmp_path)

    assert reference.filesystem_incremental is True
    assert reference.column_types == RUN_OPTIONS["column_types"]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.scheme)
def test_lister_is_namespaced_only_when_incremental_is_requested(case, tmp_path):
    """The no-op fingerprint: a plain `filesystem` lister means the flag was dropped."""
    _skip_unsupported(case.scheme)

    enabled, _ = _drive(case, tmp_path, filesystem_incremental=True)
    disabled, _ = _drive(case, tmp_path, filesystem_incremental=False)

    assert enabled.filesystem_incremental is True
    assert disabled.filesystem_incremental is False
    assert enabled.incremental_resource_name.startswith("filesystem_")
    assert enabled.incremental_resource_name != "filesystem"


def test_uri_column_types_remains_the_fallback(tmp_path):
    """A URI-supplied value still reaches the reference when the run supplies none."""
    case = Case("ftp", "ftp://user:pw@host/bucket/data.csv?column_types=from-uri")
    reference, calls = _drive(case, tmp_path, column_types=None)

    assert reference.column_types == "from-uri"
    assert "column_types" not in calls[-1]


def test_run_column_types_wins_over_the_uri(tmp_path):
    """Where both name it, the run value is the one the reader gets."""
    case = Case("ftp", "ftp://user:pw@host/bucket/data.csv?column_types=from-uri")
    reference, _ = _drive(case, tmp_path)

    assert reference.column_types == RUN_OPTIONS["column_types"]


def test_uri_parameter_named_like_a_run_option_never_reaches_the_constructor(tmp_path):
    """The query string is the second carrier, and it gets the same rule.

    Before the boundary existed the run value overwrote the URI value and both
    reached the constructor; now neither does, and a genuine connection argument
    beside them is unaffected.
    """
    case = Case(
        "ftp",
        "ftp://user:pw@host/bucket/data.csv?filesystem_incremental=true&page_size=7&tls=true",
    )
    _, calls = _drive(case, tmp_path)

    received = calls[-1]
    assert "filesystem_incremental" not in received
    assert "page_size" not in received
    assert received["tls"] is True


def _api_call_site_keywords() -> set[str]:
    """Read the keywords `run_ingest` passes into `source.dlt_source`."""
    import omniload.api

    source_file = omniload.api.__file__
    assert source_file is not None
    tree = ast.parse(Path(source_file).read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "dlt_source":
            return {
                kw.arg
                for kw in node.keywords
                if kw.arg is not None and kw.arg not in {"uri", "table"}
            }
    raise AssertionError("no `source.dlt_source(...)` call found in omniload.api")


def test_run_option_keys_match_the_api_call_site():
    """Adding a run parameter without classifying it must fail here, not leak silently."""
    assert _api_call_site_keywords() == set(RUN_OPTION_KEYS)


def test_every_registered_filesystem_scheme_is_covered():
    """A new transport cannot join the family without landing in this matrix."""
    from omniload.core.registry import sources

    # Read the unresolved dotted paths, so asserting on the registry does not import
    # every SaaS connector just to learn which module a scheme belongs to.
    registered = {
        scheme
        for scheme, dotted_path in sources._paths.items()
        if dotted_path.startswith("dlt_filesystem.")
    }
    assert registered == {case.scheme for case in CASES}


def test_reference_defaults_read_as_a_run_that_enabled_nothing():
    """`infer_resource` may be called without options; that must not enable anything."""
    from dlt_filesystem.source.model import ResourceOptions

    options = ResourceOptions()
    assert options.filesystem_incremental is False
    assert options.column_types is None


def _split(**kwargs) -> tuple[Any, dict[str, Any]]:
    from dlt_filesystem.source.model import split_run_options

    return split_run_options(kwargs)


def test_split_is_subtractive_not_a_whitelist():
    """A programmatic caller's own connector keyword must pass straight through.

    GCS relies on this today: it honours a caller-supplied `token`.
    """
    options, connector = _split(
        token="anon",  # noqa: S106 - gcsfs anonymous-access sentinel, not a secret
        page_size=1000,
        filesystem_incremental=True,
    )

    assert connector == {"token": "anon"}
    assert options.filesystem_incremental is True


def test_split_leaves_the_caller_mapping_alone():
    """The connectors read `kwargs` after splitting, so the split must not mutate it."""
    original: dict[str, Any] = {"page_size": 1000, "token": "anon"}
    _split(**original)

    assert original == {"page_size": 1000, "token": "anon"}


def test_resource_options_default_when_run_omits_them():
    options, connector = _split(token="anon")  # noqa: S106 - anonymous-access sentinel

    assert options.filesystem_incremental is False
    assert options.column_types is None
    assert connector == {"token": "anon"}


#: A fixed modification time, so the second run sees an unchanged file.
STAMP = datetime(2026, 1, 1, tzinfo=timezone.utc)


class MemoryBackedFTP(MemoryFileSystem):
    """An in-memory stand-in for a locator connector's backend.

    Two adjustments make it drivable end to end: paths are stripped for any scheme
    rather than `memory://` alone, and listings carry `mtime`, which is what the
    lister resolves a file's modification date from.
    """

    cachable = False

    @classmethod
    def _strip_protocol(cls, path):
        return "/" + str(path).split("://", 1)[-1].lstrip("/")

    def __init__(self, *args, **kwargs):
        super().__init__()

    def info(self, path, **kwargs):
        return dict(super().info(path, **kwargs), mtime=STAMP)

    def ls(self, path, detail=True, **kwargs):
        listing = super().ls(path, detail=detail, **kwargs)
        if not detail:
            return listing
        return [dict(entry, mtime=STAMP) for entry in listing]


def test_second_run_over_a_locator_connector_loads_nothing_new(tmp_path):
    """The behaviour the dropped flag cost: an unchanged file is read once, not every run.

    Before the boundary existed, `--filesystem-incremental` never reached the
    reference on this path, so every run re-read every file.
    """
    import dlt
    import duckdb

    MemoryBackedFTP().pipe_file("ftp://host:21/bucket/people.csv", b"name\nAlice\n")
    database = tmp_path / "warehouse.duckdb"

    def build_source():
        return FTPSource().dlt_source(
            "ftp://user:pw@host/bucket/people.csv",
            "bucket/people.csv",
            filesystem_incremental=True,
            incremental_key=None,
        )

    pipeline = dlt.pipeline(
        pipeline_name="filesystem_option_ownership",
        destination=dlt.destinations.duckdb(str(database)),
        dataset_name="out",
        pipelines_dir=str(tmp_path / "state"),
    )
    with mock.patch("fsspec.implementations.ftp.FTPFileSystem", MemoryBackedFTP):
        pipeline.run(build_source(), table_name="people")
        second = pipeline.run(build_source(), table_name="people")

    assert second.load_packages == []

    connection = duckdb.connect(str(database))
    try:
        rows = connection.sql("select name from out.people").fetchall()
    finally:
        connection.close()
    assert rows == [("Alice",)]


def test_optional_reference_column_types(tmp_path):
    """A run that names no columns leaves the reference field unset."""
    case = Case("ftp", "ftp://user:pw@host/bucket/data.csv")
    reference, _ = _drive(case, tmp_path, column_types=None)

    assert reference.column_types is None
