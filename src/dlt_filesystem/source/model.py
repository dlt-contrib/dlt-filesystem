import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Type, Union
from urllib.parse import parse_qs

from dlt.common.configuration import configspec, resolve_type
from dlt.common.configuration.specs import CredentialsConfiguration
from dlt.common.storages import FilesystemConfiguration
from dlt.common.storages.configuration import FileSystemCredentials
from fsspec import AbstractFileSystem

from dlt_filesystem.error import InvalidBlobTableError
from dlt_filesystem.util.fsspec import infer_storage_options
from dlt_filesystem.util.web import shrink_qs_dict


@configspec
class FilesystemConfigurationResource(FilesystemConfiguration):
    credentials: Optional[Union[FileSystemCredentials, AbstractFileSystem]] = None
    file_glob: Optional[str] = "*"
    files_per_page: int = 100
    extract_content: bool = False

    @resolve_type("credentials")
    def resolve_credentials_type(self) -> Type[CredentialsConfiguration]:
        # use known credentials or empty credentials for unknown protocol
        return Union[  # ty: ignore[invalid-return-type]
            self.PROTOCOL_CREDENTIALS.get(self.protocol)  # ty: ignore[invalid-type-form]
            or Optional[CredentialsConfiguration],
            AbstractFileSystem,
        ]


#: Run-level parameters omniload passes into every source's ``dlt_source``.
#:
#: These name the run, not the storage service: they arrive from CLI flags and
#: `run_ingest` arguments rather than from the source URI. Keeping them in one place
#: is what lets `split_run_options` subtract them; the set must stay in step with the
#: keywords `omniload.api` actually passes, or a new run parameter starts leaking
#: silently, so a test compares the two.
RUN_OPTION_KEYS: frozenset = frozenset(
    {
        "column_types",
        "data_item_format",
        "extract_parallelism",
        "filesystem_incremental",
        "incremental_key",
        "interval_end",
        "interval_start",
        "merge_key",
        "page_size",
        "requested_incremental_key",
        "requested_primary_key",
        "sql_backend",
        "sql_exclude_columns",
        "sql_limit",
        "sql_reflection_level",
    }
)


def strip_run_options(options: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``options`` without any name omniload owns.

    The query string is the second carrier a run option can arrive on, so it gets the
    same ownership rule `split_run_options` applies to the run itself: a name in
    `RUN_OPTION_KEYS` never reaches a filesystem constructor, whichever carrier it
    came in on. `column_types` remains readable from `FilesystemOptions.params`,
    which is where the reference picks it up.
    """
    return {key: value for key, value in options.items() if key not in RUN_OPTION_KEYS}


class QueryMode(str, Enum):
    """What a source URI's query string means.

    Two transports disagree about this, and no rule derived from the URI can tell
    them apart, so each source states which it is.

    - `CONNECTOR_OPTIONS` reads the query as arguments for the connection:
      `s3://bucket/data.csv?access_key_id=...` names a credential, not a file. This
      is the family default and every existing scheme keeps it.
    - `ADDRESS` reads the query as part of the resource's address, to be sent
      verbatim: an HTTP URL's `?X-Amz-Signature=...` *is* the authorization, and
      dropping it or reading it as a constructor argument makes the URL unusable.

    The distinction cannot be made by inspecting the parsed protocol, because
    WebDAV rewrites `http+webdav` to `http` (`fsspec/webdav.py`) and deliberately
    wants its query read as connector options.
    """

    CONNECTOR_OPTIONS = "connector_options"
    ADDRESS = "address"


@dataclass
class FilesystemOptions:
    """Bundle filesystem options from URL."""

    address: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)

    @property
    def fs_kwargs(self) -> Dict[str, Any]:
        """Return inferred storage options modulo `protocol` and `path` fields."""
        # FIXME: Review using URL options as a baseline.
        response = deepcopy(self.address)
        # Remove certain options like `_get_kwargs_from_urls` is doing it.
        response.pop("path", None)
        response.pop("protocol", None)
        response.update(self.params)
        return strip_run_options(response)


@dataclass
class ResourceOptions:
    """The run options the filesystem package itself consumes.

    Every other run option is none of this package's business, and every field
    here belongs to the resource rather than to the filesystem: they are read
    when building the `FilesystemReference`, never when constructing an fsspec
    filesystem.
    """

    filesystem_incremental: bool = False
    column_types: Optional[Dict[str, Any]] = None
    #: Reader arguments a source derived from something other than the URI
    #: fragment, e.g. a programmatic `chunksize=`. Merged over `locator.hints`,
    #: so a per-URI hint stays the more specific of the two.
    reader_hints: Optional[Dict[str, Any]] = None


def split_run_options(kwargs: Dict[str, Any]) -> tuple[ResourceOptions, Dict[str, Any]]:
    """Separate omniload's run options from the connector's own keyword arguments.

    This is the ownership boundary for the filesystem family. `filesystem_incremental`
    and `column_types` are **resource** options: they configure how the reader
    resource is built and must reach `FilesystemReference`. Every other member of
    `RUN_OPTION_KEYS` is consumed elsewhere in the pipeline and is meaningless to a
    filesystem. Neither group is a **connector** option, so none of them may reach an
    fsspec constructor: an unknown keyword there fragments fsspec's instance cache
    (its key is built from the constructor arguments) and a backend that forwards
    unknown keywords into a client library, as `SFTPFileSystem` does into
    `paramiko.SSHClient.connect`, rejects them outright.

    The split is **subtractive**: it removes the pinned omniload keys and returns
    everything else untouched. A whitelist would be wrong, because a programmatic
    caller may legitimately pass connector keywords straight through, which GCS
    supports today by honouring a caller-supplied ``token``.

    S3 (`s3_filesystem_kwargs`), SFTP (its explicit ``params`` dict) and the local
    source already build their constructor arguments explicitly and take nothing from
    the run; this generalizes that shape to the connectors that merged the run
    parameters wholesale instead.

    Args:
        kwargs: The keyword arguments a `dlt_source` implementation received.

    Returns:
        The resource options to thread into `FilesystemReference`, and the remaining
        keyword arguments, which are safe to pass to a filesystem constructor.
    """
    resource_options = ResourceOptions(
        filesystem_incremental=kwargs.get("filesystem_incremental", False),
        column_types=kwargs.get("column_types"),
    )
    connector_kwargs = {
        key: value for key, value in kwargs.items() if key not in RUN_OPTION_KEYS
    }
    return resource_options, connector_kwargs


@dataclass
class FilesystemLocator:
    """A full filesystem information locator."""

    # FIXME: Get rid of inline imports by applying another round of refactoring.

    name: str
    fs_class: Type[AbstractFileSystem]
    uri: str
    path: str
    default_port: Optional[int] = None
    address: Dict[str, str] = field(default_factory=dict)
    options: FilesystemOptions = field(default_factory=FilesystemOptions)
    accept_no_bucket_name: Optional[bool] = False
    accept_no_host_name: Optional[bool] = False
    query_mode: QueryMode = QueryMode.CONNECTOR_OPTIONS
    #: The URI query, verbatim, when `query_mode` is `ADDRESS`; empty otherwise.
    #: Never merged into `fs_kwargs`, and never part of `file_glob`, so identity,
    #: reader selection and error messages all stay free of it.
    query: str = field(init=False, default="")

    def __post_init__(self):
        """Decode fundamental options right away."""
        self.read_options()

    def read_options(self) -> "FilesystemLocator":
        """
        Destructure input URL as a baseline for fsspec kwargs.

        Let's use the fsspec approach of decoding
        URIs, based on `fsspec.utils.infer_storage_options`.
        """

        self.address = infer_storage_options(self.uri)

        # URL query parameters, read as whichever of the two things they are.
        url_query = self.address.pop("url_query", "")
        # TODO: Why not compute hints right here instead of doing it at runtime?
        self.address.pop("url_fragment", None)

        if self.query_mode is QueryMode.ADDRESS:
            self.query = url_query
            params: Dict[str, Any] = {}
        else:
            params = shrink_qs_dict(parse_qs(url_query))

        # Reader or writer hints.
        self.options = FilesystemOptions(
            address=self.address,
            params=params,
        )
        return self

    def validate(self):
        """Decode into base url and url path / file glob, and apply sanity checks."""
        if not self.bucket_name or not self.file_glob:
            if self.accept_no_bucket_name:
                return
            # TODO: Rename exception.
            raise InvalidBlobTableError(self.name)

    @property
    def bucket_url(self) -> str:
        """URL without credentials and path."""

        if self.accept_no_host_name:
            return self.uri

        address = self.options.address
        if "host" in address and ("port" in address or self.default_port is not None):
            return f"{address['protocol']}://{address['host']}:{address.get('port', self.default_port)}"
        elif "host" in address:
            return f"{address['protocol']}://{address['host']}"

        # dlt will fail per `verify_bucket_url()` when no netloc or path is given:
        #   dlt.common.configuration.exceptions.ConfigurationValueError: File `path`
        #   and `netloc` are missing. Field `bucket_url` of `FilesystemClientConfiguration`
        #   must contain valid url with a path or host:password component.
        # When that happens, try to borrow a hostname from other suitable parameters
        # like `endpoint`.
        else:
            surrogate_host_port = self.options.params.get("endpoint")
            if not surrogate_host_port:
                raise ValueError("dlt needs bucket_url to include netloc and path")
            return f"{address['protocol']}://{surrogate_host_port}"

    @property
    def bucket_name(self) -> str:
        """URL component that describes the bucket name."""
        # FIXME: Inline imports!
        from dlt_filesystem.source.router import parse_uri

        bucket_name, _ = parse_uri(self.uri, self.path)
        return bucket_name

    @property
    def file_glob(self) -> str:
        """URL path component that describes the file glob."""
        # FIXME: Inline imports!
        from dlt_filesystem.source.router import parse_uri

        _, file_glob = parse_uri(self.uri, self.path)
        return file_glob

    @property
    def require_file_match(self) -> bool:
        """Whether the locator's raw carrier names one concrete file."""
        from dlt_filesystem.source.router import source_selects_single_file

        return source_selects_single_file(self.uri, self.path)

    @property
    def format_hint(self) -> Optional[str]:
        """The ``#format`` token the source selection names, if it names one.

        Read off the same carrier as :attr:`hints`, so a URI-borne ``#csv`` and a
        URI-borne ``#sheet_name=foo`` are honoured on the same terms.
        """
        from urllib.parse import urlparse

        from dlt_filesystem.source.router import blob_directives

        return blob_directives(urlparse(self.uri), self.path)[0]

    @property
    def hints(self) -> Dict[str, Any]:
        """
        Destructure reader or writer hints from URL fragment.

        Let's use the omniload approach of decoding
        URL fragments, because it handles a few edge cases, also taking
        the URL path into consideration.

        TODO: Refactor inline imports!
        """
        from urllib.parse import urlparse

        from dlt_filesystem.source.router import blob_hints

        parsed_uri = urlparse(self.uri)
        return blob_hints(parsed_uri, self.path)


@dataclass
class FilesystemReference:
    """
    Bundle the arguments needed by `resource_for_reader` to build a resource.

    Args:
        fs (AbstractFilesystem): fsspec filesystem instance.
        bucket_url (str): The url to the bucket.
        file_glob (str): The filter to apply to the files in glob format.
        reader_name (str): The name of the reader resource to build, e.g. `read_csv`.
        storage_namespace (str): Secret-free identity for the storage service or
            endpoint. Defaults to ``filesystem`` for callers that do not need to
            distinguish transports. The bucket URL and glob are added separately
            when deriving the incremental resource-state key.
        filesystem_incremental (bool): Whether to filter files using their
            modification time and persistent dlt resource state. A **resource**
            option owned by this reference, never an fsspec constructor argument;
            see `split_run_options` for the boundary that keeps it here.
        require_file_match (bool): Whether extraction must fail when the concrete
            source selection matches no file.
        hints (dict[str, str]): Free-form per-URI reader hints parsed from the
            `#key=value` fragment (e.g. `{"sheet_name": "ticker-symbols"}`). The
            key a reader looks up is that reader's contract; no reader consumes
            hints yet, so this is currently populated but unread.
        column_types (dict[str, Any], optional): Column name to type mapping, e.g. used by
            `read_csv_headless`. A **resource** option like `filesystem_incremental`;
            it may arrive from the run (`--columns`) or from a URI query parameter, and
            the run value wins where both are given.

    TODO: Zap into / synchronize with the new `FilesystemLocator`?
    """

    fs: AbstractFileSystem
    bucket_url: str
    file_glob: str
    reader_name: str
    storage_namespace: str = "filesystem"
    filesystem_incremental: bool = False
    require_file_match: bool = False
    hints: dict[str, str] = field(default_factory=dict)
    column_types: Optional[dict[str, Any]] = None

    @property
    def incremental_resource_name(self) -> str:
        """Return a stable, secret-free resource name for this file selection."""
        identity = json.dumps(
            [self.storage_namespace, self.bucket_url, self.file_glob],
            ensure_ascii=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]
        return f"filesystem_{digest}"
