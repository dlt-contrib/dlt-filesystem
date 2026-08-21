import base64
import json
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlparse, urlsplit

from dlt_filesystem.error import MissingConnectorOption

AZURE_SERVICE_PRINCIPAL_FIELDS = ("tenant_id", "client_id", "client_secret")
#: The service labels an Azure endpoint suffix carries, which arrow keeps apart.
AZURE_BLOB_LABEL = ".blob."
AZURE_DFS_LABEL = ".dfs."


def _first(params: dict[str, list[str]], key: str) -> Optional[str]:
    """Return the first query-parameter value, matching existing URI semantics."""
    return params.get(key, [None])[0]


def s3_filesystem_kwargs(
    params: dict[str, list[str]], connector: str = "S3"
) -> dict[str, Any]:
    """Translate omniload S3 URI parameters into ``s3fs`` arguments."""
    access_key_id = _first(params, "access_key_id")
    if not access_key_id:
        raise MissingConnectorOption("access_key_id", connector)

    secret_access_key = _first(params, "secret_access_key")
    if not secret_access_key:
        raise MissingConnectorOption("secret_access_key", connector)

    kwargs: dict[str, Any] = {
        "key": access_key_id,
        "secret": secret_access_key,
        # S3FileSystem caches directory listings by default. Disable the cache so
        # long-lived processes see objects created between incremental runs.
        "use_listings_cache": False,
    }
    endpoint_url = _first(params, "endpoint_url")
    if endpoint_url:
        kwargs["endpoint_url"] = endpoint_url
    region = _first(params, "region")
    if region:
        kwargs["client_kwargs"] = {"region_name": region}
    return kwargs


def s3_arrow_filesystem_kwargs(
    params: dict[str, list[str]], connector: str = "S3"
) -> dict[str, Any]:
    """Translate omniload S3 URI parameters into ``pyarrow.fs`` arguments."""
    access_key_id = _first(params, "access_key_id")
    if not access_key_id:
        raise MissingConnectorOption("access_key_id", connector)

    secret_access_key = _first(params, "secret_access_key")
    if not secret_access_key:
        raise MissingConnectorOption("secret_access_key", connector)

    kwargs: dict[str, Any] = {
        "access_key": access_key_id,
        "secret_key": secret_access_key,
    }

    endpoint_url = _first(params, "endpoint_url")
    if endpoint_url:
        endpoint = urlparse(endpoint_url)
        if (
            endpoint.scheme not in {"http", "https"}
            or not endpoint.hostname
            or endpoint.username is not None
            or endpoint.password is not None
        ):
            raise ValueError(
                "Invalid endpoint_url. Must be an HTTP or HTTPS URL with a host."
            )
        if (
            endpoint.path.rstrip("/")
            or endpoint.params
            or endpoint.query
            or endpoint.fragment
        ):
            raise ValueError(
                "Invalid endpoint_url. Arrow S3 endpoints must not include a path, "
                "query, or fragment."
            )
        kwargs["scheme"] = endpoint.scheme
        kwargs["endpoint_override"] = endpoint.netloc

    region = _first(params, "region")
    if region:
        kwargs["region"] = region
    return kwargs


def gcs_filesystem_kwargs(
    params: dict[str, list[str]], inherited: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Translate omniload GCS URI parameters into ``gcsfs`` arguments."""
    kwargs = dict(inherited or {})
    remaining = {key: list(values) for key, values in params.items()}
    credentials_path = _first(remaining, "credentials_path")
    credentials_base64 = _first(remaining, "credentials_base64")
    remaining.pop("credentials_path", None)
    remaining.pop("credentials_base64", None)

    for key, values in remaining.items():
        kwargs.setdefault(key, values[0])

    if "token" not in kwargs:
        if credentials_path:
            kwargs["token"] = credentials_path
        elif credentials_base64:
            kwargs["token"] = json.loads(base64.b64decode(credentials_base64).decode())
        else:
            kwargs["token"] = "anon"  # noqa: S105 - gcsfs anonymous-access sentinel
    return kwargs


@dataclass
class AzureBlobAuth:
    """Resolved Azure blob-storage credentials parsed from URI query params.

    Holds the ingestr-style short names (``account_name`` / ``account_key`` /
    ``sas_token`` / ``tenant_id`` / ``client_id`` / ``client_secret`` /
    ``account_host``), which three consumers then map their own way: the source
    onto ``pyarrow.fs.AzureFileSystem`` arguments, remote-database staging onto
    ``adlfs.AzureBlobFileSystem`` kwargs (whose names these match), and the
    destination onto dlt's ``AzureCredentials`` /
    ``AzureServicePrincipalCredentials`` spec fields. A connection string also
    resolves account and endpoint identity here; the credential it carries is
    read where a client needs it as a field, and adlfs keeps receiving the
    original string.
    """

    account_name: Optional[str] = None
    account_key: Optional[str] = field(default=None, repr=False)
    sas_token: Optional[str] = field(default=None, repr=False)
    tenant_id: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = field(default=None, repr=False)
    account_host: Optional[str] = None
    #: The Data Lake endpoint a connection string names, when it names one. It is
    #: never a URI parameter, and it stays out of the storage namespace, which
    #: identifies an account by its Blob endpoint.
    dfs_host: Optional[str] = None
    connection_string: Optional[str] = field(default=None, repr=False)
    api_version: Optional[str] = None

    @property
    def is_service_principal(self) -> bool:
        """True when a full service-principal triplet is present.

        Derived from all three fields (not just ``tenant_id``) so it stays
        honest even if an ``AzureBlobAuth`` is built outside ``parse_azure_blob_auth``.
        """
        return (
            self.tenant_id is not None
            and self.client_id is not None
            and self.client_secret is not None
        )


@dataclass(frozen=True)
class _AzureConnectionString:
    """What one Azure connection string resolves to, read through public SDK APIs.

    The account and endpoints identify the storage the string addresses, and are
    secret-free; the credential is read here too, because a client that takes no
    connection string needs it as a field.
    """

    account_name: Optional[str]
    blob_endpoint: str
    dfs_endpoint: Optional[str] = None
    account_key: Optional[str] = field(default=None, repr=False)
    sas_token: Optional[str] = field(default=None, repr=False)


def _parse_azure_connection_string(connection_string: str) -> _AzureConnectionString:
    """Resolve one connection string through Azure's public SDK APIs, once.

    ``BlobServiceClient.from_connection_string`` performs no network I/O and is
    authoritative for the composed endpoint and the credential;
    ``parse_connection_string`` supplies the case-insensitive field mapping,
    which carries the account name for the endpoint forms the client leaves it
    unset on, and the Data Lake endpoint the client models no other way.
    """
    from azure.core.utils import parse_connection_string
    from azure.storage.blob import BlobServiceClient

    try:
        settings = parse_connection_string(connection_string)
        client = BlobServiceClient.from_connection_string(connection_string)
    except ValueError:
        raise ValueError("Invalid Azure connection_string.") from None

    try:
        account_name = client.account_name or settings.get("accountname")
        composed = urlsplit(client.url)
        blob_endpoint = composed._replace(query="", fragment="").geturl()
        resolved = _AzureConnectionString(
            account_name=account_name,
            blob_endpoint=blob_endpoint,
            dfs_endpoint=settings.get("dfsendpoint"),
            account_key=getattr(client.credential, "account_key", None),
            sas_token=composed.query or None,
        )
    finally:
        client.close()

    if not resolved.blob_endpoint:
        raise ValueError("Invalid Azure connection_string.")
    return resolved


def parse_azure_blob_auth(params: dict) -> AzureBlobAuth:
    """Parse and validate Azure blob-storage credentials from URI query params.

    ``params`` is the ``urllib.parse.parse_qs`` output (each value is a list).
    Values must be URL-encoded in the URI: Azure account keys are base64
    (``+`` / ``/`` / ``=``) and SAS tokens embed their own ``&`` / ``=`` pairs,
    which ``parse_qs`` would otherwise mangle (``+`` becomes a space, an
    unencoded SAS token shatters into junk params).

    Three auth modes are supported:

    * connection string: ``connection_string``
    * account-key / SAS: ``account_key`` or ``sas_token``
    * service principal: the full ``tenant_id`` + ``client_id`` +
      ``client_secret`` triplet

    Raises:
        MissingValueError: if ``account_name`` is absent, if no auth material is
            supplied, or if the service-principal triplet is only partially
            supplied (naming the missing field(s)).
        ValueError: if mutually exclusive credentials are supplied together,
            or if a connection string is malformed.
    """

    connection_string = _first(params, "connection_string")
    api_version = _first(params, "api_version")
    account_name = _first(params, "account_name")
    account_key = _first(params, "account_key")
    sas_token = _first(params, "sas_token")
    sp_values = {
        field: _first(params, field) for field in AZURE_SERVICE_PRINCIPAL_FIELDS
    }
    account_host = _first(params, "account_host")

    if connection_string is not None:
        conflicting_fields = [
            field
            for field, value in {
                "account_name": account_name,
                "account_key": account_key,
                "sas_token": sas_token,
                "account_host": account_host,
                **sp_values,
            }.items()
            if value is not None
        ]
        if conflicting_fields:
            raise ValueError(
                "Conflicting Azure credentials: connection_string cannot be "
                f"combined with {', '.join(conflicting_fields)}."
            )
        resolved = _parse_azure_connection_string(connection_string)
        return AzureBlobAuth(
            account_name=resolved.account_name,
            account_key=resolved.account_key,
            sas_token=resolved.sas_token,
            account_host=resolved.blob_endpoint,
            dfs_host=resolved.dfs_endpoint,
            connection_string=connection_string,
            api_version=api_version,
        )

    if account_name is None:
        raise MissingConnectorOption("account_name", "Azure")

    if account_key is not None and sas_token is not None:
        raise ValueError(
            "Conflicting Azure credentials: supply either account_key or "
            "sas_token, not both."
        )

    has_shared_key = account_key is not None or sas_token is not None
    supplied_sp_fields = [f for f, v in sp_values.items() if v is not None]
    has_service_principal = len(supplied_sp_fields) > 0

    if has_shared_key and has_service_principal:
        raise ValueError(
            "Conflicting Azure credentials: supply either account_key/sas_token "
            "or the service-principal triplet (tenant_id, client_id, "
            "client_secret), not both."
        )

    if has_service_principal:
        missing = [f for f in AZURE_SERVICE_PRINCIPAL_FIELDS if sp_values[f] is None]
        if missing:
            raise MissingConnectorOption(", ".join(missing), "Azure service principal")
    elif not has_shared_key:
        raise MissingConnectorOption(
            "account_key, sas_token, or a service-principal triplet "
            "(tenant_id, client_id, client_secret)",
            "Azure",
        )

    return AzureBlobAuth(
        account_name=account_name,
        account_key=account_key,
        sas_token=sas_token,
        account_host=account_host,
        connection_string=connection_string,
        api_version=api_version,
        **sp_values,
    )


def azure_blob_filesystem_kwargs(auth: AzureBlobAuth) -> dict[str, Any]:
    """Translate parsed Azure credentials into ``adlfs`` arguments.

    The source reads through ``pyarrow.fs``, so this now serves the staging
    download that materializes a remote database file, which is neither source
    nor destination and has no Arrow client to build.
    """
    if auth.connection_string is not None:
        kwargs: dict[str, Any] = {
            "account_name": None,
            "connection_string": auth.connection_string,
        }
        if auth.api_version is not None:
            kwargs["api_version"] = auth.api_version
        return kwargs

    kwargs: dict[str, Any] = {"account_name": auth.account_name}
    if auth.account_key is not None:
        kwargs["account_key"] = auth.account_key
    if auth.sas_token is not None:
        kwargs["sas_token"] = auth.sas_token
    if auth.tenant_id is not None:
        kwargs["tenant_id"] = auth.tenant_id
    if auth.client_id is not None:
        kwargs["client_id"] = auth.client_id
    if auth.client_secret is not None:
        kwargs["client_secret"] = auth.client_secret
    if auth.account_host is not None:
        kwargs["account_host"] = auth.account_host
    if auth.api_version is not None:
        kwargs["api_version"] = auth.api_version
    return kwargs


def _azure_service_authorities(authority: str) -> tuple[str, str]:
    """Pair one endpoint authority with the sibling authority of the other service.

    Arrow addresses Blob and Data Lake Gen2 through separate endpoints, and
    resolves a hierarchical-namespace account through the Data Lake one, so
    naming only the Blob endpoint would send half the requests to the public
    cloud.

    A domain suffix (which Arrow reads as the part following the account name)
    yields its sibling by substituting the service label wherever it sits, since
    a Private Link name carries it a label in: `.privatelink.blob.core.windows.net`
    pairs with `.privatelink.dfs.core.windows.net`. A suffix naming no service, and
    a fully qualified host (an emulator, a gateway), serve both as they are.
    """
    if authority.startswith("."):
        for label, sibling in (
            (AZURE_BLOB_LABEL, AZURE_DFS_LABEL),
            (AZURE_DFS_LABEL, AZURE_BLOB_LABEL),
        ):
            head, found, tail = authority.partition(label)
            if found:
                swapped = f"{head}{sibling}{tail}"
                if label == AZURE_BLOB_LABEL:
                    return authority, swapped
                return swapped, authority
    return authority, authority


def _azure_authority_and_scheme(
    account_name: str, endpoint_value: str, field_name: str
) -> tuple[str, str]:
    """Read one Azure endpoint as the authority-and-scheme pair arrow takes.

    Arrow reads an authority that starts with a dot as a domain suffix to prepend
    the account name to, and any other authority as a fully qualified host the
    account name follows in the URL path. Both forms are derivable from an
    endpoint that names the account, which is the only shape any carrier
    produces: a bare ``account_host`` host, or an endpoint a connection string
    names or composes.
    """
    endpoint = urlsplit(
        endpoint_value if "://" in endpoint_value else f"//{endpoint_value}",
        scheme="https",
    )
    if (
        endpoint.scheme not in {"http", "https"}
        or not endpoint.hostname
        or endpoint.username is not None
        or endpoint.password is not None
    ):
        raise ValueError(
            f"Invalid {field_name}. Must be a hostname[:port], optionally "
            "prefixed with http:// or https://."
        )
    if endpoint.query or endpoint.fragment:
        raise ValueError(
            f"Invalid {field_name}. Azure endpoints must not include a query "
            "or fragment."
        )

    netloc = endpoint.netloc.lower()
    path = endpoint.path.strip("/")
    if path:
        parent, _, account_segment = path.rpartition("/")
        if account_segment.lower() != account_name.lower():
            raise ValueError(
                f"Invalid {field_name}. A path-style Azure endpoint must end in "
                f"the storage account name, but {account_segment!r} is not "
                f"{account_name!r}."
            )
        # Arrow composes `scheme://{authority}/{account}`, so carrying the
        # leading path segments in the authority reproduces the endpoint that
        # was given, whatever its depth.
        return (f"{netloc}/{parent}" if parent else netloc), endpoint.scheme

    prefix = f"{account_name.lower()}."
    if not netloc.startswith(prefix):
        raise ValueError(
            f"Invalid {field_name}. A host-style Azure endpoint must start in "
            f"the storage account name, but {netloc!r} does not start with "
            f"{prefix!r}."
        )
    return netloc[len(account_name) :], endpoint.scheme


def _azure_endpoint_kwargs(
    account_name: str, account_host: str, dfs_host: Optional[str] = None
) -> dict[str, Any]:
    """Translate an Azure account's endpoints into arrow's two authority pairs.

    A connection string may name the Data Lake endpoint outright, in which case
    it is used as given; otherwise it is derived from the Blob endpoint, which is
    all a single ``account_host`` can offer.
    """
    authority, blob_scheme = _azure_authority_and_scheme(
        account_name, account_host, "account_host"
    )
    blob_authority, derived_dfs_authority = _azure_service_authorities(authority)
    if dfs_host is not None:
        dfs_authority, dfs_scheme = _azure_authority_and_scheme(
            account_name, dfs_host, "DfsEndpoint"
        )
    else:
        dfs_authority, dfs_scheme = derived_dfs_authority, blob_scheme
    return {
        "blob_storage_authority": blob_authority,
        "blob_storage_scheme": blob_scheme,
        "dfs_storage_authority": dfs_authority,
        "dfs_storage_scheme": dfs_scheme,
    }


def _azure_sas_query(sas_token: str) -> str:
    """Return a SAS token in the leading-`?` form arrow requires.

    Arrow appends the token to the account URL verbatim, so a token without the
    delimiter lands in the URL path and every request fails as a malformed
    resource name. adlfs added the delimiter itself, and both carriers here yield
    the token without one: a URI parameter is the bare token, and a connection
    string's is read off the composed URL's query.
    """
    return f"?{sas_token.removeprefix('?')}"


def azure_arrow_filesystem_kwargs(auth: AzureBlobAuth) -> dict[str, Any]:
    """Translate parsed Azure credentials into ``pyarrow.fs`` arguments."""
    if auth.api_version is not None:
        raise ValueError(
            "api_version is not supported by the Azure source, which reads "
            "through pyarrow.fs. Arrow offers no API-version override, so "
            "remove the parameter from the URI."
        )
    if not auth.account_name:
        raise ValueError(
            "Azure needs an account name, which pyarrow.fs takes as the root of "
            "the filesystem. A connection string that omits AccountName "
            "identifies its account by endpoint alone, which arrow cannot "
            "address; name the account in the connection string, or supply "
            "account_name with account_key or sas_token."
        )

    kwargs: dict[str, Any] = {"account_name": auth.account_name}
    if auth.account_key is not None:
        kwargs["account_key"] = auth.account_key
    if auth.sas_token is not None:
        kwargs["sas_token"] = _azure_sas_query(auth.sas_token)
    if auth.tenant_id is not None:
        kwargs["tenant_id"] = auth.tenant_id
    if auth.client_id is not None:
        kwargs["client_id"] = auth.client_id
    if auth.client_secret is not None:
        kwargs["client_secret"] = auth.client_secret
    if auth.account_host is not None:
        kwargs.update(
            _azure_endpoint_kwargs(auth.account_name, auth.account_host, auth.dfs_host)
        )
    return kwargs
