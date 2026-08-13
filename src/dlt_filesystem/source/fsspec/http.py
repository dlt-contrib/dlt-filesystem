"""Read a file addressed by an `http://` or `https://` URL.

An HTTP URL is unlike every other address in this family in one way that shapes
the whole module: its query string is part of the address, not a set of connection
arguments. `?X-Amz-Signature=...` *is* the authorization to read the object, so it
has to reach the wire byte for byte, and must not be read as a constructor
argument, folded into the file's identity, or printed in an error.

Three deviations from `fsspec`'s `HTTPFileSystem` follow from that and from
keeping every file the previous connector could read readable:

- the query is carried by the filesystem and re-attached where a URL goes on the
  wire, so the path the rest of the family sees stays query-free;
- URLs are escaped the way `requests` escaped them and handed to `yarl` already
  encoded, because `yarl`'s own normalization rewrites `%2F` as `/` and voids a
  signature;
- a server that cannot serve byte ranges, or reports no size at all, is read whole
  instead of failing, which is what the previous connector did for every server.
"""

import io
from typing import Any, Dict, Optional, Tuple, Type
from urllib.parse import unquote, urlsplit, urlunsplit

from fsspec import AbstractFileSystem
from fsspec.implementations.http import HTTPFileSystem

from dlt_filesystem.error import MissingConnectorOption
from dlt_filesystem.source.base import FilesystemSource
from dlt_filesystem.source.core import infer_resource
from dlt_filesystem.source.model import (
    FilesystemLocator,
    QueryMode,
    split_run_options,
)
from dlt_filesystem.util.python import cast_to_bool, cast_to_dict, cast_to_int
from dlt_filesystem.util.web import requote_uri

#: Connect and read timeouts, in seconds. Named after what they bound rather than
#: an overall budget: a total timeout would abort a large download that is making
#: perfectly good progress.
DEFAULT_TIMEOUT_SECONDS = 30


class HttpFileSystem(HTTPFileSystem):
    """`HTTPFileSystem` with the query as address, and no unreadable servers.

    Args:
        url_query: The source URI's query string, verbatim and already escaped.
            Re-attached to any URL that carries none of its own, which is every
            URL this filesystem is handed: the family composes paths and globs
            from a query-free locator on purpose.
    """

    def __init__(self, *args: Any, url_query: str = "", **kwargs: Any) -> None:
        # `encoded` is not optional here: it tells `yarl` the URL is already
        # escaped and must be sent as given. With the default, `yarl` unescapes
        # and re-escapes with its own rules, which turns `%2F` into `/` and makes
        # a signed URL fail authorization rather than fail loudly.
        kwargs["encoded"] = True
        super().__init__(*args, **kwargs)
        self.url_query = url_query
        self._range_readable: Dict[str, bool] = {}

    def encode_url(self, url: str) -> Any:
        """Return the URL to request: query re-attached, escaped as `requests` did.

        Every read path in `HTTPFileSystem` (`_exists`, `_info`, `_cat_file`, and
        both file classes) funnels through here, so this is the one place the
        address has to be complete.
        """
        return super().encode_url(requote_uri(self._addressed(url)))

    def _addressed(self, url: str) -> str:
        """Re-attach the source query to a URL that carries none."""
        if not self.url_query:
            return url
        scheme, netloc, path, query, fragment = urlsplit(url)
        if query:
            return url
        return urlunsplit((scheme, netloc, path, self.url_query, fragment))

    def _open(
        self,
        path: str,
        mode: str = "rb",
        block_size: Optional[int] = None,
        autocommit: Optional[bool] = None,
        cache_type: Optional[str] = None,
        cache_options: Optional[Dict[str, Any]] = None,
        size: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
        """Open a file, reading it whole when it cannot be read in ranges.

        `HTTPFileSystem` needs both a size and working range requests to hand back
        a seekable file. Without them it either raises on the second block ("the
        HTTP server doesn't appear to support range requests") or returns a
        stream that raises on the first seek, and a parquet footer read seeks
        immediately. Both servers worked before this connector joined the family,
        because the whole body was downloaded every time; reading them whole here
        keeps that true while everything else gains streaming.
        """
        if block_size == 0:
            # An explicit ask for the streaming interface; nothing to second-guess.
            return super()._open(
                path,
                mode=mode,
                block_size=block_size,
                autocommit=autocommit,
                cache_type=cache_type,
                cache_options=cache_options,
                size=size,
                **kwargs,
            )
        if self._serves_ranges(path):
            return super()._open(
                path,
                mode=mode,
                block_size=block_size,
                autocommit=autocommit,
                cache_type=cache_type,
                cache_options=cache_options,
                size=size,
                **kwargs,
            )
        return io.BytesIO(self.cat_file(path))

    def _serves_ranges(self, path: str) -> bool:
        """Ask the server, once per URL, whether it answers a range request.

        One byte is requested and the answer is read from the status: a `206` with
        a numeric total in `Content-Range` means the server both honours ranges and
        knows how long the resource is, which is exactly what a seekable file needs.
        A `200` means the range was ignored, and no total means no size, so neither
        can be read in ranges.

        The alternative -- reading `Accept-Ranges` from the `HEAD` the listing
        already made -- does not work: a server that ignores `Range` typically says
        nothing about ranges at all rather than advertising `none`.
        """
        from fsspec.asyn import sync

        cached = self._range_readable.get(path)
        if cached is not None:
            return cached

        readable = bool(sync(self.loop, self._probe_range, path))
        self._range_readable[path] = readable
        return readable

    async def _probe_range(self, path: str) -> bool:
        request_kwargs = {
            key: value for key, value in self.kwargs.items() if key != "headers"
        }
        headers = dict(self.kwargs.get("headers") or {})
        headers["Range"] = "bytes=0-0"
        # An identity encoding, so a compressing proxy cannot make a partial
        # response look like a whole one.
        headers["Accept-Encoding"] = "identity"

        session = await self.set_session()
        response = await session.get(
            self.encode_url(path), headers=headers, **request_kwargs
        )
        async with response:
            if response.status != 206:
                return False
            total = response.headers.get("Content-Range", "").rpartition("/")[2]
            return total.isdigit()


class HttpFilesystemSource(FilesystemSource):
    """
    Access files over HTTP and HTTPS.

    https://developer.mozilla.org/en-US/docs/Web/HTTP
    """

    @property
    def fs_class(self) -> Type["AbstractFileSystem"]:
        return HttpFileSystem

    def supports_filesystem_incremental(self) -> bool:
        """Return `False`: an HTTP response need not carry a modification time.

        dlt's own per-scheme extractor substitutes *now* for a missing
        `Last-Modified` header, so file selection by modification time would not
        select anything -- it would reload every file on every run while reporting
        that it had filtered. Refused until the transport can answer honestly.
        """
        return False

    def dlt_source(self, uri: str, table: str, **kwargs):
        if kwargs.get("incremental_key"):
            raise ValueError(
                "HTTP takes care of incrementality on its own, you should not provide incremental_key"
            )
        if kwargs.get("filesystem_incremental"):
            raise ValueError(
                "HTTP does not support file selection by modification time: a "
                "response need not carry a 'Last-Modified' header, and a missing "
                "one reads as 'just now', so every file would be reloaded on "
                "every run. Omit '--filesystem-incremental'."
            )

        # A programmatic caller may name the format and the reader's chunk size
        # directly. Both are reader arguments rather than connection arguments, so
        # they are translated into the channels the family reads them from and
        # never reach the filesystem constructor, which would forward an unknown
        # keyword into aiohttp and fail the request.
        file_format = kwargs.pop("file_format", None)
        chunksize = kwargs.pop("chunksize", None)
        if file_format:
            table = f"{table}#{file_format}"

        locator = FilesystemLocator(
            name="HTTP",
            fs_class=self.fs_class,
            uri=uri,
            path=table,
            query_mode=QueryMode.ADDRESS,
        )

        resource_options, connector_kwargs = split_run_options(kwargs)
        if chunksize is not None:
            resource_options.reader_hints = {"chunksize": int(chunksize)}

        fs_kwargs = locator.options.fs_kwargs
        fs_kwargs.update(connector_kwargs)
        cast_to_bool(fs_kwargs, ["simple_links", "same_scheme"])
        cast_to_dict(fs_kwargs, ["headers", "client_kwargs"])
        cast_to_int(fs_kwargs, ["block_size"])

        if not fs_kwargs.get("host"):
            raise MissingConnectorOption("host", "HTTP")

        # Addressing information, not connection arguments: the URL already
        # carries the host, port and credentials, and aiohttp accepts none of them.
        fs_kwargs.pop("host", None)
        fs_kwargs.pop("port", None)
        credentials = self._credentials(fs_kwargs)

        client_kwargs = dict(fs_kwargs.pop("client_kwargs", None) or {})
        self._apply_network_defaults(client_kwargs, credentials)

        fs = self.fs_class(
            url_query=locator.query, client_kwargs=client_kwargs, **fs_kwargs
        )
        return infer_resource(fs=fs, locator=locator, options=resource_options)

    @staticmethod
    def _credentials(fs_kwargs: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        """Take the URL's userinfo out of the connector arguments, decoded.

        A password has to be percent-encoded to survive a URI at all (`@` would end
        the userinfo, `/` the netloc), and the parser hands back the raw substring,
        so forwarding it unchanged authenticates as the wrong user.
        """
        username = fs_kwargs.pop("username", None)
        password = fs_kwargs.pop("password", None)
        if not username:
            return None
        return unquote(username), unquote(password or "")

    @staticmethod
    def _apply_network_defaults(
        client_kwargs: Dict[str, Any], credentials: Optional[Tuple[str, str]]
    ) -> None:
        """Pin the session behaviour, keeping what a caller asked for.

        aiohttp defaults differ from the ones this source has always had: it
        applies a five-minute budget to the whole exchange and ignores the
        environment's proxy and `.netrc` settings. Both are set back here so that
        moving from one HTTP client to another does not silently change which
        requests can be made or when they give up.
        """
        import aiohttp

        client_kwargs.setdefault("trust_env", True)
        client_kwargs.setdefault(
            "timeout",
            aiohttp.ClientTimeout(
                total=None,
                sock_connect=DEFAULT_TIMEOUT_SECONDS,
                sock_read=DEFAULT_TIMEOUT_SECONDS,
            ),
        )
        if credentials is not None:
            # Sent as a session header rather than aiohttp's `auth=`, which is
            # deprecated, and rather than a per-request keyword, which would put
            # the credential in the arguments every request logs its kwargs from.
            headers = dict(client_kwargs.get("headers") or {})
            headers.setdefault("Authorization", aiohttp.encode_basic_auth(*credentials))
            client_kwargs["headers"] = headers
