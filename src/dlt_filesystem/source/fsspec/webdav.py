import re
from typing import Any, Dict, Optional, Type
from urllib.parse import quote, unquote, urlsplit, urlunsplit

from fsspec import AbstractFileSystem
from webdav4.client import Client
from webdav4.fsspec import WebdavFileSystem

from dlt_filesystem.error import MissingConnectorOption
from dlt_filesystem.source.base import FilesystemSource
from dlt_filesystem.source.core import infer_resource
from dlt_filesystem.source.impl.util import strip_protocol_suffix
from dlt_filesystem.source.model import FilesystemLocator, ResourceOptions
from dlt_filesystem.util.python import asbool, cast_to_bool, cast_to_dict, cast_to_int


class WebdavPathClient(Client):
    """Escape decoded filesystem paths when constructing a request URL."""

    def join_url(self, path, add_trailing_slash=False):
        return super().join_url(
            quote(path, safe="/"), add_trailing_slash=add_trailing_slash
        )


class WebdavPathFileSystem(WebdavFileSystem):
    """Translate HTTP URLs to paths relative to the WebDAV server root."""

    root_marker = "/"

    def __init__(self, base_url, auth=None, client=None, **client_opts):
        client_opts.setdefault("chunk_size", self.blocksize)
        if client is None:
            client = WebdavPathClient(base_url, auth=auth, **client_opts)
        super().__init__(base_url, auth=auth, client=client, **client_opts)

    @classmethod
    def _strip_protocol(cls, path: str) -> str:
        parsed = urlsplit(path)
        if parsed.scheme in {"http", "https"}:
            path = unquote(parsed.path)
        else:
            path = super()._strip_protocol(path)
        return "/" + path.strip("/")

    def info(self, path: str, **kwargs):
        info = super().info(path, **kwargs)
        info["name"] = self._strip_protocol(info["name"])
        return info

    def ls(self, path: str, detail: bool = True, **kwargs):
        items = super().ls(path, detail=detail, **kwargs)
        return [
            self._strip_protocol(item)
            if isinstance(item, str)
            else {**item, "name": self._strip_protocol(item["name"])}
            for item in items
        ]

    def glob(self, path, maxdepth=None, **kwargs):
        # Match decoded names, retaining the distinction between URI-escaped
        # literal glob characters and the pattern's operators.
        parsed = urlsplit(path)
        pattern = parsed.path if parsed.scheme in {"http", "https"} else path
        literals = {"2a": "[*]", "3f": "[?]", "5b": "[[]", "5d": "[]]"}
        pattern = re.sub(
            r"%(2a|3f|5b|5d)",
            lambda match: literals[match.group(1).lower()],
            pattern,
            flags=re.IGNORECASE,
        )
        matches = super().glob(unquote(pattern), maxdepth=maxdepth, **kwargs)
        # The lister reconstructs URLs from these paths, so escape only after
        # matching; requests decode the URL once at _strip_protocol above.
        if isinstance(matches, dict):
            return {quote(name, safe="/"): info for name, info in matches.items()}
        return [quote(name, safe="/") for name in matches]


class WebdavSource(FilesystemSource):
    """
    Access files on WebDAV.

    https://skshetry.github.io/webdav4/
    https://en.wikipedia.org/wiki/WebDAV
    """

    @property
    def fs_class(self) -> Type["AbstractFileSystem"]:
        return WebdavPathFileSystem

    def dlt_source(
        self,
        uri: str,
        table: str,
        *,
        filesystem_incremental: bool = False,
        column_types: Optional[Dict[str, Any]] = None,
        reader_hints: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):

        # This adjustment is specific to WebDAV.
        # This package addresses WebDAV as `https+webdav://`; fsspec uses `https://`.
        uri = strip_protocol_suffix(uri, "dav", "webdav")

        # Bundle essential information to infer filesystem wrapper.
        locator = FilesystemLocator(
            name="WebDAV", fs_class=self.fs_class, uri=uri, path=table
        )

        # Decode individual arguments.
        resource_options = ResourceOptions(
            filesystem_incremental=filesystem_incremental,
            column_types=column_types,
            reader_hints=reader_hints,
        )
        fs_kwargs = locator.options.fs_kwargs
        fs_kwargs.update(kwargs)
        cast_to_bool(fs_kwargs, ["retry", "trust_env"])
        cast_to_dict(fs_kwargs, ["headers", "cookies", "proxies"])
        cast_to_int(fs_kwargs, ["chunk_size", "max_redirects"])
        if "verify" in fs_kwargs:
            try:
                fs_kwargs["verify"] = asbool(fs_kwargs["verify"])
            except ValueError:
                pass

        # Extract authentication credentials.
        auth = None
        if "username" in fs_kwargs:
            auth = (fs_kwargs.pop("username"), fs_kwargs.pop("password", None))

        # Sanity checks.
        if "host" not in fs_kwargs or not fs_kwargs["host"]:
            raise MissingConnectorOption("host", "WebDAV")

        # Downstream implementation does not accept those kwargs.
        fs_kwargs.pop("host", None)
        fs_kwargs.pop("port", None)

        # Create filesystem and dlt resource wrapper.
        parsed_uri = urlsplit(uri)
        base_url = urlunsplit((parsed_uri.scheme, parsed_uri.netloc, "", "", ""))
        fs = self.fs_class(base_url, auth=auth, **fs_kwargs)
        return infer_resource(fs=fs, locator=locator, options=resource_options)
