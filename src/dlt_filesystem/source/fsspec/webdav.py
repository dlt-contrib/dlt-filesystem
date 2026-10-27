from typing import Any, Dict, List, Optional, Type, cast
from urllib.parse import quote, urlsplit, urlunsplit

from fsspec import AbstractFileSystem
from webdav4.client import Client
from webdav4.fsspec import WebdavFileSystem
from webdav4.multistatus import prepare_propfind_request_data

from dlt_filesystem.error import MissingConnectorOption
from dlt_filesystem.source.base import FilesystemSource
from dlt_filesystem.source.core import infer_resource
from dlt_filesystem.source.impl.util import strip_protocol_suffix
from dlt_filesystem.source.model import FilesystemLocator, ResourceOptions
from dlt_filesystem.util.python import asbool, cast_to_bool, cast_to_dict, cast_to_int


class WebdavPathClient(Client):
    """Match property responses by the decoded path of the requested URL."""

    def get_props(self, path, name=None, namespace=None, data=None):
        data = data or prepare_propfind_request_data(name, namespace)
        headers = {"Content-Type": "application/xml"} if data else {}
        result = self.propfind(path, headers=headers, data=data)
        return result.responses[self.join_url(path).path.rstrip("/") or "/"].properties


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
            path = parsed.path
        else:
            path = super()._strip_protocol(path)
        return "/" + path.strip("/")

    def info(self, path: str, **kwargs):
        info = super().info(path, **kwargs)
        info["name"] = self._strip_protocol(quote(info["name"], safe="/"))
        return info

    def ls(self, path: str, detail: bool = True, **kwargs):
        # Keep names escaped through URL reconstruction and the client's join_url.
        # A decoded '#' or '?' would otherwise become a fragment or query.
        items = cast(List[Dict[str, Any]], super().ls(path, detail=True, **kwargs))
        listing = [
            {**item, "name": self._strip_protocol(quote(item["name"], safe="/"))}
            for item in items
        ]
        return listing if detail else [item["name"] for item in listing]


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
