from typing import Any, Dict, Optional, Type
from urllib.parse import urlsplit, urlunsplit

from fsspec import AbstractFileSystem
from webdav4.fsspec import WebdavFileSystem

from dlt_filesystem.error import MissingConnectorOption
from dlt_filesystem.source.base import FilesystemSource
from dlt_filesystem.source.core import infer_resource
from dlt_filesystem.source.impl.util import strip_protocol_suffix
from dlt_filesystem.source.model import FilesystemLocator, ResourceOptions
from dlt_filesystem.util.python import asbool, cast_to_bool, cast_to_dict, cast_to_int


class WebdavPathFileSystem(WebdavFileSystem):
    """Translate HTTP URLs to paths relative to the WebDAV server root."""

    root_marker = "/"

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
