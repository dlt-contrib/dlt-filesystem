from typing import Any, Dict, Optional, Type

from dropboxdrivefs import DropboxDriveFileSystem
from fsspec import AbstractFileSystem

from dlt_filesystem.source.base import FilesystemSource
from dlt_filesystem.source.core import infer_resource
from dlt_filesystem.source.model import FilesystemLocator, ResourceOptions


class DropboxFileSystem(DropboxDriveFileSystem):
    """Use absolute Dropbox paths for discovery and file reads."""

    protocol = "dropbox"
    root_marker = "/"

    @classmethod
    def _strip_protocol(cls, path):
        return "/" + super()._strip_protocol(path).lstrip("/")

    def info(self, url, **kwargs):
        # The backend's info and ls bypass fsspec's protocol stripping.
        return super().info(self._strip_protocol(url), **kwargs)

    def ls(self, path, detail=True, recursive=False, **kwargs):
        path = self._strip_protocol(path)
        # Dropbox addresses the account's root folder with an empty string.
        listing = super().ls(
            "" if path == "/" else path, detail=detail, recursive=recursive, **kwargs
        )
        # The backend logs SDK errors and returns None instead of a listing.
        if listing is None:
            raise RuntimeError(f"Dropbox directory listing failed for {path}")
        return listing

    def _refactor_metadata(self, metadata, detail=True):
        """Expose the server's modification time under the shared lister's key."""
        info = super()._refactor_metadata(metadata, detail=detail)
        if detail and info["type"] == "file":
            info["mtime"] = info["server_modified"]
        return info


class DropboxSource(FilesystemSource):
    """
    Access files on Dropbox.

    https://github.com/fsspec/dropboxdrivefs
    """

    @property
    def fs_class(self) -> Type["AbstractFileSystem"]:
        return DropboxFileSystem

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

        # Bundle essential information to infer filesystem wrapper.
        locator = FilesystemLocator(
            name="Dropbox", fs_class=self.fs_class, uri=uri, path=table
        )

        # Decode individual options (type casting, default values, sanity checks). Schema:
        resource_options = ResourceOptions(
            filesystem_incremental=filesystem_incremental,
            column_types=column_types,
            reader_hints=reader_hints,
        )
        fs_kwargs = locator.options.fs_kwargs
        fs_kwargs.update(kwargs)

        # Create filesystem and dlt resource wrapper.
        fs = self.fs_class(**fs_kwargs)
        return infer_resource(fs=fs, locator=locator, options=resource_options)
