from typing import Any, Dict, Optional, Type

from fsspec import AbstractFileSystem
from fsspec.implementations.webhdfs import WebHDFS

from dlt_filesystem.error import MissingConnectorOption
from dlt_filesystem.source.base import FilesystemSource
from dlt_filesystem.source.core import infer_resource
from dlt_filesystem.source.model import FilesystemLocator, ResourceOptions
from dlt_filesystem.util.python import (
    apply_alias,
    cast_to_bool,
    cast_to_dict,
    cast_to_int,
)


class WebHDFSFileSystem(WebHDFS):
    """Address every WebHDFS path from the root of the HDFS namespace.

    The REST API takes absolute paths only (`/webhdfs/v1/<PATH>`), and fsspec's
    client appends a path to `/webhdfs/v1` exactly as it receives it. The lister
    strips the bucket URL, which names only the host, to an empty root and joins
    the glob onto it, so every request reached the server as
    `/webhdfs/v1path/to/data.parquet` whichever form carried the path, and
    discovery found nothing. Rooting the stripped path at `/` sends it as
    `/webhdfs/v1/path/to/data.parquet`.
    """

    root_marker = "/"

    @classmethod
    def _strip_protocol(cls, path):
        return "/" + super()._strip_protocol(path).lstrip("/")


class WebHDFSSource(FilesystemSource):
    """
    Access files on WebHDFS.
    """

    @property
    def fs_class(self) -> Type["AbstractFileSystem"]:
        return WebHDFSFileSystem

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
            name="WebHDFS",
            fs_class=self.fs_class,
            uri=uri,
            path=table,
        )

        # Decode individual options (type casting, default values, sanity checks).
        resource_options = ResourceOptions(
            filesystem_incremental=filesystem_incremental,
            column_types=column_types,
            reader_hints=reader_hints,
        )
        fs_kwargs = locator.options.fs_kwargs
        fs_kwargs.update(kwargs)
        apply_alias(fs_kwargs, "kerberos_options", "kerb_kwargs")
        cast_to_int(fs_kwargs, ["port"])
        cast_to_bool(fs_kwargs, ["kerberos", "use_https", "session_verify"])
        cast_to_dict(fs_kwargs, ["kerb_kwargs", "data_proxy"])

        # Sanity checks.
        if "host" not in fs_kwargs or not fs_kwargs["host"]:
            raise MissingConnectorOption("host", "WebHDFS")

        # Create filesystem and dlt resource wrapper.
        fs = self.fs_class(**fs_kwargs)
        return infer_resource(fs=fs, locator=locator, options=resource_options)
