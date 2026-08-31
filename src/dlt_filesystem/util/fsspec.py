import re
from typing import Any
from urllib.parse import urlsplit

from fsspec.implementations.arrow import ArrowFSWrapper
from fsspec.utils import update_storage_options


class ReadIntoArrowFSMixin:
    """Hand out Arrow read handles that satisfy `readinto`.

    fsspec's ``ArrowFile`` mirrors a fixed list of methods from the pyarrow stream it
    wraps, and ``readinto`` is not on that list, so the handle serves ``read()`` but
    fails anything that fills a caller-supplied buffer.

    Gzip is where that surfaces. fsspec registers isal's ``IGzipFile`` as its ``gzip``
    codec whenever ``isal`` is importable and falls back to the stdlib ``GzipFile`` only
    when it is not, and isal decompresses through ``readinto``. Any dependency pulling
    ``xopen`` brings isal along on x86-64 and AArch64, so a `.gz` file read through an
    Arrow-backed filesystem fails with ``AttributeError`` in an environment that happens
    to carry it, and reads fine everywhere else.

    The pyarrow stream implements ``readinto`` itself, so the handle only has to expose
    it. Mixed into a filesystem class rather than applied to ``ArrowFile``, because
    re-boxing a handle would leave the discarded wrapper to close the stream from its
    finalizer.

    TODO: Drop once fsspec mirrors `readinto` on `ArrowFile`.
    """

    def _open(self, path: str, mode: str = "rb", *args: Any, **kwargs: Any) -> Any:
        """Return an Arrow handle, `readinto` included when the handle is readable."""
        # A mixin has no `_open` on its own MRO; the filesystem it composes with does.
        handle = super()._open(path, mode, *args, **kwargs)  # ty: ignore[unresolved-attribute]
        # Only read handles, so `hasattr(handle, "readinto")` keeps reading as a
        # readability probe on a write handle, as it does on a plain `ArrowFile`.
        if "r" in mode and not hasattr(handle, "readinto"):
            handle.readinto = handle.stream.readinto
        return handle


class ReadIntoArrowFSWrapper(ReadIntoArrowFSMixin, ArrowFSWrapper):
    """The generic Arrow filesystem wrapper, with `readinto` on its read handles."""


def infer_storage_options(
    urlpath: str, inherit_storage_options: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Infer storage options from URL path and merge it with existing storage
    options.

    TODO: This is a copy of `fsspec.utils.infer_storage_options` with two
          adjustments. If this can land upstream, let's remove the function
          again.
          - The regex: r"^[a-zA-Z0-9+]+://".
            https://github.com/fsspec/filesystem_spec/pull/2085
          - Don't suppress parsing http/https urls.

    Parameters
    ----------
    urlpath: str or unicode
        Either local absolute file path or URL (hdfs://namenode:8020/file.csv)
    inherit_storage_options: dict (optional)
        Its contents will get merged with the inferred information from the
        given path

    Returns
    -------
    Storage options dict.

    Examples
    --------
    >>> infer_storage_options('/mnt/datasets/test.csv')  # doctest: +SKIP
    {"protocol": "file", "path", "/mnt/datasets/test.csv"}
    >>> infer_storage_options(
    ...     'hdfs://username:pwd@node:123/mnt/datasets/test.csv?q=1',
    ...     inherit_storage_options={'extra': 'value'},
    ... )  # doctest: +SKIP
    {"protocol": "hdfs", "username": "username", "password": "pwd",
    "host": "node", "port": 123, "path": "/mnt/datasets/test.csv",
    "url_query": "q=1", "extra": "value"}
    """

    # Discover Windows paths including disk name in this special case.
    is_filesystem = re.match(r"^[a-zA-Z]:[\\/]", urlpath)

    # Discover URI according to RFC 3986: Scheme names consist of a
    # sequence of characters beginning with a letter and followed by
    # any combination of letters, digits, plus ("+"), period ("."),
    # or hyphen ("-").
    # https://datatracker.ietf.org/doc/html/rfc3986#section-3.1
    is_uri = re.match(r"^[a-zA-Z0-9+.-]+://", urlpath)

    if is_filesystem and not is_uri:
        return {"protocol": "file", "path": urlpath}

    parsed_path = urlsplit(urlpath)
    protocol = parsed_path.scheme or "file"
    if parsed_path.fragment:
        path = "#".join([parsed_path.path, parsed_path.fragment])
    else:
        path = parsed_path.path
    if protocol == "file":
        # Special case parsing file protocol URL on Windows according to:
        # https://msdn.microsoft.com/en-us/library/jj710207.aspx
        windows_path = re.match(r"^/([a-zA-Z])[:|]([\\/].*)$", path)
        if windows_path:
            drive, path = windows_path.groups()
            path = f"{drive}:{path}"

    # Within omniload, we _want_ to parse, to create fewer anomalies.
    # Specifically, the WebDAV connector needs it because it uses the
    # `http` protocol scheme.
    """
    if protocol in ["http", "https"]:
        # for HTTP, we don't want to parse, as requests will anyway
        return {"protocol": protocol, "path": urlpath}
    """

    options: dict[str, Any] = {"protocol": protocol, "path": path}

    if parsed_path.netloc:
        # Parse `hostname` from netloc manually because `parsed_path.hostname`
        # lowercases the hostname which is not always desirable (e.g. in S3):
        # https://github.com/dask/dask/issues/1417
        options["host"] = parsed_path.netloc.rsplit("@", 1)[-1].rsplit(":", 1)[0]

        if protocol in ("s3", "s3a", "gcs", "gs"):
            options["path"] = options["host"] + options["path"]
        else:
            options["host"] = options["host"]
        if parsed_path.port:
            options["port"] = parsed_path.port
        if parsed_path.username:
            options["username"] = parsed_path.username
        if parsed_path.password:
            options["password"] = parsed_path.password

    if parsed_path.query:
        options["url_query"] = parsed_path.query
    if parsed_path.fragment:
        options["url_fragment"] = parsed_path.fragment

    if inherit_storage_options:
        update_storage_options(options, inherit_storage_options)

    return options
