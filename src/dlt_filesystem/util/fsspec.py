import re
from typing import Any
from urllib.parse import unquote, urlsplit

from fsspec.implementations.arrow import ArrowFSWrapper
from fsspec.utils import update_storage_options


class ReadIntoArrowFSMixin:
    """Hand out Arrow read handles that satisfy `readinto`.

    fsspec's ``ArrowFile`` mirrors a fixed list of methods from the pyarrow stream it
    wraps, and ``readinto`` joined that list in fsspec 2026.9.0, so every version from
    this project's ``>=2024.6`` floor up to 2026.7.0 hands out a handle that serves
    ``read()`` and fails anything that fills a caller-supplied buffer.

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

    From 2026.9.0 the `hasattr` guard below makes this a no-op.

    TODO: Drop once the fsspec floor reaches 2026.9.0, which mirrors `readinto` on
    `ArrowFile` itself.
    """

    def _open(self, path: str, mode: str = "rb", *args: Any, **kwargs: Any) -> Any:
        """Return an Arrow handle, `readinto` included when the handle is readable."""
        # A mixin has no `_open` on its own MRO; the filesystem it composes with does.
        handle = super()._open(path, mode, *args, **kwargs)  # ty: ignore[missing-attribute]
        # The `"r" in mode` gate is what leaves write handles untouched. `hasattr` never
        # made that distinction: measured on a plain `ArrowFile`, it answers False for
        # both modes below fsspec 2026.9.0 and True for both from it. It is here to
        # leave a handle that already carries the method alone, which from 2026.9.0 is
        # all of them.
        if "r" in mode and not hasattr(handle, "readinto"):
            handle.readinto = handle.stream.readinto
        return handle


class ReadIntoArrowFSWrapper(ReadIntoArrowFSMixin, ArrowFSWrapper):
    """The generic Arrow filesystem wrapper, with `readinto` on its read handles."""


def unquote_userinfo(value: str) -> str:
    """Percent-decode a URI's username or password, all or nothing.

    Mirrors `fsspec.utils._unquote_userinfo` (fsspec/filesystem_spec#2114): a value
    whose escapes do not decode to valid UTF-8 is returned raw, rather than with
    U+FFFD in place of the bytes that failed.
    """
    try:
        return unquote(value, errors="strict")
    except UnicodeDecodeError:
        return value


def infer_storage_options(
    urlpath: str, inherit_storage_options: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Infer storage options from URL path and merge it with existing storage
    options.

    TODO: This is a copy of `fsspec.utils.infer_storage_options` with these
          adjustments:
          - It parses http/https URLs, which fsspec returns unparsed. The
            WebDAV connector needs the parse.
          - The URI regex accepts `+`, `.` and `-` in a scheme. Upstream since
            fsspec 2026.7.0 (fsspec/filesystem_spec#2085); kept while the
            floor is `>=2024.6`.
          - Userinfo is percent-decoded (fsspec/filesystem_spec#2114) and an
            IPv6 host keeps its brackets and colons
            (fsspec/filesystem_spec#2170), both ported ahead of a fsspec
            release that carries them.
          Remove once the floor reaches a tag carrying both ported fixes and
          http/https parsing is no longer needed.

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

    # Here we _want_ to parse, to create fewer anomalies.
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
        host = parsed_path.netloc.rsplit("@", 1)[-1]
        if host.startswith("[") and "]" in host:
            # An IPv6 literal carries colons of its own, so only a colon after
            # the closing bracket separates the port.
            options["host"] = host[: host.index("]") + 1]
        else:
            options["host"] = host.rsplit(":", 1)[0]

        if protocol in ("s3", "s3a", "gcs", "gs"):
            options["path"] = options["host"] + options["path"]
        else:
            options["host"] = options["host"]
        if parsed_path.port:
            options["port"] = parsed_path.port
        if parsed_path.username:
            options["username"] = unquote_userinfo(parsed_path.username)
        if parsed_path.password:
            options["password"] = unquote_userinfo(parsed_path.password)

    if parsed_path.query:
        options["url_query"] = parsed_path.query
    if parsed_path.fragment:
        options["url_fragment"] = parsed_path.fragment

    if inherit_storage_options:
        update_storage_options(options, inherit_storage_options)

    return options
