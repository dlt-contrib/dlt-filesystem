import re
from enum import Enum
from typing import Any, Dict, Optional, Tuple, TypeAlias, Union
from urllib.parse import ParseResult, parse_qsl, urlparse

import dlt
from dlt.common.configuration import with_config
from dlt.common.storages import FilesystemConfiguration, fsspec_filesystem
from dlt.common.storages.configuration import FileSystemCredentials
from dlt.extract import DltResource
from fsspec import AbstractFileSystem

from dlt_filesystem.source.error import UnsupportedEndpointError
from dlt_filesystem.source.format.registry import (
    FORMAT_TO_READER,
    reader_for_format,
)
from dlt_filesystem.source.impl.util import has_glob_magic

BucketName: TypeAlias = str
FileGlob: TypeAlias = str


class Carrier(str, Enum):
    """Which of a source's two inputs carries the path of its selection."""

    URI = "uri"
    TABLE = "table"


def select_carrier(uri: Union[ParseResult, str], table: Optional[str]) -> Carrier:
    """Decide whether the URI path or ``table`` carries the selection's path.

    The URI path wins whenever it names anything beyond ``/``, and ``table`` is
    read only when it does not: the split form, ``s3://?...`` + ``bucket/data.csv``
    or ``hdfs://host:8020/`` + ``path/data.csv``. A non-empty ``table`` beside a
    URI path is ignored for the path.

    Every derivation of the path reads this answer instead of deciding again: the
    bucket and glob (:func:`parse_uri`), the single-file classification
    (:func:`source_selects_single_file`) and the reader a file extension selects
    (:func:`select_reader`). ``#`` directives are not paths and are read off both
    carriers (:func:`blob_directives`).
    """
    if isinstance(uri, str):
        uri = urlparse(uri)
    if uri.path.strip().strip("/") or not (table or "").strip():
        return Carrier.URI
    return Carrier.TABLE


def parse_uri(uri: Union[ParseResult, str], table: str) -> Tuple[BucketName, FileGlob]:
    """
    parse the URI of a blob storage and
    return the bucket name and the file glob.

    Supports the following Forms:
    - uri: "gs://"
      table: "bucket-name/file-glob"
    - uri: "gs://uri-bucket-name" (uri-bucket-name is preferred)
      table: "gs://table-bucket-name/file-glob"
    - uri: "gs://"
      table: "gs://bucket-name/file-glob"
    - uri: gs://bucket-name/file-glob
      table: None
    - uri: "gs://bucket-name"
      table: "file-glob"

    The first form is the preferred method. Other forms are supported but discouraged.
    Which of the two carries the path is :func:`select_carrier`'s decision.
    """

    if isinstance(uri, str):
        uri = urlparse(uri)

    table = table.strip()
    host = uri.netloc.strip()

    # Form: scheme://bucket-name/file-glob
    # Note: This form was previously slated for deprecation.
    if select_carrier(uri, table) is Carrier.URI:
        return host, uri.path.lstrip("/")

    table_uri = urlparse(table)

    if host != "":
        return host, table_uri.path.lstrip("/")

    # The netloc rather than `hostname`, which lowercases: a bucket or folder name
    # keeps its case on this form as it does on the URI form above.
    table_host = re.sub(r":\d*$", "", table_uri.netloc.rpartition("@")[2])
    if table_host:
        return table_host, table_uri.path.lstrip("/")

    parts = table_uri.path.lstrip("/").split("/", maxsplit=1)
    if len(parts) != 2:
        return "", parts[0]

    return parts[0], parts[1]


def parse_endpoint(path: str) -> str:
    """
    Parse the endpoint kind from the URI.

    The kind is the reader name the path's file format routes to; a `.gz` suffix is
    stripped first, so `events.csv.gz` resolves the same as `events.csv`. Every format
    in the reader registry is accepted, and an unknown one raises.
    """
    file_extension = path.split(".")[-1]
    if file_extension == "gz":
        file_extension = path.split(".")[-2]
    return reader_for_format(file_extension)


def parse_fragment(spec: str) -> Tuple[str, Optional[str], Dict[str, str]]:
    """Split a filesystem-source spec into ``(path, format_hint, hints)``.

    The trailing ``#fragment`` (everything after the *last* ``#``) is a per-URI
    reader-directive channel. It may carry, in any combination:

    - a bare **format hint** -- a single known-format token (``#csv``); and/or
    - **named hints** -- ``key=value`` pairs (``#sheet_name=foo&header=0``), parsed
      with :func:`urllib.parse.parse_qsl`. That means ``=``-partition semantics
      (``#x=a=b`` -> ``{"x": "a=b"}``, not a split), percent-decoding of keys
      and values (``#sheet_name=My%20Sheet`` -> ``My Sheet``; and, form-encoding
      style, a literal ``+`` decodes to a space), and empty values are kept
      (``#sheet_name=`` -> ``{"sheet_name": ""}``; a reader decides whether that means
      "unset"). Duplicate keys are last-wins.

    Grammar of the fragment (``&``-separated segments):

    - at most one bare (no ``=``) segment, which must be a known format;
    - any number of ``key=value`` segments;
    - empty segments (from a trailing/doubled ``&``) are ignored.

    **All-or-nothing.** If any segment is neither a ``key=value`` pair nor a
    single known-format token -- an unknown bare token (``#sheet_name=foo&bad``), a
    second/duplicate bare format (``#csv&parquet``) -- the whole ``#...`` is
    treated as a *literal part of the path* and returned unchanged as
    ``(spec, None, {})``. A malformed tail never silently drops a valid hint,
    and a typo never gets silently absorbed as one. A fragment that interprets
    to nothing (a bare trailing ``#``) is likewise left literal.

    This preserves literal ``#`` in a path (``/feeds/vendor#1/data.csv``). The
    one narrowed case is a trailing segment that looks exactly like
    ``key=value`` (``vendor#x=y`` as the final segment), which now parses as a
    hint; percent-encode the ``#`` as ``%23`` to force it literal.
    """
    path, sep, fragment = spec.rpartition("#")
    if not sep:
        return spec, None, {}

    format_hint: Optional[str] = None
    hints: Dict[str, str] = {}
    for segment in fragment.split("&"):
        if segment == "":
            # Empty separator noise (trailing or doubled '&'); harmless.
            continue
        if "=" in segment:
            for key, value in parse_qsl(segment, keep_blank_values=True):
                hints[key] = value
        elif segment in FORMAT_TO_READER and format_hint is None:
            format_hint = segment
        else:
            # Unknown bare token, or a second/duplicate bare format: the '#' is
            # a literal part of the path, not a fragment.
            return spec, None, {}

    if format_hint is None and not hints:
        # Nothing to interpret (e.g. a bare trailing '#'); keep it literal.
        return spec, None, {}

    return path, format_hint, hints


_URI_SCHEME = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*:")


def _raw_uri_path(uri: str) -> str:
    """Return the URI path carrier without letting ``urlparse`` consume ``?`` or ``#``.

    The scheme is matched at the start, so a ``://`` inside a query value
    (``dbfs:/data.csv?host=https://...``) is not taken for it, and the authority
    ends at the first ``/``, ``?`` or ``#``.
    """
    scheme = _URI_SCHEME.match(uri)
    if not scheme:
        return uri
    remainder = uri[scheme.end() :]
    if not remainder.startswith("//"):
        return remainder.lstrip("/")
    authority_end = re.search(r"[/?#]", remainder[2:])
    if authority_end is None or authority_end.group() != "/":
        return ""
    return remainder[2 + authority_end.end() :]


def source_selects_single_file(uri: str, table: Optional[str]) -> bool:
    """Return whether the raw source carrier selects one concrete file.

    The carrier is :func:`select_carrier`'s, but the URI path comes from the
    original string so question-mark globs and glob syntax after a literal ``#``
    remain visible. A URI tail made entirely of ``key=value`` segments is treated
    as connection parameters; if a glob also contains ``?``, only the final query
    delimiter is removed. Valid format and reader-hint fragments are removed before
    classification. Empty selections are treated like globs because they mean
    recursive discovery for filesystem implementations that accept an omitted
    bucket or path.
    """
    table = (table or "").strip()
    from_uri = select_carrier(uri, table) is Carrier.URI
    carrier = _raw_uri_path(uri) if from_uri else table
    if from_uri or "://" in carrier:
        carrier_query = urlparse(carrier).query
        query_segments = [segment for segment in carrier_query.split("&") if segment]
        if query_segments and all("=" in segment for segment in query_segments):
            carrier = carrier.rsplit("?", maxsplit=1)[0]
    selection, _, _ = parse_fragment(carrier)
    return bool(selection) and not has_glob_magic(selection)


def split_format_hint(table: str) -> Tuple[str, Optional[str]]:
    """Split a table spec into ``(path, format_hint)``.

    Thin wrapper over :func:`parse_fragment` that drops the named hints, so
    every existing caller keeps its ``(path, format_hint)`` contract. A trailing
    ``#segment`` is an explicit format hint only when ``segment`` is a
    recognized format; otherwise the ``#`` stays part of the path (literal
    ``#`` in file paths like ``/feeds/vendor#1/data.csv`` keep working).
    """
    path, format_hint, _ = parse_fragment(table)
    return path, format_hint


def merge_directives(
    format_hint: Optional[str], hints: Dict[str, str], table: Optional[str]
) -> Tuple[Optional[str], Dict[str, str]]:
    """Fill the directives a URI does not carry from ``table``'s fragment.

    A ``#fragment`` directs the reader rather than naming a file, so unlike the
    path it is not tied to one carrier: a table-borne ``#csv_headless`` beside a
    URI path applies, which is also how a programmatic ``file_format=`` reaches
    the HTTP source. Each directive is taken from the URI when the URI carries
    it, so the ``#format`` token comes from the URI if it names one and hints
    merge per key with the URI's winning.
    """
    _, table_format, table_hints = parse_fragment((table or "").strip())
    return format_hint or table_format, {**table_hints, **hints}


def blob_directives(
    parsed_uri: ParseResult, table: str
) -> Tuple[Optional[str], Dict[str, str]]:
    """Extract the ``#format`` token and ``#key=value`` hints of a source selection.

    Read off both carriers through :func:`merge_directives`, the URI's fragment
    first. :func:`urllib.parse.urlparse` has already split the URI's fragment
    into ``parsed_uri.fragment``; :func:`parse_fragment`'s all-or-nothing rule is
    what keeps a literal ``#`` in an object name (``vendor#1/data.csv``) from
    reading as a directive.
    """
    format_hint: Optional[str] = None
    hints: Dict[str, str] = {}
    if parsed_uri.fragment:
        _, format_hint, hints = parse_fragment(
            f"{parsed_uri.path}#{parsed_uri.fragment}"
        )
    return merge_directives(format_hint, hints, table)


def blob_hints(parsed_uri: ParseResult, table: str) -> Dict[str, str]:
    """Extract ``#key=value`` reader hints for a blob source selection.

    The hints half of :func:`blob_directives`. In the recommended form the
    fragment rides ``--source-table`` (``s3://?...`` + table
    ``bucket/book.xlsx#sheet_name=foo``); on the URI-path form it rides the URI
    (``s3://bucket/book.xlsx#sheet_name=foo``). Where both name the same key, the
    URI's value wins.
    """
    return blob_directives(parsed_uri, table)[1]


def select_reader(uri: Union[ParseResult, str], table: str) -> str:
    """Name the reader for a source selection.

    A ``#format`` directive wins (:func:`blob_directives`). Otherwise the file
    extension decides, read off the carrier :func:`select_carrier` chose, so the
    reader and the files always come from the same place: ``s3://bucket/a.csv``
    with a table of ``b.parquet`` reads ``a.csv`` as CSV.
    """
    if isinstance(uri, str):
        uri = urlparse(uri)
    format_hint, _ = blob_directives(uri, table)
    if format_hint is not None:
        return reader_for_format(format_hint)
    _, file_glob = parse_uri(uri, table)
    if select_carrier(uri, table) is Carrier.URI:
        return determine_endpoint(file_glob, file_glob)
    return determine_endpoint(table, file_glob)


def determine_endpoint(table: str, path: str) -> str:
    """
    Find the designated reader method from either `table` or URL `path` component.
    """

    _, file_format = split_format_hint(table)
    if file_format is not None:
        return reader_for_format(file_format)

    try:
        return parse_endpoint(table)
    except Exception:
        try:
            return parse_endpoint(path)
        except UnsupportedEndpointError:
            raise
        except Exception as e:
            raise ValueError(
                f"Failed to parse endpoint from table '{table}' or path '{path}'"
            ) from e


def fsspec_from_resource(filesystem_instance: DltResource) -> AbstractFileSystem:
    """Extract authorized fsspec client from a filesystem resource"""

    @with_config(
        spec=FilesystemConfiguration,
        sections=("sources", filesystem_instance.section, filesystem_instance.name),
    )
    def _get_fsspec(
        bucket_url: str, credentials: Optional[FileSystemCredentials]
    ) -> AbstractFileSystem:
        kwargs: Dict[str, Any] = {}
        if credentials is not None:
            kwargs["credentials"] = credentials
        return fsspec_filesystem(bucket_url, **kwargs)[0]

    return _get_fsspec(
        filesystem_instance.explicit_args.get("bucket_url", dlt.config.value),
        filesystem_instance.explicit_args.get("credentials", dlt.secrets.value),
    )
