"""File discovery for the filesystem sources.

``glob_files`` is vendored from dlt (Apache-2.0,
``dlt.common.storages.fsspec_filesystem``) so a file's modification date is
resolved from the shape the filesystem client actually returns.

dlt selects that value through a table keyed by URL scheme, where each entry
reads one backend-specific key out of the listing (``LastModified`` for ``s3``,
``updated`` for ``gs``, ``last_modified`` for ``az``). Both halves of that
coupling break here:

- A scheme dlt does not know raises ``KeyError`` on the scheme itself, so
  listing crashes on the first matched file for ``r2://``, ``oss://``,
  ``hdfs://``, ``smb://``, ``ftp://``, ``dbfs://``, ``oci://`` and
  ``webhdfs://``.
- A client whose listing uses a different key than its scheme is mapped to
  raises ``KeyError`` on the key, which is what any ``pyarrow.fs``-backed
  client does (its entries carry ``mtime`` whatever the scheme).

Resolving from the listing keeps every scheme dlt does know byte-identical,
because its own table is consulted first.
"""

import glob
import pathlib
import posixpath
import re
from datetime import datetime, timezone
from typing import Any, Callable, Iterator, Mapping, Tuple
from urllib.parse import urlparse

from dlt.common.storages import FilesystemConfiguration
from dlt.common.storages.configuration import make_fsspec_url
from dlt.common.storages.fsspec_filesystem import (
    MTIME_DISPATCH,
    FileItem,
    guess_mime_type,
)
from dlt.common.time import ensure_pendulum_datetime_utc
from fsspec import AbstractFileSystem
from fsspec.utils import glob_translate


def _arrow_glob(fs_client: AbstractFileSystem, path: str) -> dict | None:
    """List an Arrow-backed glob with one native filesystem request."""
    from fsspec.implementations.arrow import ArrowFSWrapper

    if not isinstance(fs_client, ArrowFSWrapper):
        return None

    from pyarrow.fs import FileSelector

    if glob.has_magic(path):
        magic_at = min(path.find(char) for char in "*?[" if char in path)
        fixed_prefix = path[:magic_at]
        base_dir = fixed_prefix.rsplit("/", 1)[0] if "/" in fixed_prefix else ""
        selector = FileSelector(
            base_dir,
            allow_not_found=True,
            recursive=True,
        )
        file_infos = fs_client.fs.get_file_info(selector)
        pattern = re.compile(glob_translate(path))
    else:
        file_infos = fs_client.fs.get_file_info([path])
        pattern = None

    entries = {}
    for file_info in file_infos:
        try:
            entry = fs_client._make_entry(file_info)
        except FileNotFoundError:
            continue
        if pattern is None or pattern.match(entry["name"]):
            entries[entry["name"]] = entry
    return dict(sorted(entries.items()))


def _from_epoch_millis(value: Any) -> Any:
    """Read a timestamp in milliseconds, as WebHDFS reports one."""
    return ensure_pendulum_datetime_utc(float(value) / 1000)


def _from_mlsd_timestamp(value: Any) -> Any:
    """Read an RFC 3659 ``modify`` fact, ``YYYYMMDDHHMMSS[.sss]``, as FTP reports one.

    fsspec falls back to parsing ``dir`` output on servers without MLSD, which
    yields a year-less ``Aug 4 09:30``. That cannot name an instant, so it is
    rejected here and surfaces as the "no usable modification date" error.
    """
    stamp = datetime.strptime(str(value)[:14], "%Y%m%d%H%M%S")
    return ensure_pendulum_datetime_utc(stamp.replace(tzinfo=timezone.utc))


# The key each backend's listing carries a file's last-modified time under,
# paired with how that backend encodes it. Every entry is taken from the
# backend's own `modified()` implementation, which reads the same key.
MODIFICATION_DATE_KEYS: Tuple[Tuple[str, Callable[[Any], Any]], ...] = (
    ("LastModified", ensure_pendulum_datetime_utc),  # s3fs, ossfs
    ("last_modified", ensure_pendulum_datetime_utc),  # adlfs
    ("updated", ensure_pendulum_datetime_utc),  # gcsfs
    ("modifiedTime", ensure_pendulum_datetime_utc),  # Google Drive
    ("modificationTime", _from_epoch_millis),  # WebHDFS
    ("timeModified", ensure_pendulum_datetime_utc),  # ocifs
    ("modified", ensure_pendulum_datetime_utc),  # fsspec-databricks
    ("modify", _from_mlsd_timestamp),  # FTP
    # `mtime` last: SMB carries both it and `time`, where `time` is the access
    # time, so a backend that offers a more specific key is preferred first.
    ("mtime", ensure_pendulum_datetime_utc),  # local, SMB, pyarrow.fs clients
)


def resolve_modification_date(scheme: str, file_info: Mapping[str, Any]) -> Any:
    """Return a file's modification date from one entry of a filesystem listing.

    Prefers dlt's own per-scheme extractor so known schemes keep their exact
    behaviour, then reads the key the backend emits and decodes it the way that
    backend encodes it.

    Raises:
        ValueError: when the listing carries no usable modification date, naming
            the scheme, the keys present, and why any candidate was rejected.
    """
    extractor = MTIME_DISPATCH.get(scheme)
    if extractor is not None:
        try:
            return extractor(file_info)
        except KeyError:
            # The scheme is known but this client reports a different key, e.g.
            # any pyarrow.fs filesystem addressed as `s3://`.
            pass

    rejected = []
    for key, decode in MODIFICATION_DATE_KEYS:
        if key not in file_info:
            continue
        try:
            return decode(file_info[key])
        except (ValueError, TypeError, OverflowError) as ex:
            rejected.append(f"{key}={file_info[key]!r} ({ex})")

    detail = f" Rejected: {'; '.join(rejected)}." if rejected else ""
    raise ValueError(
        f"Filesystem listing for scheme '{scheme}' carries no usable modification "
        f"date. Keys present: {sorted(file_info)}.{detail}"
    )


def file_url_for(scheme: str, fs_path: str, bucket_url: str) -> str:
    """Compose the URL a reader opens for one listed file.

    dlt composes this through a per-scheme table (`MAKE_URI_DISPATCH`) and falls
    back to `f"{scheme}://{fs_path}"` for a scheme it does not know. That fallback
    assumes the client reports a path relative to the bucket, which is true of
    every object store but not of HTTP, whose paths *are* absolute URLs: the
    fallback then doubles the scheme (`http://http://host/f.csv`). Listing still
    succeeds, so the mistake only surfaces when a reader opens the file.

    A path that already carries the scheme is therefore taken as the URL it is.
    Every other case is left to dlt, so no scheme it does know changes shape.
    """
    if fs_path.startswith(f"{scheme}://"):
        return fs_path
    return make_fsspec_url(scheme, fs_path, bucket_url)


def glob_files(
    fs_client: AbstractFileSystem, bucket_url: str, file_glob: str = "**"
) -> Iterator[FileItem]:
    """Get the files from the filesystem client.

    Args:
        fs_client (AbstractFileSystem): The filesystem client.
        bucket_url (str): The url to the bucket.
        file_glob (str): A glob for the filename filter.

    Returns:
        Iterable[FileItem]: The list of files.
    """
    is_local_fs = "file" in fs_client.protocol
    if is_local_fs and FilesystemConfiguration.is_local_path(bucket_url):
        bucket_url = FilesystemConfiguration.make_file_url(bucket_url)
    bucket_url_parsed = urlparse(bucket_url)

    if is_local_fs:
        root_dir = FilesystemConfiguration.make_local_path(bucket_url)
        # use a Python glob to get files
        files = glob.glob(
            str(pathlib.Path(root_dir).joinpath(file_glob)), recursive=True
        )
        glob_result = {file: fs_client.info(file) for file in files}
    else:
        # convert to fs_path
        root_dir = fs_client._strip_protocol(bucket_url)
        filter_url = posixpath.join(root_dir, file_glob)
        # dlt's copy invalidates the listing cache for the `hf` protocol here.
        # No HuggingFace scheme is registered, so that branch is left out.
        glob_result = _arrow_glob(fs_client, filter_url)
        if glob_result is None:
            glob_result = fs_client.glob(filter_url, detail=True)
        if isinstance(glob_result, list):
            raise NotImplementedError(
                "Cannot request details when using fsspec.glob. For adlfs (Azure)"
                " please use version 2023.9.0 or later"
            )

    for file, md in glob_result.items():
        if md["type"] != "file":
            continue
        scheme = bucket_url_parsed.scheme

        # relative paths are always POSIX
        if is_local_fs:
            # use OS pathlib for local paths
            loc_path = pathlib.Path(file)
            file_name = loc_path.name
            rel_path = loc_path.relative_to(root_dir).as_posix()
            file_url = FilesystemConfiguration.make_file_url(file)
        else:
            file_name = posixpath.basename(file)
            rel_path = posixpath.relpath(file, root_dir)
            file_url = file_url_for(scheme, file, bucket_url)

        mime_type, encoding = guess_mime_type(rel_path)
        # A listing need not carry a size: an HTTP response with neither a
        # `Content-Length` nor a `Content-Range` reports none, which is ordinary
        # chunked transfer rather than an exotic case, and casting it crashed
        # discovery for a concrete file that reads perfectly well.
        #
        # `FileItem` declares the field required, but zero would be
        # indistinguishable from an empty file, so an unknown size is left out
        # instead and a consumer sees a missing key rather than a wrong number. No
        # reader consumes it.
        size = md.get("size")
        file_item = FileItem(  # ty: ignore[missing-typed-dict-key]
            file_name=file_name,
            relative_path=rel_path,
            file_url=file_url,
            mime_type=mime_type,
            modification_date=resolve_modification_date(scheme, md),
        )
        if size is not None:
            file_item["size_in_bytes"] = int(size)
        if encoding is not None:
            file_item["encoding"] = encoding
        yield file_item
