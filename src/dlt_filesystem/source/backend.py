"""Storage backends whose package sits behind an extra."""

from typing import Any
from urllib.parse import urlsplit

from dlt_filesystem.source.error import MissingBackendError


def import_ocifs() -> Any:
    """Import ``ocifs``, or raise the install hint when the ``oci`` extra is missing.

    The hint also covers ``oci``, the SDK ``ocifs`` requires, because installing the extra
    restores it. Any other import failure is a broken install rather than a missing extra,
    and installing the extra would not fix it, so it propagates as it is.
    """
    try:
        import ocifs
    except ModuleNotFoundError as e:
        if e.name not in ("ocifs", "oci"):
            raise
        raise MissingBackendError(
            "Accessing Oracle Cloud Object Storage needs the ocifs package. "
            "Install it with: pip install 'dlt-filesystem[oci]'"
        ) from e
    return ocifs


def require_backend(bucket_url: str) -> None:
    """Raise the install hint before dlt resolves a backend that sits behind an extra.

    dlt's ``fsspec_filesystem`` answers any ``ImportError`` with
    ``pip install "dlt[<protocol>]"``, and dlt has no ``oci`` extra to install. Every other
    scheme stays with dlt's own handling.
    """
    if urlsplit(bucket_url).scheme == "oci":
        import_ocifs()
