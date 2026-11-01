"""A backend behind an extra names that extra on every path that can reach it.

Mock-only (no services, no credentials). `OCISource` imports its filesystem class itself,
while `filesystem()`, `readers()` and `fsspec_from_resource()` go through dlt's
`fsspec_filesystem`, whose own hint for `oci` names a dlt extra that does not exist.
"""

import sys
from unittest.mock import patch

import pytest
from dlt.common.exceptions import MissingDependencyException
from dlt.extract.exceptions import ResourceExtractionError
from fsspec.implementations.memory import MemoryFileSystem
from fsspec.registry import _registry

from dlt_filesystem.source import adapter
from dlt_filesystem.source.adapter import filesystem, readers
from dlt_filesystem.source.error import MissingBackendError
from dlt_filesystem.source.fsspec.oci import OCISource
from dlt_filesystem.source.router import fsspec_from_resource

OCI_URL = "oci://bucket@ns/prefix"
HINT = "pip install 'dlt-filesystem[oci]'"

ENTRY_POINTS = {
    "filesystem": lambda url: filesystem(url, file_glob="*.parquet"),
    "readers": lambda url: readers(url, file_glob="*.parquet").read_parquet(),
}


def extraction_error(entry_point: str, url: str) -> BaseException:
    """Iterate a resource and return what caused dlt's `ResourceExtractionError`."""
    with pytest.raises(ResourceExtractionError) as exc:
        list(ENTRY_POINTS[entry_point](url))
    assert exc.value.__cause__ is not None
    return exc.value.__cause__


@pytest.fixture
def without_ocifs():
    with patch.dict(sys.modules, {"ocifs": None}):
        yield


@pytest.fixture
def ocifs_raising(tmp_path, monkeypatch):
    """Install an `ocifs` whose import runs the given statement."""

    def install(statement: str) -> None:
        package = tmp_path / "ocifs"
        package.mkdir()
        (package / "__init__.py").write_text(statement + "\n")
        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.delitem(sys.modules, "ocifs", raising=False)

    return install


@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
@pytest.mark.parametrize("url", [OCI_URL, "OCI://bucket@ns/prefix"])
@pytest.mark.usefixtures("without_ocifs")
def test_public_api_names_the_extra_to_install(entry_point, url):
    """The scheme is matched the way `urlsplit` reads it, so its case does not matter."""
    cause = extraction_error(entry_point, url)

    assert isinstance(cause, MissingBackendError)
    assert HINT in str(cause)
    assert isinstance(cause.__cause__, ImportError)


@pytest.mark.usefixtures("without_ocifs")
def test_fsspec_from_resource_names_the_extra_to_install():
    resource = filesystem(OCI_URL, file_glob="*.parquet")

    with pytest.raises(MissingBackendError, match=r"dlt-filesystem\[oci\]") as exc:
        fsspec_from_resource(resource)

    assert isinstance(exc.value.__cause__, ImportError)


@pytest.mark.usefixtures("without_ocifs")
def test_a_constructed_filesystem_needs_no_backend():
    """A caller who hands over a client has no need of the package behind the URL."""
    listed = list(
        filesystem(OCI_URL, credentials=MemoryFileSystem(), file_glob="*.parquet")
    )

    assert listed == []


def test_an_installed_backend_reaches_dlt():
    """The guard fires on a missing package only."""
    pytest.importorskip("ocifs")
    reached = []

    def spy(bucket_url, credentials=None, **kwargs):
        reached.append(bucket_url)
        return MemoryFileSystem(), "/bucket"

    with patch.object(adapter, "fsspec_filesystem", spy):
        list(filesystem(OCI_URL, file_glob="*.parquet"))

    assert reached == [OCI_URL]


@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
@pytest.mark.usefixtures("without_ocifs")
def test_other_schemes_do_not_need_ocifs(entry_point):
    assert list(ENTRY_POINTS[entry_point]("memory://nothing-here")) == []


@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
def test_other_schemes_keep_dlt_handling_of_a_missing_package(entry_point):
    """Only `oci` sits behind an extra, so dlt keeps answering for everything else."""
    with patch.dict(sys.modules, {"ossfs": None, "ocifs": None}), patch.dict(_registry):
        _registry.pop("oss", None)
        cause = extraction_error(entry_point, "oss://bucket/prefix")

    assert isinstance(cause, MissingDependencyException)
    assert not isinstance(cause, MissingBackendError)
    assert "dlt[oss]" in str(cause)


BROKEN_IMPORTS = [
    pytest.param("raise ImportError('libsomething.so: cannot open')", id="shared-lib"),
    pytest.param("import a_module_ocifs_needs_that_is_gone", id="other-module"),
    pytest.param(
        "raise ModuleNotFoundError(\"No module named 'oci.auth'\", name='oci.auth')",
        id="sdk-submodule",
    ),
]


@pytest.mark.parametrize("statement", BROKEN_IMPORTS)
def test_oci_source_does_not_call_a_broken_import_a_missing_extra(
    ocifs_raising, statement
):
    ocifs_raising(statement)

    with pytest.raises(ImportError) as exc:
        OCISource().dlt_source(f"{OCI_URL}/x.parquet", "")

    assert HINT not in str(exc.value)


@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
@pytest.mark.parametrize("statement", BROKEN_IMPORTS)
def test_public_api_does_not_call_a_broken_import_a_missing_extra(
    ocifs_raising, entry_point, statement
):
    """The real failure surfaces, rather than either install hint."""
    ocifs_raising(statement)

    cause = extraction_error(entry_point, OCI_URL)

    assert isinstance(cause, ImportError)
    assert not isinstance(cause, (MissingBackendError, MissingDependencyException))


@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
def test_a_missing_sdk_names_the_extra_too(ocifs_raising, entry_point):
    """The extra restores the `oci` SDK as well, which is what `ocifs` needs from it."""
    ocifs_raising("raise ModuleNotFoundError(\"No module named 'oci'\", name='oci')")

    cause = extraction_error(entry_point, OCI_URL)

    assert isinstance(cause, MissingBackendError)
    assert HINT in str(cause)
