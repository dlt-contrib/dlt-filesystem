"""`dlt_filesystem`'s source entry points are drop-in replacements for dlt's own.

The package is positioned as the delta over ``dlt.sources.filesystem``, which only
holds if a pipeline written against dlt's ``filesystem`` / ``readers`` keeps working
when it imports ours instead. These tests read dlt's signature at runtime rather than
a hand-typed list, so a dlt release that adds a keyword fails here instead of in a
caller's pipeline.
"""

import inspect
import os
from pathlib import Path
from unittest.mock import patch

import dlt
import dlt.sources.filesystem as dlt_filesystem_source
import fsspec
import pytest
from dlt.common.typing import TSortOrder
from fsspec.implementations.memory import MemoryFileSystem

from dlt_filesystem.source import adapter
from dlt_filesystem.source.adapter import filesystem, readers

ENTRY_POINTS = ("filesystem", "readers")


def _dlt_parameters(name: str):
    return inspect.signature(getattr(dlt_filesystem_source, name)).parameters


def _parameter_cases():
    return [
        pytest.param(entry_point, parameter.name, id=f"{entry_point}-{parameter.name}")
        for entry_point in ENTRY_POINTS
        for parameter in _dlt_parameters(entry_point).values()
    ]


@pytest.mark.parametrize(("entry_point", "parameter_name"), _parameter_cases())
def test_entry_point_declares_dlts_parameter_with_dlts_default(
    entry_point: str, parameter_name: str
):
    theirs = _dlt_parameters(entry_point)[parameter_name]
    ours = inspect.signature(getattr(adapter, entry_point)).parameters

    assert parameter_name in ours, (
        f"{entry_point}() drops dlt's {parameter_name!r}, so a pipeline that passes it "
        "raises TypeError against ours where dlt accepts it"
    )
    assert ours[parameter_name].default == theirs.default


@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
def test_every_dlt_keyword_binds(entry_point: str):
    """Binding is the property a caller sees; declaring the name is not enough."""
    keywords = {
        name: None
        for name in _dlt_parameters(entry_point)
        if name not in ("bucket_url", "credentials")
    }

    inspect.signature(getattr(adapter, entry_point)).bind(
        "memory://bucket", MemoryFileSystem(), **keywords
    )


def test_filesystem_does_not_extract_content_by_default(tmp_path: Path):
    """Pins the runtime behaviour the declaration now agrees with.

    `FilesystemConfigurationResource.extract_content` is False and dlt resolves the
    default from the spec, so this passes whatever the signature says; the
    declaration itself is guarded by the parameter-default test above.
    """
    (tmp_path / "people.csv").write_text("name\nAlice\n")
    fs_client = fsspec.filesystem("file")

    default_call = list(filesystem(str(tmp_path), fs_client, file_glob="*.csv"))
    explicit_call = list(
        filesystem(str(tmp_path), fs_client, file_glob="*.csv", extract_content=True)
    )

    assert "file_content" not in default_call[0]
    assert explicit_call[0]["file_content"] == b"name\nAlice\n"


@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
def test_fsspec_client_is_built_with_kwargs_and_client_kwargs(entry_point: str):
    """Parity of the signature is not forwarding, and only this observes forwarding."""
    received: list[dict] = []

    def spy(bucket_url, credentials=None, **kwargs):
        received.append(kwargs)
        return MemoryFileSystem(), "/bucket"

    with patch.object(adapter, "fsspec_filesystem", spy):
        built = getattr(adapter, entry_point)(
            "memory://bucket",
            None,
            file_glob="*.none",
            kwargs={"use_ssl": True},
            client_kwargs={"verify": "public.crt"},
        )
        if entry_point == "readers":
            built = built.resources["read_csv"]._parent
        assert list(built) == []

    assert received == [
        {"kwargs": {"use_ssl": True}, "client_kwargs": {"verify": "public.crt"}}
    ]


def test_a_constructed_filesystem_ignores_kwargs_and_client_kwargs(tmp_path: Path):
    """A caller who hands over a client has already spent both, as in dlt's resource."""
    (tmp_path / "people.csv").write_text("name\nAlice\n")

    def explode(*args, **kwargs):
        raise AssertionError("fsspec_filesystem must not be called")

    with patch.object(adapter, "fsspec_filesystem", explode):
        listed = list(
            filesystem(
                str(tmp_path),
                fsspec.filesystem("file"),
                file_glob="*.csv",
                kwargs={"use_ssl": True},
                client_kwargs={"verify": "public.crt"},
            )
        )

    assert [item["file_name"] for item in listed] == ["people.csv"]


@pytest.mark.parametrize("row_order", ("asc", "desc"))
@pytest.mark.parametrize("last_value_func", (max, min))
def test_row_order_yields_the_same_listing_as_dlt(
    tmp_path: Path, row_order: TSortOrder, last_value_func
):
    """Differential against dlt, so a change to its ordering rule is caught here.

    Both halves of dlt's `reverse` expression need exercising: it is true for
    (`asc`, `min`) and for (`desc`, `max`), so the default `max` alone leaves a
    swapped comparison passing.
    """
    for name, modified_at in (
        ("b.csv", 1_700_000_100),
        ("a.csv", 1_700_000_000),
        ("c.csv", 1_700_000_200),
    ):
        listed_file = tmp_path / name
        listed_file.write_text("name\nAlice\n")
        os.utime(listed_file, (modified_at, modified_at))

    fs_client = fsspec.filesystem("file")

    def file_names(source_module):
        return [
            item["file_name"]
            for item in source_module.filesystem(
                str(tmp_path),
                fs_client,
                file_glob="*.csv",
                incremental=dlt.sources.incremental(
                    "modification_date",
                    row_order=row_order,
                    last_value_func=last_value_func,
                ),
            )
        ]

    ordered = file_names(adapter)
    assert ordered == file_names(dlt_filesystem_source)
    assert sorted(ordered) == ["a.csv", "b.csv", "c.csv"]


@pytest.mark.parametrize(
    ("row_order", "expected"),
    (("asc", ["a.csv", "b.csv"]), ("desc", ["b.csv", "a.csv"])),
)
def test_readers_forwards_the_incremental_cursor_to_its_lister(
    tmp_path: Path, row_order: TSortOrder, expected: list[str]
):
    """`readers` builds one lister where dlt builds four, a separate forwarding path."""
    for name, modified_at in (("b.csv", 1_700_000_100), ("a.csv", 1_700_000_000)):
        listed_file = tmp_path / name
        listed_file.write_text("name\nAlice\n")
        os.utime(listed_file, (modified_at, modified_at))

    source = readers(
        str(tmp_path),
        fsspec.filesystem("file"),
        file_glob="*.csv",
        incremental=dlt.sources.incremental("modification_date", row_order=row_order),
    )
    lister = source.resources["read_csv"]._parent

    assert [item["file_name"] for item in lister] == expected
