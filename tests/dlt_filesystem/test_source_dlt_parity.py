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
from dlt.extract.exceptions import ResourceExtractionError
from fsspec.implementations.memory import MemoryFileSystem

from dlt_filesystem.source import adapter
from dlt_filesystem.source.adapter import filesystem, readers
from dlt_filesystem.source.error import NoFilesFoundError

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


def _write_listing(tmp_path: Path) -> None:
    """Three files whose alphabetical order is not their modification order.

    Sorting by `file_name` would otherwise reproduce sorting by `modification_date`,
    which leaves the cursor field itself unasserted.
    """
    for name, modified_at in (
        ("c.csv", 1_700_000_000),
        ("a.csv", 1_700_000_100),
        ("b.csv", 1_700_000_200),
    ):
        listed_file = tmp_path / name
        listed_file.write_text("name\nAlice\n")
        os.utime(listed_file, (modified_at, modified_at))


@pytest.mark.parametrize("cursor_path", ("modification_date", "file_name"))
@pytest.mark.parametrize("row_order", ("asc", "desc"))
@pytest.mark.parametrize("last_value_func", (max, min))
def test_row_order_yields_the_same_listing_as_dlt(
    tmp_path: Path, row_order: TSortOrder, last_value_func, cursor_path: str
):
    """Differential against dlt, so a change to its ordering rule is caught here.

    Both halves of dlt's `reverse` expression need exercising: it is true for
    (`asc`, `min`) and for (`desc`, `max`), so the default `max` alone leaves a
    swapped comparison passing. Two cursor fields, because a sort key hard-coded to
    `modification_date` passes every case that only uses it.
    """
    _write_listing(tmp_path)
    fs_client = fsspec.filesystem("file")

    def file_names(source_module):
        return [
            item["file_name"]
            for item in source_module.filesystem(
                str(tmp_path),
                fs_client,
                file_glob="*.csv",
                incremental=dlt.sources.incremental(
                    cursor_path,
                    row_order=row_order,
                    last_value_func=last_value_func,
                ),
            )
        ]

    ordered = file_names(adapter)
    assert ordered == file_names(dlt_filesystem_source)
    assert sorted(ordered) == ["a.csv", "b.csv", "c.csv"]

    by_modification = ["c.csv", "a.csv", "b.csv"]
    expected = (
        by_modification if cursor_path == "modification_date" else sorted(ordered)
    )
    if (row_order == "asc") is (last_value_func is max):
        assert ordered == expected
    else:
        assert ordered == list(reversed(expected))


@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
def test_the_three_additions_are_keyword_only(entry_point: str):
    """Appended keyword-only, not placed where dlt has them.

    dlt puts `kwargs` and `client_kwargs` where our `require_file_match` and
    `filesystem_incremental` sit, so adopting dlt's order would rebind an existing
    positional call's two booleans and silently disarm the strict lister.
    """
    parameters = inspect.signature(getattr(adapter, entry_point)).parameters

    for name in ("kwargs", "client_kwargs", "incremental"):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


def test_a_legacy_positional_call_still_binds_to_its_own_flags(tmp_path: Path):
    """The seventh and eighth positional arguments stay `require_file_match` etc."""
    bound = inspect.signature(filesystem).bind(
        str(tmp_path), fsspec.filesystem("file"), "*.csv", 100, False, True, True
    )

    assert bound.arguments["require_file_match"] is True
    assert bound.arguments["filesystem_incremental"] is True

    # `require_file_match` is still armed: nothing matches, so the strict lister
    # raises rather than yielding an empty listing.
    with pytest.raises(ResourceExtractionError) as raised:
        list(filesystem(*bound.args, **bound.kwargs))

    assert isinstance(raised.value.__cause__, NoFilesFoundError)


def test_bound_clones_share_one_cursor_exactly_as_dlts_do(tmp_path: Path):
    """Pins our clone semantics to dlt's, which is what declaring `incremental` buys.

    dlt's bound cloning shallow-copies the pipe steps and does not reinject the
    wrapper, so hints applied to one clone reach every clone. That is dlt's own
    behaviour, and diverging from it in the package positioned as dlt's drop-in would
    cost more than it buys; this test fails if either side changes.
    """
    for name in ("a.csv", "b.csv", "c.csv"):
        (tmp_path / name).write_text("name\nAlice\n")

    fs_client = fsspec.filesystem("file")

    def clone_listings(source_module):
        base = source_module.filesystem(str(tmp_path), fs_client, file_glob="*.csv")
        first, second = base.with_name("first"), base.with_name("second")
        first.apply_hints(incremental=dlt.sources.incremental("file_name", "b.csv"))
        second.apply_hints(incremental=dlt.sources.incremental("file_name", "c.csv"))
        return [
            [item["file_name"] for item in resource] for resource in (first, second)
        ]

    assert clone_listings(adapter) == clone_listings(dlt_filesystem_source)


@pytest.mark.parametrize(
    ("row_order", "expected"),
    (("asc", ["b.csv", "a.csv"]), ("desc", ["a.csv", "b.csv"])),
)
def test_readers_forwards_the_incremental_cursor_to_its_lister(
    tmp_path: Path, row_order: TSortOrder, expected: list[str]
):
    """`readers` builds one lister where dlt builds four, a separate forwarding path."""
    for name, modified_at in (("b.csv", 1_700_000_000), ("a.csv", 1_700_000_100)):
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
