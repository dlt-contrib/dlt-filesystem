"""`dlt_filesystem`'s source entry points are drop-in replacements for dlt's own.

The package is positioned as the delta over ``dlt.sources.filesystem``, which only
holds if a pipeline written against dlt's ``filesystem`` / ``readers`` keeps working
when it imports ours instead. These tests read dlt's signature at runtime rather than
a hand-typed list, so a dlt release that adds a keyword fails here instead of in a
caller's pipeline.
"""

import glob
import inspect
import os
from contextlib import contextmanager
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


#: The three orders a cursor sort has to be told apart from, all mutually distinct.
#: A sort is only observable where its output differs from the order the lister
#: already returns, and a `file_name` cursor is only distinguishable from a
#: `modification_date` one where those two differ as well. Leaving any pair to
#: coincide leaves a missing sort passing.
LISTING_ORDER = ["b.csv", "c.csv", "a.csv"]
MODIFICATION_ORDER = ["c.csv", "b.csv", "a.csv"]
NAME_ORDER = ["a.csv", "b.csv", "c.csv"]


@contextmanager
def _listing_order_pinned():
    """Serve `glob.glob` in `LISTING_ORDER`, for dlt's local lister and for ours.

    Both read a local directory through `glob.glob`, whose order Python documents as
    the filesystem's own and therefore arbitrary. A fixture that lets it stand cannot
    keep the three orders apart: a filesystem that enumerates in name order silently
    hides a missing `file_name` sort, on that machine only. Patching the module both
    listers call keeps the differential comparing like with like.
    """
    real_glob = glob.glob

    def ordered_glob(*args, **kwargs):
        found = {Path(path).name: path for path in real_glob(*args, **kwargs)}
        return [found[name] for name in LISTING_ORDER if name in found] + [
            path for name, path in found.items() if name not in LISTING_ORDER
        ]

    with patch.object(glob, "glob", ordered_glob):
        yield


def _write_listing(tmp_path: Path) -> None:
    """Write the three files and time them into `MODIFICATION_ORDER`."""
    for position, name in enumerate(MODIFICATION_ORDER):
        listed_file = tmp_path / name
        listed_file.write_text("name\nAlice\n")
        modified_at = 1_700_000_000 + position * 100
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
        with _listing_order_pinned():
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
    assert sorted(ordered) == NAME_ORDER

    expected = MODIFICATION_ORDER if cursor_path == "modification_date" else NAME_ORDER
    if (row_order == "asc") is (last_value_func is max):
        assert ordered == expected
    else:
        assert ordered == list(reversed(expected))


def test_a_cursor_without_row_order_leaves_the_listing_alone(tmp_path: Path):
    """Only `row_order` orders the listing; a bare cursor must not sort it.

    dlt sorts on `incremental.row_order` alone, so a condition widened to any cursor
    would reorder every incremental run and materialise a listing that should stay
    lazy. The fixture times its files so the two orders differ, which is what makes a
    stray sort visible here.
    """
    _write_listing(tmp_path)
    fs_client = fsspec.filesystem("file")

    def file_names(source_module, **cursor):
        with _listing_order_pinned():
            return [
                item["file_name"]
                for item in source_module.filesystem(
                    str(tmp_path), fs_client, file_glob="*.csv", **cursor
                )
            ]

    bare_cursor = {"incremental": dlt.sources.incremental("modification_date")}

    assert file_names(adapter) == LISTING_ORDER
    assert file_names(adapter, **bare_cursor) == LISTING_ORDER
    assert file_names(adapter, **bare_cursor) == file_names(
        dlt_filesystem_source, **bare_cursor
    )


def test_the_default_listing_stays_lazy(tmp_path: Path):
    """Ordering is the one branch allowed to materialise the listing.

    Everything else pages through `glob_files` as it yields, so a page reaches the
    caller before the last file has been listed. A `list()` or `sorted()` on the
    default path is invisible to every other test here.
    """
    _write_listing(tmp_path)
    listed_before_first_page: list[int] = []
    real_glob_files = adapter.glob_files

    def counting_glob_files(*args, **kwargs):
        for count, file_model in enumerate(real_glob_files(*args, **kwargs), start=1):
            listed_before_first_page.append(count)
            yield file_model

    with patch.object(adapter, "glob_files", counting_glob_files):
        pages = filesystem(
            str(tmp_path),
            fsspec.filesystem("file"),
            file_glob="*.csv",
            files_per_page=1,
        )
        next(iter(pages))

    assert listed_before_first_page == [1]


def test_readers_pipes_one_shared_lister_where_dlt_builds_one_each():
    """A deliberate divergence from dlt, and the reason `readers` forwards once."""
    ours = readers("memory://bucket", MemoryFileSystem(), file_glob="*.none")
    theirs = dlt_filesystem_source.readers(
        "memory://bucket", MemoryFileSystem(), file_glob="*.none"
    )

    def listers(source):
        parents = [resource._parent for resource in source.resources.values()]
        # An unpiped transformer still reports a `_parent`: one shared empty
        # placeholder, named `None`, which counts as a single identity just as a real
        # shared lister does. Only the pipe tells the two apart.
        assert parents and all(
            parent is not None
            and parent.name == "filesystem"
            and not parent._pipe.is_empty
            for parent in parents
        )
        return {id(parent) for parent in parents}

    assert len(listers(ours)) == 1
    assert len(listers(theirs)) == len(theirs.resources)


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


@pytest.mark.parametrize(
    ("require_file_match", "filesystem_incremental"), ((True, False), (False, True))
)
def test_a_legacy_positional_call_still_binds_to_its_own_flags(
    tmp_path: Path, require_file_match: bool, filesystem_incremental: bool
):
    """The sixth and seventh positional arguments stay ours, and stay in that order.

    Both orderings are driven because two equal booleans bind the same way whichever
    parameter receives which, so a swap of the two would pass unnoticed.
    """
    bound = inspect.signature(filesystem).bind(
        str(tmp_path),
        fsspec.filesystem("file"),
        "*.csv",
        100,
        False,
        require_file_match,
        filesystem_incremental,
    )

    assert bound.arguments["require_file_match"] is require_file_match
    assert bound.arguments["filesystem_incremental"] is filesystem_incremental

    if require_file_match:
        # Still armed: nothing matches, so the strict lister raises rather than
        # yielding an empty listing.
        with pytest.raises(ResourceExtractionError) as raised:
            list(filesystem(*bound.args, **bound.kwargs))
        assert isinstance(raised.value.__cause__, NoFilesFoundError)
    else:
        assert list(filesystem(*bound.args, **bound.kwargs)) == []


def test_bound_clones_share_one_cursor_exactly_as_dlts_do(tmp_path: Path):
    """Pins our clone semantics to dlt's, which is what declaring `incremental` buys.

    dlt's bound cloning shallow-copies the pipe steps and does not reinject the
    wrapper, so hints applied to one clone reach every clone. That is dlt's own
    behaviour, and diverging from it in the package positioned as dlt's drop-in would
    cost more than it buys. This pins the two together, not the sharing itself: an
    upstream change that isolates both leaves it green, and a caller who needs
    isolation today builds separate resources.
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
    """`readers` forwards through its own call site, so it needs its own case."""
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
