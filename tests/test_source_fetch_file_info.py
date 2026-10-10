"""`fetch_file_info`: fill a listing's missing size and modification date.

dlt 1.31 added the parameter to its `filesystem` resource. These tests go through
our `filesystem()` resource rather than the lister alone, because the parity tests
only inspect the signature and would keep passing if the keyword were accepted and
never forwarded.
"""

from datetime import datetime, timezone
from typing import Any

import pytest
from dlt.extract.exceptions import ResourceExtractionError
from fsspec.implementations.memory import MemoryFileSystem

from dlt_filesystem.source.adapter import filesystem
from dlt_filesystem.source.lister import glob_files

LISTED = datetime(2026, 8, 20, 22, 39, 23, tzinfo=timezone.utc)
FETCHED = datetime(2026, 9, 1, 8, 0, 0, tzinfo=timezone.utc)
NAMES = ("/bucket/alpha.csv", "/bucket/bravo.csv")


class ListingFileSystem(MemoryFileSystem):
    """A filesystem whose listing and `info()` answers are set per test."""

    cachable = False

    def __init__(self, listed: dict[str, Any], fetched: dict[str, Any]):
        super().__init__()
        self.listed = listed
        self.fetched = fetched
        self.info_calls: list[str] = []

    def glob(self, path, maxdepth=None, **kwargs):
        return {name: {"name": name, "type": "file", **self.listed} for name in NAMES}

    def info(self, path, **kwargs):
        self.info_calls.append(path)
        return {"name": path, "type": "file", **self.fetched}

    def modified(self, path):
        raise AssertionError("modified() must not be called")


def listed_items(fs: ListingFileSystem, **kwargs) -> list[dict]:
    return list(filesystem("memory://bucket", fs, file_glob="*.csv", **kwargs))


@pytest.mark.parametrize(
    "listed",
    [
        pytest.param({}, id="both-missing"),
        pytest.param({"created": LISTED}, id="size-missing"),
        pytest.param({"size": 7}, id="date-missing"),
    ],
)
def test_incomplete_listing_is_completed_with_one_fetch_per_file(listed):
    fs = ListingFileSystem(listed, {"size": 7, "created": FETCHED})

    items = listed_items(fs, fetch_file_info=True)

    assert [item["size_in_bytes"] for item in items] == [7, 7]
    # What `info()` reports wins over the listing, as it does in dlt's replace.
    assert [item["modification_date"] for item in items] == [FETCHED, FETCHED]
    assert fs.info_calls == list(NAMES)


def test_complete_listing_costs_no_fetch():
    fs = ListingFileSystem({"size": 3, "created": LISTED}, {})

    items = listed_items(fs, fetch_file_info=True)

    assert [item["size_in_bytes"] for item in items] == [3, 3]
    assert fs.info_calls == []


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({}, id="default"),
        pytest.param({"fetch_file_info": False}, id="off"),
    ],
)
def test_without_the_flag_nothing_is_fetched(kwargs):
    fs = ListingFileSystem({"created": LISTED}, {"size": 7})

    items = listed_items(fs, **kwargs)

    assert all("size_in_bytes" not in item for item in items)
    assert fs.info_calls == []


def test_flag_is_read_from_dlt_config(monkeypatch):
    monkeypatch.setenv("SOURCES__FILESYSTEM__FETCH_FILE_INFO", "true")
    fs = ListingFileSystem({"created": LISTED}, {"size": 7})

    items = listed_items(fs)

    assert [item["size_in_bytes"] for item in items] == [7, 7]
    assert fs.info_calls == list(NAMES)


def test_listing_keys_survive_a_fetch_that_omits_them():
    """Merged rather than replaced: `info()` here reports no date, the listing did."""
    fs = ListingFileSystem({"created": LISTED}, {"size": 7})

    items = listed_items(fs, fetch_file_info=True)

    assert [item["modification_date"] for item in items] == [LISTED, LISTED]


def test_unrecognised_date_key_still_raises_after_a_fetch():
    """A fetch gets one chance to supply a date; a backend with none still raises.

    The `now()` fallback belongs to a transport whose listing is known to carry
    no date (HTTP), and is covered in `test_source_http.py`.
    """
    fs = ListingFileSystem({"size": 7}, {"size": 7})

    with pytest.raises(
        ResourceExtractionError, match="carries no usable modification date"
    ):
        listed_items(fs, fetch_file_info=True)

    assert fs.info_calls == [NAMES[0]]


def test_strict_mode_still_refuses_an_undatable_file():
    fs = ListingFileSystem({"size": 7}, {"size": 7})

    with pytest.raises(ValueError, match="carries no usable modification date"):
        list(
            glob_files(
                fs,
                "memory://bucket",
                "*.csv",
                filesystem_incremental=True,
                fetch_file_info=True,
            )
        )


@pytest.mark.parametrize("scheme", ["http", "https"])
def test_http_date_never_comes_from_dlts_extractor(monkeypatch, scheme):
    """Before 1.31 dlt's HTTP extractor stamps the current time on an undated entry.

    Consulting it would make an undated HTTP listing look dated, so the fetch would
    be skipped and strict mode would accept an invented time.
    """
    from dlt_filesystem.source import lister

    monkeypatch.setitem(lister.MTIME_DISPATCH, scheme, lambda f: FETCHED)
    entry = {"name": f"{scheme}://host/a.csv", "size": 7, "type": "file"}

    assert lister.resolve_modification_date(scheme, entry) is None
