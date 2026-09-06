"""Write-side registry: derivation, ambiguity rejection, and drift against the docs.

`test_target_local.py` covers URI/format resolution; this file covers the registry the
resolver reads and the claims the documentation makes about it.
"""

import re
from pathlib import Path

import pytest

from dlt_filesystem.source.format.registry import FORMAT_TO_READER
from dlt_filesystem.target.registry import (
    FORMAT_TO_WRITER,
    WRITE_FORMATS,
    WRITE_FORMATS_TEXT,
    WRITER_REGISTRATIONS,
    WriterRegistration,
    _build_writer_map,
    supported_write_format_message,
    writer_for_format,
)

DOCS = Path(__file__).resolve().parents[2] / "docs" / "supported-sources"


def test_write_formats_is_derived_from_the_registrations():
    """The map and the tuple cannot disagree, because one is built from the other."""
    registered = tuple(
        key for registration in WRITER_REGISTRATIONS for key in registration.format_keys
    )
    assert WRITE_FORMATS == registered
    assert set(FORMAT_TO_WRITER) == set(registered)
    assert WRITE_FORMATS_TEXT == ", ".join(registered)


def test_duplicate_format_registration_is_rejected():
    """Two writers claiming one format is a bug, not a last-one-wins override."""
    with pytest.raises(ValueError, match="Duplicate file format registration: csv"):
        _build_writer_map(
            (
                WriterRegistration(lambda path, rows: None, ("csv",)),
                WriterRegistration(lambda path, rows: None, ("csv",)),
            )
        )


@pytest.mark.parametrize("file_format", WRITE_FORMATS)
def test_every_registered_format_resolves_to_a_callable(file_format):
    assert callable(writer_for_format(file_format))


def test_unregistered_format_raises():
    with pytest.raises(NotImplementedError, match="Unsupported file format: bson"):
        writer_for_format("bson")


def test_every_write_format_is_also_a_read_format():
    """A file omniload writes and cannot read back is a dead end, so the write set stays
    a subset of the read set. The reverse does not hold: several formats are read-only
    on purpose (see the registration comments)."""
    assert set(WRITE_FORMATS) <= set(FORMAT_TO_READER)


def test_the_supported_format_message_names_the_registered_set():
    message = supported_write_format_message("txt")
    assert WRITE_FORMATS_TEXT in message
    assert "(got 'txt')" in message


# --- documentation drift -------------------------------------------------------------
#
# Every place the docs state which formats `file://` writes, with the parser that reads
# the claim back out. A new format then needs no edit here; a new *claim site* does, and
# a stale claim at an existing site fails.


def _matrix_write_formats() -> set[str]:
    """Formats whose Write column is ticked in the filesystem format matrix."""
    formats = set()
    for line in (DOCS / "filesystem.md").read_text().splitlines():
        cells = [cell.strip() for cell in line.split("|")]
        # | Format | Description | Extensions | Format hint | Read | Write |
        if len(cells) != 8 or not cells[4].startswith("#"):
            continue
        if cells[6] == "✅":
            formats.add(cells[4].lstrip("#"))
    return formats


def _prose_formats(text: str) -> set[str]:
    """Format names out of a comma/and-separated prose list, backticks optional."""
    return {
        token.strip("`. ").lower()
        for token in re.split(r",|\band\b", text)
        if token.strip()
    }


PROSE_CLAIMS = [
    (
        "filesystem.md",
        r"Supported formats for write operations are currently ([^.]+)\.",
    ),
    (
        "file.md",
        r"Supported output formats\s+are ([^;]+);",
    ),
]


def test_the_matrix_write_column_matches_the_registry():
    assert _matrix_write_formats() == set(WRITE_FORMATS)


@pytest.mark.parametrize(
    ("page", "pattern"), PROSE_CLAIMS, ids=[c[0] for c in PROSE_CLAIMS]
)
def test_prose_claims_match_the_registry(page, pattern):
    """A page that enumerates the write set in prose must enumerate the current one."""
    text = (DOCS / page).read_text()
    match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
    assert match, (
        f"{page}: the write-format claim this test pins is gone; drop the entry"
    )
    assert _prose_formats(match.group(1)) == set(WRITE_FORMATS)


def test_the_pages_that_stopped_naming_the_write_set_have_not_regrown_it():
    """`yaml.md` and `xml.md` used to spell the write set out in passing, which is how
    both came to name a stale one. They link the matrix instead now. This is a tripwire
    for that one phrasing, not a general parser: a genuinely new claim site needs an
    entry in PROSE_CLAIMS."""
    stale = re.compile(r"writes\s+`csv`", re.IGNORECASE)
    offenders = [
        page.name
        for page in sorted(DOCS.glob("*.md"))
        if stale.search(page.read_text())
    ]
    assert offenders == []
