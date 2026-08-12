"""Test the JSON filesystem reader across the three shapes a `.json` file carries in the wild.

Mock-only unit lane (no Docker, no credentials): files are written to ``tmp_path`` and read back
through ``LocalFilesystemSource``, so the assertions cover routing as well as decoding.

A `.json` extension does not say whether the body is one object, an array of objects, or
line-delimited records saved under the wrong extension. ``read_json`` parses the whole document
first and only falls back to line-delimited parsing, which is what makes a pretty-printed array
read as an array. The retired standalone HTTP reader decided on line count instead and misread
every pretty-printed file; the tests below pin that this reader does not.

``jsonl`` keeps routing to the strict line reader, so the two formats stay independent.
"""

import gzip
import json

import pytest

from dlt_filesystem.source.format.readers import read_json
from dlt_filesystem.source.format.registry import (
    advertised_file_formats,
    reader_for_format,
)
from dlt_filesystem.source.fsspec.local import LocalFilesystemSource
from dlt_filesystem.testing.stub import FileItemStub


def _read_via_source(path):
    """Read a local file end-to-end through the shared filesystem reader."""
    return list(LocalFilesystemSource().dlt_source(f"file://{path}", ""))


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


# --- document -> row shapes --------------------------------------------------


def test_single_object_is_one_row(tmp_path):
    """A document that is one JSON object loads as exactly one row."""
    path = _write(tmp_path / "one.json", '{"id": 1, "name": "alice"}')
    assert _read_via_source(path) == [{"id": 1, "name": "alice"}]


def test_array_expands_to_one_row_per_element(tmp_path):
    """A top-level array expands to one row per element, not one nested row."""
    path = _write(tmp_path / "list.json", '[{"id": 1}, {"id": 2}, {"id": 3}]')
    assert [row["id"] for row in _read_via_source(path)] == [1, 2, 3]


def test_pretty_printed_array_is_not_mistaken_for_line_delimited(tmp_path):
    """An indented array spans many lines and still loads as an array.

    This is the case the retired HTTP reader got wrong: it treated any body with more than one
    line as line-delimited, so a pretty-printed array raised or loaded fragments.
    """
    path = _write(
        tmp_path / "pretty.json", json.dumps([{"id": 1}, {"id": 2}], indent=2)
    )
    assert [row["id"] for row in _read_via_source(path)] == [1, 2]


def test_pretty_printed_object_is_one_row(tmp_path):
    """An indented single object stays one row rather than becoming one row per line."""
    path = _write(
        tmp_path / "pretty-object.json",
        json.dumps({"id": 1, "nested": {"a": 1}}, indent=2),
    )
    rows = _read_via_source(path)
    assert len(rows) == 1
    assert rows[0]["nested"] == {"a": 1}


def test_line_delimited_body_under_a_json_extension_still_loads(tmp_path):
    """Records saved one-per-line as `.json` load as records.

    The whole-document parse fails on the second line, so the fallback takes over. Parsing only
    the first line would have truncated the file to one row.
    """
    path = _write(tmp_path / "lines.json", '{"id": 1}\n{"id": 2}\n{"id": 3}\n')
    assert [row["id"] for row in _read_via_source(path)] == [1, 2, 3]


def test_empty_array_loads_zero_rows(tmp_path):
    """An empty array is a valid document and a valid zero-row load, not a failure."""
    assert _read_via_source(_write(tmp_path / "empty.json", "[]")) == []


def test_empty_file_loads_zero_rows(tmp_path):
    """A file that exists but holds nothing is a zero-row load, not an error.

    This is the family's contract for an existing-but-empty source: discovery decides whether a
    source is missing, and a file that resolved is a valid load however little it holds. A
    whitespace-only body is the same case.
    """
    assert _read_via_source(_write(tmp_path / "nothing.json", "")) == []
    assert _read_via_source(_write(tmp_path / "blank.json", "  \n\n")) == []


def test_scalar_documents_pass_through_like_the_sibling_readers(tmp_path):
    """A scalar document, or an array of scalars, yields scalar rows rather than raising.

    `.json` behaves here exactly as `.yaml` and `.jsonl` already do on the same input, so the
    format does not invent a stricter contract than the family it joins. What a destination
    makes of a scalar row is the destination's business.
    """
    assert _read_via_source(_write(tmp_path / "scalar.json", "42")) == [42]
    assert _read_via_source(_write(tmp_path / "scalars.json", '[1, 2, "x"]')) == [
        1,
        2,
        "x",
    ]


def test_byte_order_mark_is_not_treated_as_content(tmp_path):
    """A file exported with a UTF-8 BOM loads, on both the document and the fallback path.

    The BOM is not valid JSON, so without stripping it the whole-document parse fails and the
    line fallback then fails on line 1, taking the whole file down.
    """
    document = tmp_path / "bom-object.json"
    document.write_bytes(b"\xef\xbb\xbf" + b'{"id": 1}')
    assert _read_via_source(document) == [{"id": 1}]

    records = tmp_path / "bom-records.json"
    records.write_bytes(b"\xef\xbb\xbf" + b'{"id": 1}\n{"id": 2}\n')
    assert [row["id"] for row in _read_via_source(records)] == [1, 2]


def test_crlf_line_endings_load(tmp_path):
    """Records separated by `\\r\\n` load, so a file written on Windows is not a failure.

    Both paths are covered: line-delimited records take the fallback, while a pretty-printed
    array with the same line endings is one document.
    """
    records = tmp_path / "crlf-records.json"
    records.write_bytes(b'{"id": 1}\r\n{"id": 2}\r\n')
    assert [row["id"] for row in _read_via_source(records)] == [1, 2]

    document = tmp_path / "crlf-document.json"
    document.write_bytes(b'[\r\n  {"id": 1},\r\n  {"id": 2}\r\n]\r\n')
    assert [row["id"] for row in _read_via_source(document)] == [1, 2]


def test_gzipped_json_resolves_and_loads(tmp_path):
    """`.json.gz` unwraps to the `json` format and decompresses transparently."""
    path = tmp_path / "data.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write('[{"id": 1}, {"id": 2}]')
    assert [row["id"] for row in _read_via_source(path)] == [1, 2]


# --- failure behaviour -------------------------------------------------------


def test_malformed_body_raises_naming_the_line(tmp_path):
    """A body that is neither one document nor line-delimited records fails loudly.

    Dropping the bad record and loading the rest would be worse than failing: the load would
    report success while silently losing rows.
    """
    path = _write(tmp_path / "broken.json", '{"id": 1}\n{"id": 2\n')
    with pytest.raises(ValueError, match="line 2 does not parse"):
        list(read_json(iter([FileItemStub(path)])))  # ty: ignore[invalid-argument-type]


def test_malformed_body_fails_the_load_rather_than_loading_zero_rows(tmp_path):
    """The reader's failure reaches the caller through the source, message intact.

    dlt wraps an extraction error, so the assertion is on what the user reads, not on the type.
    """
    path = _write(tmp_path / "broken.json", '{"id": 1}\n{"id": 2\n')
    with pytest.raises(Exception, match="line 2 does not parse"):
        _read_via_source(path)


def test_single_line_malformed_document_names_line_one(tmp_path):
    """A one-line body that is not valid JSON reports line 1, not a bare decoder error."""
    path = _write(tmp_path / "truncated.json", '{"id": 1')
    with pytest.raises(ValueError, match="line 1 does not parse"):
        list(read_json(iter([FileItemStub(path)])))  # ty: ignore[invalid-argument-type]


def test_blank_lines_between_records_are_skipped(tmp_path):
    """A trailing newline or a blank separator line does not fail the load."""
    path = _write(tmp_path / "blanks.json", '{"id": 1}\n\n{"id": 2}\n\n')
    assert [row["id"] for row in _read_via_source(path)] == [1, 2]


# --- chunking ----------------------------------------------------------------


def test_array_yields_in_chunks(tmp_path):
    """A large array is yielded in chunks, so a big document does not become one huge item."""
    path = _write(tmp_path / "many.json", json.dumps([{"id": n} for n in range(10)]))
    chunks = list(read_json(iter([FileItemStub(path)]), chunksize=4))  # ty: ignore[invalid-argument-type]
    assert [len(chunk) for chunk in chunks] == [4, 4, 2]


def test_line_delimited_body_yields_in_chunks(tmp_path):
    """The fallback path yields on the same chunk boundaries as the document path.

    This pins the output batching, not input memory: the body is already read whole, which is
    inherent to deciding whether it is one document at all.
    """
    body = "".join(f'{{"id": {n}}}\n' for n in range(10))
    path = _write(tmp_path / "many-lines.json", body)
    chunks = list(read_json(iter([FileItemStub(path)]), chunksize=4))  # ty: ignore[invalid-argument-type]
    assert [len(chunk) for chunk in chunks] == [4, 4, 2]


# --- routing -----------------------------------------------------------------


def test_json_routes_to_read_json_and_jsonl_stays_strict():
    """The two formats are independent: adding `json` left `jsonl` on the strict line reader."""
    assert reader_for_format("json") == "read_json"
    assert reader_for_format("jsonl") == "read_jsonl"


def test_json_is_advertised_as_a_base_format():
    """`json` ships with the base install, so it is named in "supported formats" errors."""
    assert "json" in advertised_file_formats()


def test_format_hint_selects_the_json_reader(tmp_path):
    """`#json` resolves as a format hint, which it did not before this format existed.

    This is an intentional endpoint-resolution change: a URI whose fragment is the literal text
    `json` is now read as a format instruction.
    """
    path = _write(tmp_path / "unsuffixed", '[{"id": 1}]')
    rows = list(LocalFilesystemSource().dlt_source(f"file://{path}#json", ""))
    assert [row["id"] for row in rows] == [1]


def test_jsonl_extension_still_reaches_the_strict_reader(tmp_path):
    """A `.jsonl` file is unaffected by the new registry entry."""
    path = _write(tmp_path / "records.jsonl", '{"id": 1}\n{"id": 2}\n')
    assert [row["id"] for row in _read_via_source(path)] == [1, 2]


def test_readers_source_exposes_the_json_transformer():
    """`readers()` constructs every transformer by hand, so the registry entry alone is not enough.

    Selecting the resource by name is what a `.json` load does, and it fails if the adapter was
    not taught about the reader.
    """
    import fsspec

    from dlt_filesystem.source.adapter import readers

    source = readers(
        "file:///tmp", fsspec.filesystem("file"), file_glob="*.json"
    ).with_resources("read_json")
    assert "read_json" in source.resources
