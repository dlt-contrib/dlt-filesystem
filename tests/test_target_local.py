import datetime
import decimal
import importlib.util
import os
import re
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

import polars as pl
import pytest

import dlt_filesystem.target.local as local_module
from dlt_filesystem.error import MissingConnectorOption
from dlt_filesystem.target.local import LocalFilesystemDestination
from dlt_filesystem.target.model import DEFAULT_DATASET_NAME
from dlt_filesystem.target.registry import (
    ADVERTISED_WRITE_FORMATS_TEXT,
    WRITER_REGISTRATIONS,
)
from dlt_filesystem.target.util import _resolve_output_target

# Asserted against the registry rather than typed out, so registering a writer cannot
# leave these three expectations naming a set the code no longer has.
SUPPORTED = re.escape(f"only supports file formats: {ADVERTISED_WRITE_FORMATS_TEXT}")

# Normalized so the relative-form expectations hold on Windows too (os.getcwd() there
# returns a backslash drive path), matching the source-side test.
CWD = os.getcwd().replace(os.sep, "/")


# Output path + format resolution. Absolute forms are cwd/OS-independent; relative forms
# resolve against the working directory. Format comes from the extension unless a #hint
# overrides it, mirroring the file:// source.
resolve_cases = [
    ("file:///data/out.csv", "/data/out.csv", "csv"),
    # All three Feather extensions resolve, and each keeps its own name in the format
    # slot: the alias is what the writer is looked up by, so collapsing it here would
    # hide a registry that had stopped routing one of them.
    ("file:///data/out.feather", "/data/out.feather", "feather"),
    ("file:///data/out.arrow", "/data/out.arrow", "arrow"),
    ("file:///data/out.ipc", "/data/out.ipc", "ipc"),
    ("file:///data/out.jsonl", "/data/out.jsonl", "jsonl"),
    ("file:///data/out.orc", "/data/out.orc", "orc"),
    ("file:///data/out.parquet", "/data/out.parquet", "parquet"),
    # Windows drive and UNC, same handling as the source
    ("file:///C:/data/out.csv", "C:/data/out.csv", "csv"),
    ("file://C:/data/out.parquet", "C:/data/out.parquet", "parquet"),
    ("file:////server/share/out.jsonl", "//server/share/out.jsonl", "jsonl"),
    # #format hint wins over (or supplies) the format; the path keeps the hinted-off name
    ("file:///data/out.dat#jsonl", "/data/out.dat", "jsonl"),
    ("file:///data/feed#csv", "/data/feed", "csv"),
    ("file:///data/feed#orc", "/data/feed", "orc"),
    ("file:///data/feed#feather", "/data/feed", "feather"),
    ("file:///data/feed#arrow", "/data/feed", "arrow"),
    ("file:///data/feed#ipc", "/data/feed", "ipc"),
    # literal '#' in the path (suffix is not a known format) stays part of the path,
    # so the format falls back to the extension
    ("file:///data/v#1/out.csv", "/data/v#1/out.csv", "csv"),
    # relative path resolves against the working directory
    ("file://out.parquet", f"{CWD}/out.parquet", "parquet"),
    ("file://sub/out.csv", f"{CWD}/sub/out.csv", "csv"),
]


@pytest.mark.parametrize(
    ("uri", "path", "fmt"), resolve_cases, ids=[c[0] for c in resolve_cases]
)
def test_resolve_output_target(uri, path, fmt):
    assert _resolve_output_target(uri) == (path, fmt)


@pytest.mark.parametrize(
    "uri",
    ["file:///data/out.txt", "file:///data/out", "file:///data/archive.zip"],
)
def test_unsupported_format_reports_supported_formats(uri):
    with pytest.raises(ValueError, match=SUPPORTED):
        _resolve_output_target(uri)


@pytest.mark.parametrize("uri", ["file:///data/out.bson", "file:///data/out.dat#bson"])
def test_bson_destination_is_rejected(uri):
    """BSON is read-only. Adding ``bson`` to ``FORMAT_TO_READER`` lights up the read
    path, but the destination registers its own writers, so a ``.bson`` output (or an
    explicit ``#bson`` hint) is still rejected. This pins that contract and documents
    the harmless edge that a destination ``#bson`` suffix now reads as an (unsupported)
    format hint rather than part of the filename.
    """
    with pytest.raises(ValueError, match=SUPPORTED):
        _resolve_output_target(uri)


@pytest.mark.parametrize(
    "uri", ["file://", "file://#csv", "file://#csv_duckdb", "file://   "]
)
def test_empty_path_raises(uri):
    with pytest.raises(MissingConnectorOption):
        _resolve_output_target(uri)


@pytest.mark.parametrize("uri", ["file://out.csv#csv_duckdb", "file://out.csv_duckdb"])
def test_csv_duckdb_destination_is_rejected(uri):
    with pytest.raises(ValueError, match=SUPPORTED + re.escape(" (got 'csv_duckdb')")):
        _resolve_output_target(uri)


@pytest.mark.parametrize(
    "uri",
    ["file://data/*.csv", "file://out-?.jsonl", "file://logs/[abc].parquet"],
)
def test_glob_in_destination_path_is_rejected(uri):
    with pytest.raises(ValueError, match="globs"):
        _resolve_output_target(uri)


# A scheme that names its own format pins it. The path then need not carry one, which is
# what keeps `csv://report` and `csv://out.dat` writing CSV; only a path or hint naming a
# different known format is a conflict. Used by the `csv://` compatibility destination.
pinned_cases = [
    ("file:///data/out.csv", "/data/out.csv", "csv"),
    # no extension, and an extension that names no format at all: both fall back
    ("file:///data/report", "/data/report", "csv"),
    ("file:///data/out.dat", "/data/out.dat", "csv"),
    ("file:///data/out.txt", "/data/out.txt", "csv"),
    # an explicit matching hint is not a conflict
    ("file:///data/out.dat#csv", "/data/out.dat", "csv"),
    # a literal '#' in the path still falls back rather than reading as a format
    ("file:///data/v#1/out.log", "/data/v#1/out.log", "csv"),
]


@pytest.mark.parametrize(
    ("uri", "path", "fmt"), pinned_cases, ids=[c[0] for c in pinned_cases]
)
def test_pinned_format_falls_back_instead_of_requiring_an_extension(uri, path, fmt):
    assert _resolve_output_target(uri, "csv") == (path, fmt)


@pytest.mark.parametrize(
    "uri",
    [
        "file:///data/out.jsonl",
        "file:///data/out.parquet",
        "file:///data/out.dat#jsonl",
        "file:///data/out.dat#parquet",
        # known to the reader registry but not writable: still a named non-CSV format
        "file:///data/out.bson",
        # writable, but through `file://`: the pin still names it a non-CSV format
        "file:///data/out.xlsx",
        "file:///data/out.dat#csv_duckdb",
    ],
)
def test_pinned_format_rejects_a_different_known_format(uri):
    with pytest.raises(ValueError, match="only writes csv files"):
        _resolve_output_target(uri, "csv")


def test_pinned_format_still_rejects_a_glob(uri="file://data/*.dat"):
    """Globbing is read-only whatever the format, so the pin does not license it."""
    with pytest.raises(ValueError, match="globs"):
        _resolve_output_target(uri, "csv")


@pytest.mark.parametrize("uri", ["file://out.csv", "out.csv"])
def test_dlt_run_params_requires_uri(uri):
    destination = LocalFilesystemDestination().dlt_run_params(uri, "")
    assert destination["dataset_name"] == DEFAULT_DATASET_NAME
    assert destination["table_name"] == "out"


@pytest.mark.parametrize("table", ["justtable", "a.b.c"])
def test_dlt_run_params_requires_two_part_table(table):
    with pytest.raises(ValueError, match=re.escape("<schema>.<table>")):
        LocalFilesystemDestination().dlt_run_params("file://out.csv", table)


def test_temp_directory_is_cleared_when_the_write_fails(tmp_path, monkeypatch):
    """The staging directory goes away even when reading or writing raises partway.
    The ``csv://`` compatibility destination inherits this ``post_load()``, replacing a
    standalone cleanup that only ran on the success path."""
    destination = LocalFilesystemDestination()
    destination.dlt_dest(f"file://{tmp_path / 'out.csv'}")
    destination.dataset_name, destination.table_name = "public", "rows"
    temp_path = destination.temp_path
    assert os.path.isdir(temp_path)

    def exploding_writer(_file_format):
        raise RuntimeError("writer exploded")

    monkeypatch.setattr(
        "dlt_filesystem.target.local.registration_for_format", exploding_writer
    )
    with pytest.raises(RuntimeError, match="writer exploded"):
        destination.post_load()

    assert not os.path.exists(temp_path)


def test_parquet_refuses_a_decimal_column_staged_as_two_types(tmp_path):
    """dlt stages Arrow data whose decimal type changes mid-load as one exact file per
    type: batches it will not concatenate under ``arrow_concat_promote_options="default"``,
    or Parquet files imported into a column declared ``decimal``. Read back as one load,
    the wide value no longer fits the column Polars types at the finer scale, and the
    export used to carry a null in its place."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    destination = LocalFilesystemDestination()
    output_path = tmp_path / "out.parquet"
    destination.dlt_dest(f"file://{output_path}")
    destination.dataset_name, destination.table_name = "public", "rows"
    table_dir = Path(destination.temp_path) / "public" / "rows"
    table_dir.mkdir(parents=True)
    whole = decimal.Decimal(10**10)
    fine = decimal.Decimal("0." + "1234567890" * 3 + "12345")
    pq.write_table(
        pa.table({"v": pa.array([whole], pa.decimal128(11, 0))}),
        table_dir / "a.parquet",
    )
    pq.write_table(
        pa.table({"v": pa.array([fine], pa.decimal128(38, 35))}),
        table_dir / "b.parquet",
    )

    with pytest.raises(ValueError, match=f"'v' carries the number {whole},"):
        destination.post_load()

    assert not output_path.exists()


# A write that fails leaves the destination as it was. The library under a writer may
# open its path before the load is fully encoded, so post_load() writes into its temp
# directory and copies the finished file through the destination. These cases drive
# post_load() with the rows it reads back replaced, and iterate the registry rather than
# naming writers, so a writer registered later is covered.


class _Unencodable:
    """No encoder of any format here has a spelling for this, dlt's JSON included."""


#: Two good rows, then one no encoder can spell. The good rows come first so a writer
#: that streams row by row has already written them when it fails, which is what makes
#: a partial file look like a finished one.
_UNENCODABLE = [{"id": 1}, {"id": 2}, {"id": 3, "v": _Unencodable()}]

#: Loads a format refuses only after its library has opened the path, with the error
#: each raises. Each is a value an ordinary load can carry: a Parquet staging file hands
#: back a duration as ``timedelta``, and an empty JSON object becomes a struct with no
#: fields. xlsxwriter writes its path on close(), where a lone surrogate fails in the
#: XML encoding.
_LATE_FAILURES = {
    "csv": (
        [{"id": 1, "d": datetime.timedelta(seconds=1)}],
        pl.exceptions.ComputeError,
    ),
    "parquet": ([{"id": 1, "v": {}}], pl.exceptions.InvalidOperationError),
    "xlsx": ([{"id": 1}, {"id": 2, "s": "\ud800"}], UnicodeEncodeError),
}

_REGISTRATIONS = {
    registration.format_keys[0]: registration for registration in WRITER_REGISTRATIONS
}

_FAILURE_CASES = [
    pytest.param(fmt, _UNENCODABLE, Exception, id=f"{fmt}-unencodable")
    for fmt in _REGISTRATIONS
] + [
    pytest.param(fmt, rows, error, id=f"{fmt}-late")
    for fmt, (rows, error) in _LATE_FAILURES.items()
]

_GOOD_ROWS = [
    {"id": 1, "name": "first", "score": 1.5},
    {"id": 2, "name": "second", "score": None},
]


def _destination(path: Path) -> LocalFilesystemDestination:
    """A destination for ``path`` whose staged load holds one empty data file."""
    if path.suffix == ".vortex" and importlib.util.find_spec("vortex") is None:
        pytest.skip("vortex-data is not installed")
    destination = LocalFilesystemDestination()
    destination.dlt_dest(f"file://{path}")
    destination.dataset_name, destination.table_name = "public", "rows"
    table_dir = Path(destination.temp_path) / "public" / "rows"
    table_dir.mkdir(parents=True)
    (table_dir / "load.jsonl").touch()
    return destination


def _export(monkeypatch, path: Path, rows) -> LocalFilesystemDestination:
    """Run ``post_load`` for ``path`` as if the staged load had read back as ``rows``."""
    destination = _destination(path)
    monkeypatch.setattr(
        local_module, "load_dlt_file", lambda _path: iter([dict(row) for row in rows])
    )
    destination.post_load()
    return destination


def test_every_late_failure_names_a_registered_writer():
    assert set(_LATE_FAILURES) <= set(_REGISTRATIONS)


@pytest.mark.parametrize(("fmt", "rows", "error"), _FAILURE_CASES)
def test_a_failed_export_keeps_the_file_already_there(
    tmp_path, monkeypatch, fmt, rows, error
):
    path = tmp_path / f"out.{fmt}"
    _export(monkeypatch, path, [{"id": 0, "name": "earlier"}])
    before = path.read_bytes()

    with pytest.raises(error):
        _export(monkeypatch, path, rows)

    assert path.read_bytes() == before


@pytest.mark.parametrize(("fmt", "rows", "error"), _FAILURE_CASES)
def test_a_failed_export_leaves_no_file(tmp_path, monkeypatch, fmt, rows, error):
    path = tmp_path / f"out.{fmt}"

    with pytest.raises(error):
        _export(monkeypatch, path, rows)

    assert not path.exists()


def test_a_failed_export_leaves_no_new_parent_directory(tmp_path, monkeypatch):
    """The destination's directory is created only once the file is complete."""
    path = tmp_path / "new" / "sub" / "out.json"

    with pytest.raises(TypeError):
        _export(monkeypatch, path, _UNENCODABLE)

    assert not (tmp_path / "new").exists()


def _xlsx_members(path: Path) -> dict[str, bytes]:
    """Every zip member but ``docProps/core.xml``, which carries the creation time.
    xlsxwriter pins the members' own timestamps, so the rest is deterministic."""
    with zipfile.ZipFile(path) as archive:
        return {
            name: archive.read(name)
            for name in archive.namelist()
            if name != "docProps/core.xml"
        }


@pytest.mark.parametrize("fmt", list(_REGISTRATIONS))
def test_an_export_matches_the_writer_output(tmp_path, monkeypatch, fmt):
    """Staging changes where a file is written, never what it holds."""
    path = tmp_path / "staged" / f"out.{fmt}"
    _export(monkeypatch, path, _GOOD_ROWS)

    direct = tmp_path / "direct" / f"out.{fmt}"
    direct.parent.mkdir()
    registration = _REGISTRATIONS[fmt]
    kwargs = {"table_name": "rows"} if registration.takes_table_name else {}
    registration.writer(str(direct), [dict(row) for row in _GOOD_ROWS], **kwargs)

    if fmt == "xlsx":
        assert _xlsx_members(path) == _xlsx_members(direct)
    else:
        assert path.read_bytes() == direct.read_bytes()


def test_the_json_backend_is_the_one_this_process_asked_for():
    """The export tests above run under both of dlt's JSON backends, because json and
    jsonl stream differently under each. dlt binds the backend when ``dlt.common`` is
    first imported, so the second backend needs a process of its own."""
    from dlt.common import json as dlt_json

    expected = os.environ.get("DLT_USE_JSON")
    if expected is None:
        if importlib.util.find_spec("orjson") is None:
            pytest.skip("orjson is not installed, so the default is simplejson")
        expected = "orjson"
    assert dlt_json._impl_name == expected


@pytest.mark.skipif(
    os.environ.get("DLT_USE_JSON") == "simplejson",
    reason="this is the simplejson process",
)
def test_the_export_tests_pass_under_simplejson():
    child = subprocess.run(  # noqa: S603  # trusted: sys.executable + fixed arguments
        [
            sys.executable,
            "-m",
            "pytest",
            "-n",
            "0",
            "--no-cov",
            "-p",
            "no:cacheprovider",
            "-q",
            __file__,
            "-k",
            "failed_export or matches_the_writer_output or json_backend",
        ],
        env={**os.environ, "DLT_USE_JSON": "simplejson"},
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert child.returncode == 0, child.stdout[-4000:] + child.stderr[-2000:]
    assert " passed" in child.stdout


def test_an_export_through_a_symlink_keeps_the_link_and_the_file(tmp_path, monkeypatch):
    """The bytes are copied through the destination rather than renamed over it, so an
    existing file keeps its inode, its mode, and a symlink naming it."""
    target = tmp_path / "real.csv"
    target.write_text("old\n")
    target.chmod(0o640)
    inode = target.stat().st_ino
    link = tmp_path / "out.csv"
    link.symlink_to(target)

    _export(monkeypatch, link, _GOOD_ROWS)

    assert link.is_symlink()
    assert target.stat().st_ino == inode
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    assert target.read_text().startswith("id,name,score")


def test_an_export_keeps_a_hard_link_linked(tmp_path, monkeypatch):
    path = tmp_path / "out.csv"
    path.write_text("old\n")
    other = tmp_path / "other.csv"
    os.link(path, other)

    _export(monkeypatch, path, _GOOD_ROWS)

    assert other.stat().st_ino == path.stat().st_ino
    assert other.read_bytes() == path.read_bytes()
    assert path.read_text().startswith("id,name,score")


def test_temp_directory_is_cleared_when_a_real_writer_fails(tmp_path, monkeypatch):
    path = tmp_path / "out.json"
    destination = _destination(path)
    monkeypatch.setattr(local_module, "load_dlt_file", lambda _path: iter(_UNENCODABLE))

    with pytest.raises(TypeError):
        destination.post_load()

    assert not os.path.exists(destination.temp_path)


def test_a_failure_during_the_copy_truncates_the_destination(tmp_path, monkeypatch):
    """The boundary of the guarantee: once the copy has started the destination is open
    for writing, so a failure there (a full disk, say) leaves an existing file
    truncated. The temp directory is still cleared."""
    path = tmp_path / "out.csv"
    path.write_text("earlier\n")
    destination = _destination(path)
    monkeypatch.setattr(
        local_module, "load_dlt_file", lambda _path: iter(list(_GOOD_ROWS))
    )

    def failing_copy(_source, _target):
        raise OSError("No space left on device")

    monkeypatch.setattr(local_module.shutil, "copyfileobj", failing_copy)
    with pytest.raises(OSError, match="No space left"):
        destination.post_load()

    assert path.read_bytes() == b""
    assert not os.path.exists(destination.temp_path)
