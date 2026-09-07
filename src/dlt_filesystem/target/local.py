"""Write a single file to the local filesystem via ``file://``.

The output format is whatever ``target.registry`` registers a writer for; the
supported set is stated once there rather than restated here.

This is the write-side twin of ``dlt_filesystem.source`` ``LocalFilesystemSource``
(the ``file://`` source). It mirrors that URI grammar exactly: everything after
``file://`` is a filesystem path (never an RFC-8089 host), relative paths resolve
against the working directory, and the output format is taken from the file
extension or an explicit ``#format`` hint. See #106 for the URI-semantics discussion
and #143 for the destination request.

dlt's filesystem destination writes a directory layout
(``{dataset}/{table}/{load_id}.{ext}``), not a single named file, so this class
loads into a temp directory and, in ``post_load()``, reads the produced data back
through the format-agnostic ``load_dlt_file`` helper and re-emits it as one clean file
at the requested path (dropping dlt's bookkeeping ``_dlt_*`` columns). A load dlt split
across several files is read in full, in a stable filename order.

``omniload.target.csv`` ``CsvDestination`` (the ``csv://`` scheme) is this class with
``pinned_output_format`` set to ``csv``; it contributes only that restriction and its own
table-name parser.
"""

import os
import shutil
import tempfile
from pathlib import Path

from dlt_filesystem.target.model import DEFAULT_DATASET_NAME
from dlt_filesystem.target.registry import writer_for_format
from dlt_filesystem.target.util import _resolve_output_target, _strip_dlt_columns
from dlt_filesystem.util.loader import load_dlt_file

#: Layout for the private staging bucket below. ``post_load()`` reads those files back
#: by the format in their name, so the name has to carry one; dlt resolves an unset
#: ``layout`` from ambient configuration, where a user's own filesystem layout would
#: reach a bucket they never see. A layout without ``{ext}`` drops the extension, and one
#: like ``{table_name}/data.csv`` puts a misleading one on a gzip-JSONL file.
#:
#: It differs from dlt's own default by one separator on purpose: dlt drops a constructor
#: argument equal to the declared default and falls back to configuration, so passing
#: ``DEFAULT_FILE_LAYOUT`` verbatim would pin nothing.
STAGING_LAYOUT = "{table_name}/{load_id}-{file_id}.{ext}"


class LocalFilesystemDestination:
    """Write a single local file addressed by ``file://``, in any registered format.

    Usage mirrors the ``file://`` source: ``--dest-uri file://<path>[#format]``. The
    ``--dest-table`` value must be ``<dataset>.<table>``; it only names dlt's intermediate
    layout, the output file is the URI path.
    """

    output_path: str
    output_format: str
    temp_path: str
    dataset_name: str
    table_name: str

    #: Format this destination writes regardless of the path, for a compatibility scheme
    #: that names its format in the scheme (``csv://``). ``None`` here takes the format
    #: from the path or ``#hint`` and accepts every registered write format.
    pinned_output_format: str | None = None

    def supports_multiple_tables(self) -> bool:
        """A single output file cannot represent several worksheet tables."""
        return False

    def dlt_dest(self, uri: str, **kwargs):

        import dlt.destinations

        # Resolved (and so validated) before the temp directory exists, so a rejected
        # destination leaves nothing behind to clean up.
        self.output_path, self.output_format = _resolve_output_target(
            uri, self.pinned_output_format
        )
        self.temp_path = tempfile.mkdtemp()
        # dlt writes its layout under this temp bucket; post_load() reassembles the single
        # output file from it. Its own loader_file_format is irrelevant, load_dlt_file
        # reads whatever dlt produced (gzip-jsonl by default, or csv/parquet). Path.as_uri()
        # gives an RFC-correct file:// URL on every platform (file:///tmp/x on POSIX,
        # file:///C:/... on Windows), avoiding the drive-as-host trap of a naive
        # "file://" + path.
        # extra_placeholders is pinned for the same reason as the layout, and it is not
        # covered by pinning the layout: dlt resolves it from ambient configuration too
        # and applies its entries over the built-in ones, so an `ext` or `table_name`
        # entry rewrites a name this layout had already fixed.
        return dlt.destinations.filesystem(
            bucket_url=Path(self.temp_path).as_uri(),
            layout=STAGING_LAYOUT,
            extra_placeholders={},
        )

    def reject_reserved_table(self, table_name: str) -> None:
        """Refuse a destination table in dlt's own namespace.

        dlt writes its bookkeeping into the staging directory of a table named this way
        (a load record into ``_dlt_loads``, and so on), and ``post_load()`` reads every
        data file it finds there, so the export would interleave those rows with the
        source's. Subclasses that parse the table name themselves call this too, since
        they share the ``post_load()`` that does the reading.
        """
        if table_name.startswith("_dlt_"):
            raise ValueError(
                f"Table name {table_name} is reserved by dlt and cannot be written "
                f"to a single file"
            )

    def dlt_run_params(self, uri: str, table: str, **kwargs) -> dict:
        """Decode dataset and table name from `--dest-table` or `--dest-uri` parameters."""

        # When a destination table name is given, decode from a tuple.
        if table:
            table_fields = table.split(".")
            if len(table_fields) != 2:
                raise ValueError("Table name must be in the format <schema>.<table>")
            self.dataset_name, self.table_name = table_fields
            self.reject_reserved_table(self.table_name)

        # If it's empty, use a fixed dataset name (`public`), and derive
        # the table name from the filename path component in the URI.
        else:
            self.dataset_name = DEFAULT_DATASET_NAME
            self.table_name = Path(uri).stem

        return {
            "dataset_name": self.dataset_name,
            "table_name": self.table_name,
        }

    def post_load(self) -> None:
        table_dir = os.path.join(self.temp_path, self.dataset_name, self.table_name)
        try:
            # The whole load is materialized here before writing. dlt omits null keys per
            # row in its intermediate files, so csv output needs the column union across
            # all rows (and parquet needs the full table); streaming those correctly would
            # reintroduce the per-row re-header logic this codebase is moving away from.
            # Buffered is the simple, correct v1; streaming is a follow-up if it matters.
            rows: list[dict] = []
            if os.path.isdir(table_dir):
                # A load may be split across several data files; read them all, in a
                # stable order, so nothing is dropped. Only the target table's data files
                # live here, dlt keeps its bookkeeping tables in sibling directories.
                for name in sorted(os.listdir(table_dir)):
                    data_file = os.path.join(table_dir, name)
                    if os.path.isfile(data_file):
                        rows.extend(
                            _strip_dlt_columns(row) for row in load_dlt_file(data_file)
                        )

            out_dir = os.path.dirname(self.output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

            writer_for_format(self.output_format)(self.output_path, rows)
        finally:
            # Always clear the temp bucket, even if reading or writing failed partway.
            shutil.rmtree(self.temp_path, ignore_errors=True)
