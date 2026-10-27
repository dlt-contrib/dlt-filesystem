# dlt-filesystem

Read files from local disks, object stores, file servers and cloud drives with
[dlt](https://github.com/dlt-hub/dlt), and write them back out.

The package started inside [omniload](https://github.com/panodata/omniload) and now
ships on its own, so any dlt project can build on it. Issues and pull requests are
welcome.

## What it adds to dlt

dlt's own `dlt.sources.filesystem` reads CSV, JSONL and Parquet, plus CSV through
DuckDB. This package has the 17 readers listed under [File formats](#file-formats).
Its `readers` and `filesystem` accept every keyword argument dlt's versions accept,
so a pipeline on dlt's filesystem source moves across by changing the import, as
long as it passes `kwargs`, `client_kwargs` and `incremental` by keyword.

Of the four readers both packages have, only `read_jsonl` takes the same options:

- `read_csv` takes Polars' `read_csv` options, where dlt's takes Pandas' and also
  opens the file with any `encoding` it is given. It yields each file as one batch, so
  `chunksize` does not split it.
- `read_parquet` has no `use_pyarrow`.
- `read_csv_duckdb` hands `filename=True` to DuckDB, where dlt turns it into a
  `filename` field on each record.
- `read_parquet` and `read_csv_duckdb` default to different chunk sizes.

`filesystem()` also accepts `require_file_match=True`, which raises when the selection
matches no file instead of loading an empty table. The source classes below turn it on
when the URI names one file, and leave globs free to match nothing.

For incremental loads by modification date, the lister reads the date from what each
backend's listing returns. dlt's lookup table has no entry for `r2`, `oss`, `hdfs`,
`webhdfs`, `smb`, `ftp`, `dbfs` or `oci`, and it expects a backend-specific key that
the Arrow-backed S3 and Azure clients do not carry. Over HTTP an incremental load
needs a valid `Last-Modified` header. WebDAV is addressed over HTTP too, and asks the
WebDAV server for each file's date.

## Install

```shell
pip install dlt-filesystem
```

The base install reads and writes most formats. A few need an extra, and a format
whose extra is missing raises an error that names the extra to install.

| Extra         | Adds                                                              |
|---------------|-------------------------------------------------------------------|
| `bson`        | BSON (pymongo)                                                    |
| `duckdb`      | The DuckDB CSV reader, `csv_duckdb`                               |
| `iterable`    | CBOR and MessagePack (read and write), XML (read)                 |
| `spreadsheet` | XLSX (read and write) and ODS (read)                              |
| `vortex`      | Vortex (read and write), Python 3.11 and later                    |
| `full`        | All of the above (Vortex only on Python 3.11 and later)           |

```shell
pip install 'dlt-filesystem[full]'
```

YAML reads and writes on the base install, because dlt itself depends on PyYAML.

## Usage

Both examples load into DuckDB, so they also need `pip install duckdb`.

The `readers` source works like dlt's, with one resource per reader. Point
`bucket_url` at your own bucket, with credentials configured the way dlt expects:

```python
import dlt
from dlt_filesystem.source.adapter import readers

pipeline = dlt.pipeline(destination="duckdb", dataset_name="inbox")
pipeline.run(
    readers(bucket_url="s3://bucket/prefix", file_glob="*.parquet").read_parquet()
)
```

The source classes take a URI and pick the reader from the file name. This one
reads `data/events.csv` under the working directory:

```python
import dlt
from dlt_filesystem.source.fsspec.local import LocalFilesystemSource

pipeline = dlt.pipeline(destination="duckdb", dataset_name="inbox")
pipeline.run(
    LocalFilesystemSource().dlt_source("file://data/events.csv", table=""),
    table_name="events",
)
```

## File formats

The source classes take the format from the file extension, or from a `#format` hint
(`file://feed.dat#csv`). The S3, R2, GCS, Azure and SFTP sources read the hint from
the `table` argument rather than the URI. A `.gz` suffix is skipped when choosing the
format, so `events.csv.gz` reads as CSV.

| Format                 | Keys                      | Read | Write | Extra         |
|------------------------|---------------------------|------|-------|---------------|
| CSV (Polars)           | `csv`                     | yes  | yes   |               |
| CSV without a header   | `csv_headless`            | yes  | no    |               |
| CSV (DuckDB)           | `csv_duckdb`              | yes  | no    | `duckdb`      |
| JSON                   | `json`                    | yes  | yes   |               |
| JSON Lines             | `jsonl`                   | yes  | yes   |               |
| Parquet                | `parquet`                 | yes  | yes   |               |
| ORC                    | `orc`                     | yes  | yes   |               |
| Avro                   | `avro`                    | yes  | no    |               |
| Feather V2 (Arrow IPC) | `feather`, `arrow`, `ipc` | yes  | yes   |               |
| Vortex                 | `vortex`                  | yes  | yes   | `vortex`      |
| Excel                  | `xlsx`                    | yes  | yes   | `spreadsheet` |
| OpenDocument           | `ods`                     | yes  | no    | `spreadsheet` |
| BSON                   | `bson`                    | yes  | no    | `bson`        |
| CBOR                   | `cbor`                    | yes  | yes   | `iterable`    |
| MessagePack            | `msgpack`                 | yes  | yes   | `iterable`    |
| XML                    | `xml`                     | yes  | no    | `iterable`    |
| YAML                   | `yaml`, `yml`             | yes  | yes   |               |

The Write column is what the `file://` destination writes. The bucket destinations
write dlt's own file formats.

## Sources

Each storage system has a source class whose `dlt_source(uri, table)` returns a dlt
resource. The scheme column is the URI scheme omniload routes to the class.

| Storage                                         | Scheme                        | Class                                         | Backend                     |
|-------------------------------------------------|-------------------------------|-----------------------------------------------|-----------------------------|
| Local files                                     | `file`                        | `source.fsspec.local.LocalFilesystemSource`   | `pyarrow.fs`                |
| HTTP and HTTPS                                  | `http`, `https`               | `source.fsspec.http.HttpFilesystemSource`     | fsspec HTTP                 |
| Amazon S3 and S3-compatible stores              | `s3`                          | `source.impl.remote.S3Source`                 | `pyarrow.fs`                |
| Cloudflare R2                                   | `r2`                          | `source.fsspec.r2.R2Source`                   | `pyarrow.fs`                |
| Google Cloud Storage                            | `gs`                          | `source.impl.remote.GCSSource`                | gcsfs                       |
| Azure Blob Storage, ADLS Gen2                   | `az`, `adls`, `abfss`         | `source.impl.remote.AzureSource`              | `pyarrow.fs`                |
| SFTP                                            | `sftp`                        | `source.impl.remote.SFTPSource`               | fsspec SFTP (paramiko)      |
| FTP                                             | `ftp`                         | `source.fsspec.ftp.FTPSource`                 | fsspec FTP                  |
| HDFS                                            | `hdfs`                        | `source.fsspec.hdfs.HDFSSource`               | `pyarrow.fs`                |
| WebHDFS                                         | `webhdfs`                     | `source.fsspec.webhdfs.WebHDFSSource`         | fsspec WebHDFS              |
| SMB (Windows shares)                            | `smb`                         | `source.fsspec.smb.SMBSource`                 | fsspec SMB (smbprotocol)    |
| WebDAV                                          | `http+webdav`, `https+webdav` | `source.fsspec.webdav.WebdavSource`           | webdav4                     |
| Oracle Cloud Object Storage                     | `oci`                         | `source.fsspec.oci.OCISource`                 | ocifs                       |
| Alibaba Cloud OSS                               | `oss`                         | `source.fsspec.oss.OSSSource`                 | ossfs                       |
| Dropbox                                         | `dropbox`                     | `source.fsspec.dropbox.DropboxSource`         | dropboxdrivefs              |
| Google Drive                                    | `gdrive`                      | `source.fsspec.gdrive.GoogleDriveSource`      | gdrive-fsspec               |
| SharePoint                                      | `sharepoint`, `msgd`          | `source.fsspec.sharepoint.SharePointSource`   | msgraphfs                   |
| OneDrive                                        | `onedrive`                    | `source.fsspec.onedrive.OneDriveSource`       | msgraphfs                   |
| Databricks (UC Volumes, Workspace files, DBFS)  | `dbfs`                        | `source.fsspec.databricks.DatabricksSource`   | fsspec-databricks           |

Class paths are relative to `dlt_filesystem`. The Python packages behind every backend
install with the base package. HDFS also needs Hadoop's native `libhdfs` library,
which pyarrow loads at runtime.

## Destinations

| Storage                       | Scheme                | Class                                     | Writes                                      |
|-------------------------------|-----------------------|-------------------------------------------|---------------------------------------------|
| Local file                    | `file`                | `target.local.LocalFilesystemDestination` | One file, in any format marked Write above  |
| Amazon S3 and S3-compatible   | `s3`                  | `target.remote.S3Destination`             | dlt's filesystem layout                     |
| Google Cloud Storage          | `gs`                  | `target.remote.GCSDestination`            | dlt's filesystem layout                     |
| Azure Blob Storage, ADLS Gen2 | `az`, `adls`, `abfss` | `target.remote.AzureDestination`          | dlt's filesystem layout                     |

The `file://` destination takes its format from the extension or a `#format` hint, as
the sources do. It loads through dlt into a temporary directory, then writes the rows
to the named file without dlt's `_dlt_*` columns. The bucket destinations hand the
load to dlt's filesystem destination, write one table per run, and leave out dlt's
dataset directory level. A `layout` query parameter sets dlt's file layout.

The destination classes are adapters rather than dlt destination factories. They
expose `dlt_dest`, `dlt_run_params` and `post_load`, the interface omniload calls,
and `dlt_dest` returns the dlt destination to load into.

## Documentation

Connection options and per-format reader hints are documented with omniload:
<https://omniload.readthedocs.io/supported-sources/filesystem.html>

## License

MIT. See [LICENSE](https://github.com/dlt-contrib/dlt-filesystem/blob/main/LICENSE).
