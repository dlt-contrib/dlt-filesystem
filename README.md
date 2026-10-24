# dlt-filesystem

Read files from local disks, object stores, file servers and cloud drives with
[dlt](https://github.com/dlt-hub/dlt), and write them back out.

The package started inside [omniload](https://github.com/panodata/omniload) and now
ships on its own, so any dlt project can build on it. Issues and pull requests are
welcome.

## What it adds to dlt

dlt's own `dlt.sources.filesystem` reads CSV, JSONL and Parquet, plus CSV through
DuckDB. This package reads the 17 formats listed under [File formats](#file-formats),
and its `readers` and `filesystem` take every argument dlt's versions take. A pipeline
on dlt's filesystem source moves across by changing the import.

`filesystem()` also accepts `require_file_match=True`, which raises when the selection
matches no file instead of loading an empty table. The source classes below turn it on
when the URI names one file, and leave globs free to match nothing.

For incremental loads by modification date, the lister reads the date from what each
backend's listing returns. dlt's lookup table has no entry for `r2`, `oss`, `hdfs`,
`webhdfs`, `smb`, `ftp`, `dbfs` or `oci`, and fails on any Arrow-backed client,
whose listing uses a different key. Over HTTP the date comes from the
`Last-Modified` header.

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
| `iterable`    | CBOR, MessagePack, XML and YAML readers                           |
| `spreadsheet` | XLSX (read and write) and ODS (read)                              |
| `vortex`      | Vortex (read and write), Python 3.11 and later                    |
| `full`        | All of the above                                                  |

```shell
pip install 'dlt-filesystem[full]'
```

YAML usually reads without `iterable`, because dlt itself depends on PyYAML.

## Usage

The `readers` source works like dlt's, with one resource per reader:

```python
import dlt
from dlt_filesystem.source.adapter import readers

pipeline = dlt.pipeline(destination="duckdb", dataset_name="inbox")
pipeline.run(
    readers(bucket_url="s3://bucket/prefix", file_glob="*.parquet").read_parquet()
)
```

The source classes take a URI and pick the reader from the file name:

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

The format comes from the file extension, or from a `#format` hint at the end of the
path (`file://feed.dat#csv`). A `.gz` suffix is skipped when choosing the format, so
`events.csv.gz` reads as CSV.

| Format               | Keys                      | Read | Write | Extra         |
|----------------------|---------------------------|------|-------|---------------|
| CSV (Polars)         | `csv`                     | yes  | yes   |               |
| CSV without a header | `csv_headless`            | yes  | no    |               |
| CSV (DuckDB)         | `csv_duckdb`              | yes  | no    | `duckdb`      |
| JSON                 | `json`                    | yes  | yes   |               |
| JSON Lines           | `jsonl`                   | yes  | yes   |               |
| Parquet              | `parquet`                 | yes  | yes   |               |
| ORC                  | `orc`                     | yes  | yes   |               |
| Avro                 | `avro`                    | yes  | no    |               |
| Feather (Arrow IPC)  | `feather`, `arrow`, `ipc` | yes  | yes   |               |
| Vortex               | `vortex`                  | yes  | yes   | `vortex`      |
| Excel                | `xlsx`                    | yes  | yes   | `spreadsheet` |
| OpenDocument         | `ods`                     | yes  | no    | `spreadsheet` |
| BSON                 | `bson`                    | yes  | no    | `bson`        |
| CBOR                 | `cbor`                    | yes  | no    | `iterable`    |
| MessagePack          | `msgpack`                 | yes  | no    | `iterable`    |
| XML                  | `xml`                     | yes  | no    | `iterable`    |
| YAML                 | `yaml`, `yml`             | yes  | yes   | `iterable`    |

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
load to dlt's filesystem destination and write one table per run.

The destination classes implement the interface omniload calls (`dlt_dest`,
`dlt_run_params` and `post_load`). They are not dlt destination factories.

## Documentation

Connection options and per-format reader hints are documented with omniload:
<https://omniload.readthedocs.io/supported-sources/filesystem.html>

## License

MIT. See [LICENSE](https://github.com/dlt-contrib/dlt-filesystem/blob/main/LICENSE).
