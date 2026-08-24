import dlt

from omniload.core.tablename import two_level
from omniload.target.model import GenericSqlDestination


class DuckDBDestination(GenericSqlDestination):
    table_capability = two_level(
        "duckdb",
        "Select the database file with --dest-uri; attached database catalogs are not supported.",
    )

    def dlt_dest(self, uri: str, **kwargs):
        kwargs.pop("dest_table", None)
        kwargs.pop("staging_bucket", None)
        return dlt.destinations.duckdb(uri, **kwargs)
