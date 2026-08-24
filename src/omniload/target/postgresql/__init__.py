import dlt

from omniload.core.tablename import three_level, uri_with_database
from omniload.target.model import GenericSqlDestination


class PostgresDestination(GenericSqlDestination):
    table_capability = three_level("postgres", catalog_label="database")

    def dlt_dest(self, uri: str, **kwargs):
        dest_table = kwargs.get("dest_table")
        if dest_table:
            catalog = self.parse_table(uri, dest_table).catalog
            if catalog:
                uri = uri_with_database(uri, catalog)
        return dlt.destinations.postgres(credentials=uri, **kwargs)
