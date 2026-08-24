import dlt

from omniload.core.tablename import Defaults, two_level
from omniload.target.model import GenericSqlDestination


class SqliteDestination(GenericSqlDestination):
    table_capability = two_level("sqlite", min_components=1)

    def dlt_dest(self, uri: str, **kwargs):
        return dlt.destinations.sqlalchemy(credentials=uri)

    def table_defaults(self, uri: str) -> Defaults:
        # https://dlthub.com/docs/dlt-ecosystem/destinations/sqlalchemy#dataset-files
        return Defaults(schema="main")
