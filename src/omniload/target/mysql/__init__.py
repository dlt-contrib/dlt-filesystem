from urllib.parse import urlparse

import dlt

from omniload.core.tablename import Defaults, two_level
from omniload.target.model import GenericSqlDestination


class MySqlDestination(GenericSqlDestination):
    table_capability = two_level("mysql", min_components=1, schema_label="database")

    def dlt_dest(self, uri: str, **kwargs):
        return dlt.destinations.sqlalchemy(credentials=uri)

    def table_defaults(self, uri: str) -> Defaults:
        parsed = urlparse(uri)
        database = parsed.path.lstrip("/")
        return Defaults(schema=database or None)
