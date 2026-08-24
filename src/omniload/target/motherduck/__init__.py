import dlt

from omniload.core.tablename import three_level
from omniload.target.model import GenericSqlDestination


class MotherduckDestination(GenericSqlDestination):
    table_capability = three_level("motherduck", catalog_label="database")

    def dlt_dest(self, uri: str, **kwargs):
        from urllib.parse import parse_qs, urlparse

        parsed = urlparse(uri)
        query = parse_qs(parsed.query)
        token = query.get("token", [None])[0]
        from dlt.destinations.impl.motherduck.configuration import MotherDuckCredentials

        creds = {
            "password": token,
        }
        dest_table = kwargs.get("dest_table")
        table_database = (
            self.parse_table(uri, dest_table).catalog if dest_table else None
        )
        database = table_database or parsed.path.lstrip("/") or parsed.netloc
        if database:
            creds["database"] = database

        return dlt.destinations.motherduck(MotherDuckCredentials(creds), **kwargs)
