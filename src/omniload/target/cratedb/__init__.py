from omniload.core.tablename import two_level
from omniload.target.model import GenericSqlDestination


class CrateDBDestination(GenericSqlDestination):
    table_capability = two_level(
        "cratedb", "CrateDB has schemas but no database catalog."
    )

    def dlt_dest(self, uri: str, **kwargs):
        uri = uri.replace("cratedb://", "postgres://")
        import dlt_cratedb.impl.cratedb.factory

        return dlt_cratedb.impl.cratedb.factory.cratedb(credentials=uri, **kwargs)
