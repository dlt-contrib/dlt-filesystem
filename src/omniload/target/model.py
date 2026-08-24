from omniload.core.tablename import Capability, Defaults, TableName, two_level


class GenericSqlDestination:
    """Base implementation for SQL destinations that load into schema tables."""

    table_capability: Capability = two_level("SQL destination")

    def table_defaults(self, uri: str) -> Defaults:
        """Return connection-derived defaults for an unqualified table name."""
        return Defaults()

    def parse_table(self, uri: str, table: str) -> TableName:
        """Parse a destination table and require a resolved schema."""
        parsed = self.table_capability.parse(table, self.table_defaults(uri))
        if parsed.schema is None:
            raise self.table_capability.unresolved_schema_error(table)
        return parsed

    def dlt_run_params(self, uri: str, table: str, **kwargs) -> dict:
        """Return dlt run parameters derived from a qualified table name."""
        parsed = self.parse_table(uri, table)
        return {
            "dataset_name": parsed.schema,
            "table_name": parsed.table,
        }

    def post_load(self):
        """Run no destination-specific follow-up work by default."""
        pass
