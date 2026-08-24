from omniload.core.tablename import opaque


class MongoDBDestination:
    table_capability = opaque("mongodb", table_label="collection")

    def dlt_dest(self, uri: str, **kwargs):
        from omniload.source.mongodb.adapter import mongodb_insert

        return mongodb_insert(uri)

    def dlt_run_params(self, uri: str, table: str, **kwargs) -> dict:
        parsed_table = self.table_capability.parse(table)
        return {
            "table_name": parsed_table.table,
        }

    def post_load(self):
        pass
