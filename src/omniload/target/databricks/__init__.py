from urllib.parse import parse_qs, urlparse

import dlt

from omniload.core.tablename import Defaults, three_level
from omniload.target.model import GenericSqlDestination
from omniload.util.auth import get_databricks_oauth_token


class DatabricksDestination(GenericSqlDestination):
    table_capability = three_level(
        "databricks",
        "Specify the schema in --dest-uri when using a bare table name.",
        min_components=1,
    )

    def table_defaults(self, uri: str) -> Defaults:
        p = urlparse(uri)
        q = parse_qs(p.query)
        return Defaults(
            catalog=q.get("catalog", [None])[0],
            schema=q.get("schema", [None])[0],
        )

    def dlt_dest(self, uri: str, **kwargs):
        p = urlparse(uri)
        q = parse_qs(p.query)
        server_hostname = p.hostname
        http_path = q.get("http_path", [None])[0]
        dest_table = kwargs.get("dest_table")
        catalog = (
            self.parse_table(uri, dest_table).catalog
            if dest_table
            else q.get("catalog", [None])[0]
        )

        if not server_hostname:
            raise ValueError("Databricks URI must include a server hostname")

        # Check for OAuth M2M credentials (client_id and client_secret)
        client_id = q.get("client_id", [None])[0]
        client_secret = q.get("client_secret", [None])[0]

        access_token: str
        if client_id and client_secret:
            # OAuth M2M authentication: exchange client credentials for access token
            access_token = get_databricks_oauth_token(
                server_hostname, client_id, client_secret
            )
        else:
            # Traditional token-based authentication
            if not p.password:
                raise ValueError(
                    "Databricks URI must include an access token or client_id/client_secret"
                )
            access_token = p.password

        creds = {
            "access_token": access_token,
            "server_hostname": server_hostname,
            "http_path": http_path,
            "catalog": catalog,
        }

        return dlt.destinations.databricks(
            credentials=creds,
            **kwargs,
        )

    def dlt_run_params(self, uri: str, table: str, **kwargs) -> dict:
        return super().dlt_run_params(uri, table, **kwargs)
