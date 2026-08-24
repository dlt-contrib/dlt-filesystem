import pytest

from omniload.core.tablename import (
    Capability,
    Defaults,
    TableName,
    opaque,
    prefix_split,
    split,
    three_level,
    two_level,
    uri_with_database,
)
from omniload.error import ValidationError


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("users", ["users"]),
        ("public.users", ["public", "users"]),
        ("db.public.users", ["db", "public", "users"]),
        (" [my db] . [dbo] . [my.table] ", ["my db", "dbo", "my.table"]),
        ('"my.schema"."table"', ["my.schema", "table"]),
        ("`my.db`.`table`", ["my.db", "table"]),
        ("[a]]b].table", ["a]b", "table"]),
        ('"a""b".table', ['a"b', "table"]),
        ("`a``b`.table", ["a`b", "table"]),
    ],
)
def test_split_qualified_names(raw, expected):
    assert split(raw) == expected


def test_split_rejects_an_unterminated_quoted_identifier():
    with pytest.raises(ValidationError, match="Unterminated"):
        split('schema."table')


def test_table_name_string_and_qualified_schema():
    parsed = TableName(catalog="warehouse", schema="analytics", table="events")

    assert str(parsed) == "warehouse.analytics.events"
    assert parsed.qualified_schema() == "warehouse.analytics"
    assert TableName(None, "analytics", "events").qualified_schema() == "analytics"


def test_parse_right_aligns_and_applies_defaults():
    capability = three_level("warehouse", min_components=1)
    defaults = Defaults(catalog="default_catalog", schema="default_schema")

    assert capability.parse("events", defaults) == TableName(
        "default_catalog", "default_schema", "events"
    )
    assert capability.parse("analytics.events", defaults) == TableName(
        "default_catalog", "analytics", "events"
    )
    assert capability.parse("warehouse.analytics.events", defaults) == TableName(
        "warehouse", "analytics", "events"
    )


@pytest.mark.parametrize("raw", ["table", "catalog.schema.table", "schema..table"])
def test_two_level_rejects_invalid_component_counts_and_empty_components(raw):
    with pytest.raises(ValidationError):
        two_level("postgres").parse(raw)


def test_unbounded_capability_joins_leading_catalog_components():
    capability = Capability(
        platform="mssql",
        min_components=2,
        max_components=3,
        labels=("database", "schema", "table"),
        format_desc="<database>.<schema>.<table>",
        unbounded=True,
    )

    assert capability.parse("server.database.dbo.users") == TableName(
        "server.database", "dbo", "users"
    )


def test_capability_error_names_platform_format_and_hint():
    capability = two_level("duckdb", "Select the database file with --dest-uri.")

    with pytest.raises(ValidationError) as exc_info:
        capability.parse("catalog.schema.table")

    message = str(exc_info.value)
    assert "duckdb" in message
    assert "<schema>.<table>" in message
    assert "--dest-uri" in message


@pytest.mark.parametrize(
    ("capability", "raw", "expected"),
    [
        (
            prefix_split("gsheets", ("spreadsheet", "range")),
            "spreadsheet_id.'Q3.2026'!A1:D5",
            TableName(None, "spreadsheet_id", "'Q3.2026'!A1:D5"),
        ),
        (
            prefix_split("mongodb", ("database", "collection")),
            "db.audit.2026",
            TableName(None, "db", "audit.2026"),
        ),
    ],
)
def test_prefix_split_keeps_the_remainder_opaque(capability, raw, expected):
    assert capability.parse(raw) == expected


def test_opaque_capability_keeps_dots_and_leading_dot():
    capability = opaque("elasticsearch", table_label="index")

    assert capability.parse("filebeat-2026.03.15").table == "filebeat-2026.03.15"
    assert capability.parse(".kibana").table == ".kibana"


def test_whole_string_backtick_path_has_targeted_error():
    with pytest.raises(ValidationError, match="Quote each component separately"):
        three_level("bigquery", schema_label="dataset").parse("`project.dataset.table`")


def test_component_wise_backticks_are_accepted():
    assert three_level("bigquery", schema_label="dataset").parse(
        "`project`.`dataset`.`table`"
    ) == TableName("project", "dataset", "table")


def test_a_backtick_quoted_component_may_itself_contain_a_dot():
    """One quoted component with a dot is not the whole-path spelling."""
    assert three_level("bigquery", schema_label="dataset").parse(
        "`my.project`.`dataset`.`table`"
    ) == TableName("my.project", "dataset", "table")
    assert two_level("mysql", schema_label="database").parse(
        "`a.b`.table"
    ) == TableName(None, "a.b", "table")


def test_a_whole_path_in_one_backtick_pair_names_the_alternative():
    """The error has to say what to write instead, for either reading."""
    with pytest.raises(ValidationError) as excinfo:
        three_level("bigquery", schema_label="dataset").parse("`project.dataset.table`")

    message = str(excinfo.value)
    assert "Quote each component separately" in message
    assert "double quotes or square brackets" in message


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        (
            "postgres://user@host/old?sslmode=require",
            "postgres://user@host/new?sslmode=require",
        ),
        ("trino://user@host/old/schema", "trino://user@host/new/schema"),
        ("snowflake://user@account", "snowflake://user@account/new"),
    ],
)
def test_uri_with_database_replaces_the_first_path_component(uri, expected):
    assert uri_with_database(uri, "new") == expected
