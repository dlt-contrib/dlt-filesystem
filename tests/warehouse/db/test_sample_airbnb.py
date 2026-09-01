"""The nested-document reshape harness.

Loads the nested ``sample_airbnb``-shaped collection (see ``sample_airbnb.py``) from
Mongo into a SQL destination and pins the contract every transform lane must hit:

  - stock (no reshape): top-level arrays land as JSON columns, no child tables
    (omniload's Mongo behaviour: ``max_table_nesting=1`` + ``TypeHintMap``);
  - lane 1 (``--reshape python:...``): a plain 1->1 reshape plus a skipped TypeHintMap
    yields real ``__reviews`` / ``__amenities`` child tables, flattened lat/lng
    columns, decimal money, and a null for the absent ``weekly_price``;
  - lane 2 (``--reshape jq:...``, through Tikray), lane 3 (``--reshape polars:...``,
    through macropipe over Arrow) and lane 4 (the same reshape pushed into a Mongo
    aggregation) each have to reproduce lane 1's canonical target.

Only lane 1 preserves ``Decimal`` money: jq has no decimal type, the Polars lane casts
to ``Float64`` and the aggregation uses ``$toDouble``. The cross-lane checks therefore
run through a comparator that drops dlt metadata and normalizes numerics rather than
expecting byte-identical output.
"""

from collections import Counter
from decimal import Decimal

import pytest
import sqlalchemy
from testcontainers.mongodb import MongoDbContainer

from tests.util import invoke_ingest_command
from tests.util.common import get_random_string
from tests.warehouse.db import sample_airbnb
from tests.warehouse.manager import MONGODB_IMAGE, registry

# duckdb (embedded, fast) + postgres (real SQL DB). cratedb is reserved for the manual run.
AIRBNB_DESTS = {
    "duckdb": registry.duckdb_destination,
    "postgres": registry.postgresql,
}

RESHAPE_SPEC = "python:tests.warehouse.db.sample_airbnb:reshape"
RESHAPE_JQ_SPEC = "jq:" + sample_airbnb.RESHAPE_JQ
RESHAPE_POLARS_SPEC = "polars:" + sample_airbnb.RESHAPE_POLARS
# Lane 4 runs the value reshape server-side as a Mongo aggregation (passed via the source
# table's `collection:<pipeline-json>` form); the client pass is a no-op whose only job is
# to make omniload skip TypeHintMap so the arrays normalize into child tables.
RESHAPE_LANE4_SPEC = (
    "python:tests.warehouse.db.sample_airbnb:reshape_pushdown_client_passthrough"
)


@pytest.fixture(scope="module")
def mongo():
    container = MongoDbContainer(MONGODB_IMAGE)
    container.start()
    yield container
    container.stop()


def _tables_in_schema(engine, schema):
    with engine.connect() as conn:
        rows = conn.exec_driver_sql(
            f"select table_name from information_schema.tables where table_schema = '{schema}'"
        ).fetchall()
    return {r[0] for r in rows}


def _seed_collection(mongo):
    name = f"airbnb_{get_random_string(6)}"
    collection = mongo.get_connection_client()["test_db"][name]
    sample_airbnb.seed(collection)
    return name


@pytest.mark.parametrize(
    "dest", list(AIRBNB_DESTS.values()), ids=list(AIRBNB_DESTS.keys())
)
def test_sample_airbnb_stock(mongo, dest):
    """Without a reshape, top-level arrays stay JSON columns; no child tables form."""
    collection = _seed_collection(mongo)
    dest_uri = dest.start()

    result = invoke_ingest_command(
        mongo.get_connection_url(),
        f"test_db.{collection}",
        dest_uri,
        f"raw.{collection}",
    )
    assert result.exit_code == 0

    engine = sqlalchemy.create_engine(dest_uri)
    try:
        tables = _tables_in_schema(engine, "raw")
        assert collection in tables
        # stock = flat: arrays did NOT explode into child tables
        assert f"{collection}__reviews" not in tables
        assert f"{collection}__amenities" not in tables

        with engine.connect() as conn:
            count = conn.exec_driver_sql(
                f"select count(*) from raw.{collection}"
            ).scalar()
        assert count == sample_airbnb.EXPECTED_LISTINGS
    finally:
        engine.dispose()


@pytest.mark.parametrize(
    "dest", list(AIRBNB_DESTS.values()), ids=list(AIRBNB_DESTS.keys())
)
def test_sample_airbnb_reshape_lane1(mongo, dest):
    """Lane 1 reshape yields the canonical target: child tables + typed/flattened columns."""
    collection = _seed_collection(mongo)
    dest_uri = dest.start()

    result = invoke_ingest_command(
        mongo.get_connection_url(),
        f"test_db.{collection}",
        dest_uri,
        f"raw.{collection}",
        reshape=RESHAPE_SPEC,
    )
    assert result.exit_code == 0

    engine = sqlalchemy.create_engine(dest_uri)
    try:
        tables = _tables_in_schema(engine, "raw")
        # arrays exploded into child tables
        assert f"{collection}__reviews" in tables
        assert f"{collection}__amenities" in tables

        with engine.connect() as conn:
            rows = conn.exec_driver_sql(
                f"select name, price, weekly_price, address_location_lat, address_location_lng "
                f"from raw.{collection} order by name"
            ).fetchall()
            review_count = conn.exec_driver_sql(
                f"select count(*) from raw.{collection}__reviews"
            ).scalar()
            amenity_count = conn.exec_driver_sql(
                f"select count(*) from raw.{collection}__amenities"
            ).scalar()

        assert len(rows) == sample_airbnb.EXPECTED_LISTINGS
        assert review_count == sample_airbnb.EXPECTED_REVIEWS
        assert amenity_count == sample_airbnb.EXPECTED_AMENITIES

        by_name = {r[0]: r for r in rows}

        # coercion: Decimal128 money -> decimal column
        loft = by_name["Cozy downtown loft"]
        assert isinstance(loft[1], Decimal)
        assert loft[1] == Decimal("120.00")
        # flatten: GeoJSON coordinates -> typed lat/lng columns
        assert loft[3] == pytest.approx(-36.8485)
        assert loft[4] == pytest.approx(174.7633)

        # absent weekly_price loads as null (not an error)
        studio = by_name["Bare studio"]
        assert studio[2] is None
    finally:
        engine.dispose()


@pytest.mark.parametrize(
    "dest", list(AIRBNB_DESTS.values()), ids=list(AIRBNB_DESTS.keys())
)
def test_sample_airbnb_reshape_lane2_jq(mongo, dest):
    """Lane 2 (jq via Tikray) hits the same canonical target as lane 1.

    Same structural assertions as lane 1, but type-tolerant on money: jq has no Decimal
    type, so ``price`` arrives as a plain number rather than a ``Decimal``.
    """
    pytest.importorskip("tikray")
    collection = _seed_collection(mongo)
    dest_uri = dest.start()

    result = invoke_ingest_command(
        mongo.get_connection_url(),
        f"test_db.{collection}",
        dest_uri,
        f"raw.{collection}",
        reshape=RESHAPE_JQ_SPEC,
    )
    assert result.exit_code == 0

    engine = sqlalchemy.create_engine(dest_uri)
    try:
        tables = _tables_in_schema(engine, "raw")
        assert f"{collection}__reviews" in tables
        assert f"{collection}__amenities" in tables

        with engine.connect() as conn:
            rows = conn.exec_driver_sql(
                f"select name, price, weekly_price, address_location_lat, address_location_lng "
                f"from raw.{collection} order by name"
            ).fetchall()
            review_count = conn.exec_driver_sql(
                f"select count(*) from raw.{collection}__reviews"
            ).scalar()
            amenity_count = conn.exec_driver_sql(
                f"select count(*) from raw.{collection}__amenities"
            ).scalar()

        assert len(rows) == sample_airbnb.EXPECTED_LISTINGS
        assert review_count == sample_airbnb.EXPECTED_REVIEWS
        assert amenity_count == sample_airbnb.EXPECTED_AMENITIES

        by_name = {r[0]: r for r in rows}
        loft = by_name["Cozy downtown loft"]
        # money: numerically correct, but jq emits a plain number (no Decimal typing)
        assert float(loft[1]) == 120.0
        # flatten: GeoJSON coordinates -> typed lat/lng columns
        assert loft[3] == pytest.approx(-36.8485)
        assert loft[4] == pytest.approx(174.7633)
        # absent weekly_price still loads as null
        assert by_name["Bare studio"][2] is None
    finally:
        engine.dispose()


@pytest.mark.parametrize(
    "dest", list(AIRBNB_DESTS.values()), ids=list(AIRBNB_DESTS.keys())
)
def test_sample_airbnb_lane2_jq_matches_lane1(mongo, dest):
    """Lane 2 (jq) and lane 1 (python) produce the same canonical target.

    Both lanes load the same source collection into separate tables; the comparator then
    checks the root + child tables match once dlt metadata is excluded and numerics are
    normalized (so jq's number-typed money equals lane 1's Decimal money).
    """
    pytest.importorskip("tikray")
    collection = _seed_collection(mongo)
    dest_uri = dest.start()
    source_url = mongo.get_connection_url()

    py_table = f"{collection}_py"
    jq_table = f"{collection}_jq"
    for dest_table, spec in ((py_table, RESHAPE_SPEC), (jq_table, RESHAPE_JQ_SPEC)):
        result = invoke_ingest_command(
            source_url,
            f"test_db.{collection}",
            dest_uri,
            f"raw.{dest_table}",
            reshape=spec,
        )
        assert result.exit_code == 0

    engine = sqlalchemy.create_engine(dest_uri)
    try:
        assert_canonical_match(engine, "raw", py_table, jq_table)
    finally:
        engine.dispose()


@pytest.mark.parametrize(
    "dest", list(AIRBNB_DESTS.values()), ids=list(AIRBNB_DESTS.keys())
)
def test_sample_airbnb_reshape_lane3_polars(mongo, dest):
    """Lane 3 (macropipe/Polars over Arrow batches) hits the same canonical target.

    Unlike lanes 1-2, which map one document at a time, this lane extracts the
    collection as Arrow and
    reshapes each batch columnar via ``add_yield_map``. Same structural assertions as the
    other lanes, type-tolerant on money: Polars casts it to ``Float64`` (no Decimal), like jq.
    """
    pytest.importorskip("macropipe")
    pytest.importorskip("polars")
    collection = _seed_collection(mongo)
    dest_uri = dest.start()

    result = invoke_ingest_command(
        mongo.get_connection_url(),
        f"test_db.{collection}",
        dest_uri,
        f"raw.{collection}",
        reshape=RESHAPE_POLARS_SPEC,
    )
    assert result.exit_code == 0

    engine = sqlalchemy.create_engine(dest_uri)
    try:
        tables = _tables_in_schema(engine, "raw")
        assert f"{collection}__reviews" in tables
        assert f"{collection}__amenities" in tables

        with engine.connect() as conn:
            rows = conn.exec_driver_sql(
                f"select name, price, weekly_price, address_location_lat, address_location_lng "
                f"from raw.{collection} order by name"
            ).fetchall()
            review_count = conn.exec_driver_sql(
                f"select count(*) from raw.{collection}__reviews"
            ).scalar()
            amenity_count = conn.exec_driver_sql(
                f"select count(*) from raw.{collection}__amenities"
            ).scalar()

        assert len(rows) == sample_airbnb.EXPECTED_LISTINGS
        assert review_count == sample_airbnb.EXPECTED_REVIEWS
        assert amenity_count == sample_airbnb.EXPECTED_AMENITIES

        by_name = {r[0]: r for r in rows}
        loft = by_name["Cozy downtown loft"]
        # money: numerically correct, but Polars casts to Float64 (no Decimal typing)
        assert float(loft[1]) == 120.0
        # flatten: GeoJSON coordinates -> typed lat/lng columns
        assert loft[3] == pytest.approx(-36.8485)
        assert loft[4] == pytest.approx(174.7633)
        # absent weekly_price (arrives as the literal "None") casts to null, not an error
        assert by_name["Bare studio"][2] is None
    finally:
        engine.dispose()


@pytest.mark.parametrize(
    "dest", list(AIRBNB_DESTS.values()), ids=list(AIRBNB_DESTS.keys())
)
def test_sample_airbnb_lane3_polars_matches_lane1(mongo, dest):
    """Lane 3 (Polars batch) and lane 1 (python per-row) produce the same canonical target.

    The comparator excludes dlt metadata and normalizes numerics, so Polars' Float64 money
    equals lane 1's Decimal money and the differing execution model (Arrow batch vs per-row)
    is invisible in the final tables.
    """
    pytest.importorskip("macropipe")
    pytest.importorskip("polars")
    collection = _seed_collection(mongo)
    dest_uri = dest.start()
    source_url = mongo.get_connection_url()

    py_table = f"{collection}_py"
    polars_table = f"{collection}_polars"
    for dest_table, spec in (
        (py_table, RESHAPE_SPEC),
        (polars_table, RESHAPE_POLARS_SPEC),
    ):
        result = invoke_ingest_command(
            source_url,
            f"test_db.{collection}",
            dest_uri,
            f"raw.{dest_table}",
            reshape=spec,
        )
        assert result.exit_code == 0

    engine = sqlalchemy.create_engine(dest_uri)
    try:
        assert_canonical_match(engine, "raw", py_table, polars_table)
    finally:
        engine.dispose()


@pytest.mark.parametrize(
    "dest", list(AIRBNB_DESTS.values()), ids=list(AIRBNB_DESTS.keys())
)
def test_sample_airbnb_reshape_lane4_pushdown(mongo, dest):
    """Lane 4 (Mongo ``$``-pipeline pushdown + thin client pass) hits the canonical target.

    The value reshape runs server-side as a Mongo aggregation (flatten coords, ``$toDouble``
    money, project reviews/amenities as lists) passed via the source table's
    ``collection:<pipeline-json>`` form. A no-op client ``--reshape`` is paired only to make
    omniload skip TypeHintMap, so the arrays normalize into child tables. Type-tolerant on
    money: ``$toDouble`` drops Decimal typing, like jq/Polars.
    """
    collection = _seed_collection(mongo)
    dest_uri = dest.start()
    source_table = sample_airbnb.pushdown_source_table(f"test_db.{collection}")

    result = invoke_ingest_command(
        mongo.get_connection_url(),
        source_table,
        dest_uri,
        f"raw.{collection}",
        reshape=RESHAPE_LANE4_SPEC,
    )
    assert result.exit_code == 0

    engine = sqlalchemy.create_engine(dest_uri)
    try:
        tables = _tables_in_schema(engine, "raw")
        assert f"{collection}__reviews" in tables
        assert f"{collection}__amenities" in tables

        with engine.connect() as conn:
            rows = conn.exec_driver_sql(
                f"select name, price, weekly_price, address_location_lat, address_location_lng "
                f"from raw.{collection} order by name"
            ).fetchall()
            review_count = conn.exec_driver_sql(
                f"select count(*) from raw.{collection}__reviews"
            ).scalar()
            amenity_count = conn.exec_driver_sql(
                f"select count(*) from raw.{collection}__amenities"
            ).scalar()

        assert len(rows) == sample_airbnb.EXPECTED_LISTINGS
        assert review_count == sample_airbnb.EXPECTED_REVIEWS
        assert amenity_count == sample_airbnb.EXPECTED_AMENITIES

        by_name = {r[0]: r for r in rows}
        loft = by_name["Cozy downtown loft"]
        # money: numerically correct, but $toDouble emits a plain number (no Decimal typing)
        assert float(loft[1]) == 120.0
        # flatten: GeoJSON coordinates -> typed lat/lng columns, done server-side
        assert loft[3] == pytest.approx(-36.8485)
        assert loft[4] == pytest.approx(174.7633)
        # absent weekly_price ($toDouble of a missing field) loads as null
        assert by_name["Bare studio"][2] is None
        # absent address (L4): the outer $ifNull pins lat/lng to an explicit server-side
        # null, so both flatten columns are null rather than the row being dropped/erroring.
        mystery = by_name["Mystery rental"]
        assert mystery[3] is None  # address_location_lat
        assert mystery[4] is None  # address_location_lng
    finally:
        engine.dispose()


@pytest.mark.parametrize(
    "dest", list(AIRBNB_DESTS.values()), ids=list(AIRBNB_DESTS.keys())
)
def test_sample_airbnb_lane4_pushdown_matches_lane1(mongo, dest):
    """Lane 4 (Mongo pushdown) and lane 1 (python per-row) produce the same canonical target.

    The comparator excludes dlt metadata and normalizes numerics, so lane 4's ``$toDouble``
    money equals lane 1's Decimal money and the fact that the reshape ran in Mongo (vs a
    client-side pass) is invisible in the final tables.
    """
    collection = _seed_collection(mongo)
    dest_uri = dest.start()
    source_url = mongo.get_connection_url()

    py_table = f"{collection}_py"
    lane4_table = f"{collection}_lane4"
    for dest_table, source_table, spec in (
        (py_table, f"test_db.{collection}", RESHAPE_SPEC),
        (
            lane4_table,
            sample_airbnb.pushdown_source_table(f"test_db.{collection}"),
            RESHAPE_LANE4_SPEC,
        ),
    ):
        result = invoke_ingest_command(
            source_url,
            source_table,
            dest_uri,
            f"raw.{dest_table}",
            reshape=spec,
        )
        assert result.exit_code == 0

    engine = sqlalchemy.create_engine(dest_uri)
    try:
        assert_canonical_match(engine, "raw", py_table, lane4_table)
    finally:
        engine.dispose()


@pytest.mark.parametrize(
    "dest", list(AIRBNB_DESTS.values()), ids=list(AIRBNB_DESTS.keys())
)
def test_sample_airbnb_lane4_pushdown_only_reaches_root(mongo, dest):
    """Pushdown ALONE reaches the flattened/coerced root but NOT the child tables.

    Without a client ``--reshape``, omniload's TypeHintMap json-hints the top-level arrays
    into JSON columns, so reviews/amenities do not explode into child tables even though the
    aggregation emitted them as real lists. This pins the lane-4 readout: server-side
    ``$``-pipeline pushdown moves the scalar flatten/coerce work into Mongo, but the
    child-table normalization is a client-side decision a thin pass still has to unlock
    (contrast ``test_sample_airbnb_reshape_lane4_pushdown``, same pipeline + a no-op pass).
    """
    collection = _seed_collection(mongo)
    dest_uri = dest.start()
    source_table = sample_airbnb.pushdown_source_table(f"test_db.{collection}")

    result = invoke_ingest_command(
        mongo.get_connection_url(),
        source_table,
        dest_uri,
        f"raw.{collection}",
    )
    assert result.exit_code == 0

    engine = sqlalchemy.create_engine(dest_uri)
    try:
        tables = _tables_in_schema(engine, "raw")
        assert collection in tables
        # pushdown alone is flat: arrays stayed JSON columns, no child tables formed
        assert f"{collection}__reviews" not in tables
        assert f"{collection}__amenities" not in tables

        with engine.connect() as conn:
            count = conn.exec_driver_sql(
                f"select count(*) from raw.{collection}"
            ).scalar()
            rows = conn.exec_driver_sql(
                f"select name, price, address_location_lat, address_location_lng "
                f"from raw.{collection} order by name"
            ).fetchall()

        assert count == sample_airbnb.EXPECTED_LISTINGS
        # but the root WAS reshaped server-side: flattened lat/lng + numeric (coerced) price
        by_name = {r[0]: r for r in rows}
        loft = by_name["Cozy downtown loft"]
        assert float(loft[1]) == 120.0
        assert loft[2] == pytest.approx(-36.8485)
        assert loft[3] == pytest.approx(174.7633)
    finally:
        engine.dispose()


# --- cross-lane comparator (reused by lanes 3-4) -----------------------------------------
#
# Compares two independent loads of the canonical target. dlt assigns fresh row ids and
# load ids per load, so equality is only meaningful once those are excluded and numerics
# normalized; rows are compared as multisets (order-independent).
#
# Scope: this checks VALUE equivalence, not SQL type/scale fidelity. Numerics are compared
# loosely on purpose (Decimal/int/float collapse, floats rounded to 9 dp) so jq's
# number-typed money equals lane 1's Decimal money. It deliberately does not assert column
# types or decimal scale; that difference is a documented #112 readout point, not a failure.

_DLT_META_PREFIX = "_dlt_"


def _norm(value):
    """Normalize a cell for cross-lane comparison: Decimal/float -> rounded float."""
    if isinstance(value, Decimal):
        value = float(value)
    if isinstance(value, float):
        # collapse last-digit float drift; 120.0 == 120 stays true in Python anyway
        return round(value, 9)
    return value


def _business_columns(engine, schema, table):
    """Column names of ``table`` minus dlt bookkeeping columns, sorted for stable order."""
    with engine.connect() as conn:
        rows = conn.exec_driver_sql(
            f"select column_name from information_schema.columns "
            f"where table_schema = '{schema}' and table_name = '{table}'"
        ).fetchall()
    return sorted(c[0] for c in rows if not c[0].startswith(_DLT_META_PREFIX))


def _quoted(cols):
    return ", ".join(f'"{c}"' for c in cols)


def _root_multiset(engine, schema, table):
    cols = _business_columns(engine, schema, table)
    with engine.connect() as conn:
        rows = conn.exec_driver_sql(
            f"select {_quoted(cols)} from {schema}.{table}"
        ).fetchall()
    return cols, Counter(tuple(_norm(v) for v in r) for r in rows)


def _child_multiset(engine, schema, root_table, child_table):
    """Child rows keyed by their full parent business row, so a child can't silently move.

    Keying on the whole parent row (not a single column like ``name``) avoids any uniqueness
    assumption: real listing names repeat, so a name-only key could pool children across
    distinct listings and mask a mis-assignment.
    """
    parent_cols = _business_columns(engine, schema, root_table)
    child_cols = _business_columns(engine, schema, child_table)
    select = ", ".join(
        [f'p."{c}"' for c in parent_cols] + [f'c."{c}"' for c in child_cols]
    )
    with engine.connect() as conn:
        rows = conn.exec_driver_sql(
            f"select {select} from {schema}.{child_table} c "
            f'join {schema}.{root_table} p on c."_dlt_parent_id" = p."_dlt_id"'
        ).fetchall()
    return child_cols, Counter(tuple(_norm(v) for v in r) for r in rows)


def assert_canonical_match(engine, schema, table_a, table_b):
    """Assert two loaded canonical-target tables (root + child tables) are equivalent."""
    cols_a, root_a = _root_multiset(engine, schema, table_a)
    cols_b, root_b = _root_multiset(engine, schema, table_b)
    assert cols_a == cols_b, f"root columns differ: {cols_a} vs {cols_b}"
    assert root_a == root_b, (
        f"root rows differ:\n  only in {table_a}: {root_a - root_b}\n"
        f"  only in {table_b}: {root_b - root_a}"
    )

    tables = _tables_in_schema(engine, schema)
    for suffix in ("__reviews", "__amenities"):
        a_has = f"{table_a}{suffix}" in tables
        b_has = f"{table_b}{suffix}" in tables
        # dlt only creates a child table when the list had elements; an all-empty extract
        # produces neither, which is still a match. A one-sided table is a real divergence.
        assert a_has == b_has, (
            f"{suffix} present for only one lane: {table_a}={a_has}, {table_b}={b_has}"
        )
        if not a_has:
            continue
        child_cols_a, child_a = _child_multiset(
            engine, schema, table_a, f"{table_a}{suffix}"
        )
        child_cols_b, child_b = _child_multiset(
            engine, schema, table_b, f"{table_b}{suffix}"
        )
        assert child_cols_a == child_cols_b, (
            f"{suffix} columns differ: {child_cols_a} vs {child_cols_b}"
        )
        assert child_a == child_b, (
            f"{suffix} rows differ:\n  only in {table_a}: {child_a - child_b}\n"
            f"  only in {table_b}: {child_b - child_a}"
        )
