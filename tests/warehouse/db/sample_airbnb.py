"""Fixtures + the lane-1 (pure-Python) reshape for the issue #112 transform harness.

Models the shape of the MongoDB Atlas ``sample_airbnb.listingsAndReviews`` collection: a
deeply nested document (GeoJSON location, ``reviews[]`` / ``amenities[]`` arrays, BSON
``Decimal128`` money, some money fields absent). The ``reshape`` here is the canonical
target contract every other lane (jq, Polars, Mongo pushdown) must reproduce.

These are hand-built representative documents, not a real Atlas extract, so the test runs
without Atlas access. Swap in a real extract later without changing the contract.
"""

from decimal import Decimal

from bson.decimal128 import Decimal128

# Representative listings covering the awkward shapes:
#  L1: full (money + weekly_price, 2 reviews, 2 amenities)
#  L2: weekly_price ABSENT, 1 review, 1 amenity
#  L3: empty reviews[], 3 amenities, weekly_price present
#  L4: address ABSENT and both money fields ABSENT (null-row safety: all lanes must emit
#      null lat/lng/price for it, not raise). These docs share one extraction batch, so the
#      columns still exist in the inferred Arrow schema; only the row values are null.
SAMPLE_DOCS: list[dict] = [
    {
        "_id": "L1",
        "name": "Cozy downtown loft",
        "price": Decimal128("120.00"),
        "weekly_price": Decimal128("700.00"),
        "address": {
            "street": "1 Queen St",
            "location": {"type": "Point", "coordinates": [174.7633, -36.8485]},
        },
        "amenities": ["Wifi", "Kitchen"],
        "reviews": [
            {"reviewer_id": "r1", "comments": "Great stay"},
            {"reviewer_id": "r2", "comments": "Would return"},
        ],
    },
    {
        "_id": "L2",
        "name": "Bare studio",
        "price": Decimal128("55.00"),
        # weekly_price absent
        "address": {
            "street": "2 George St",
            "location": {"type": "Point", "coordinates": [151.2093, -33.8688]},
        },
        "amenities": ["Wifi"],
        "reviews": [
            {"reviewer_id": "r3", "comments": "Compact but clean"},
        ],
    },
    {
        "_id": "L3",
        "name": "Seaside bungalow",
        "price": Decimal128("210.00"),
        "weekly_price": Decimal128("1300.00"),
        "address": {
            "street": "3 Marine Pde",
            "location": {"type": "Point", "coordinates": [144.9631, -37.8136]},
        },
        "amenities": ["Wifi", "Parking", "Pool"],
        "reviews": [],
    },
    {
        "_id": "L4",
        "name": "Mystery rental",
        # price, weekly_price and address all absent
        "amenities": ["Wifi"],
        "reviews": [
            {"reviewer_id": "r4", "comments": "No address on file"},
        ],
    },
]

# Derived expectations the test asserts against (kept here so the contract lives in one place).
EXPECTED_LISTINGS = len(SAMPLE_DOCS)
EXPECTED_REVIEWS = sum(len(d["reviews"]) for d in SAMPLE_DOCS)
EXPECTED_AMENITIES = sum(len(d["amenities"]) for d in SAMPLE_DOCS)


def _to_decimal(value):
    """Coerce BSON Decimal128 (and friends) to a plain Decimal; pass None through."""
    if value is None:
        return None
    if isinstance(value, Decimal128):
        return value.to_decimal()
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _to_str(value):
    """Stringify an id, preserving null (so it matches jq's null-preserving ``tostring``).

    Plain ``str(None)`` would yield ``"None"`` while jq yields ``null``; keeping null aligns
    the lanes when a real ``reviewer_id`` is missing.
    """
    return None if value is None else str(value)


def _coordinate(doc, index):
    coords = ((doc.get("address") or {}).get("location") or {}).get("coordinates") or []
    return coords[index] if len(coords) > index else None


def reshape(doc: dict) -> dict:
    """Lane 1: pure-Python 1->1 reshape to the canonical target.

    Flattens ``address.location`` GeoJSON into lat/lng columns, coerces Decimal128 money to
    Decimal, and leaves ``reviews`` / ``amenities`` as real lists so dlt's normalizer creates
    ``<table>__reviews`` / ``<table>__amenities`` child tables (the reshape lane skips
    TypeHintMap, so the arrays are not json-hinted). ``_id`` is preserved because the Mongo
    adapter marks it as the non-nullable primary key.
    """
    return {
        "_id": _to_str(doc.get("_id")),
        "name": doc.get("name"),
        "price": _to_decimal(doc.get("price")),
        "weekly_price": _to_decimal(doc.get("weekly_price")),
        "address_location_lng": _coordinate(doc, 0),
        "address_location_lat": _coordinate(doc, 1),
        "amenities": list(doc.get("amenities") or []),
        "reviews": [
            {
                "reviewer_id": _to_str(r.get("reviewer_id")),
                "comments": r.get("comments"),
            }
            for r in (doc.get("reviews") or [])
        ],
    }


# Lane 2: the same canonical target expressed as a jq program (run via Tikray, see
# omniload/codec/reshape.py). It mirrors reshape() above: flatten the GeoJSON location
# into lat/lng, keep reviews/amenities as real lists (so dlt builds child tables), and
# stringify ids. The `[0]? // null` guards tolerate a missing address like reshape()'s
# _coordinate() does.
#
# Money: omniload's Mongo source hands BSON Decimal128 to the transform as a string (e.g.
# "120.00") to keep its precision through JSON. reshape() parses that back to Decimal;
# the jq lane uses `tonumber` (jq has no Decimal type, so it becomes a plain number).
# The cross-lane comparator therefore normalizes numerics rather than expecting Decimal.
RESHAPE_JQ = """
{
  _id: (._id | if . == null then null else tostring end),
  name: .name,
  price: (.price | if . == null then null else tonumber end),
  weekly_price: (.weekly_price | if . == null then null else tonumber end),
  address_location_lng: (.address.location.coordinates[0]? // null),
  address_location_lat: (.address.location.coordinates[1]? // null),
  amenities: (.amenities // []),
  reviews: [ (.reviews // [])[] | {reviewer_id: (.reviewer_id | if . == null then null else tostring end), comments: .comments} ]
}
"""


# Lane 3: the same canonical target as a macropipe recipe list (compiled Polars over Arrow
# batches, see omniload/codec/reshape.py). One recipe per line:
#   - geojson_point_flatten: drill address.location.coordinates [lng, lat] into typed columns
#     (custom omniload recipe; macropipe has no nested-struct flatten),
#   - cast_number: non-strict Float64 cast of the money columns (the Arrow path delivers
#     Decimal128 as a string, and an absent value as the literal "None", so a strict cast
#     would raise; custom omniload recipe because macropipe's builtin cast is strict),
#   - select: keep the canonical columns, leaving reviews/amenities as real lists so dlt
#     builds the child tables.
# Money loses Decimal typing here (Float64), exactly like the jq lane; the comparator
# normalizes numerics so it still matches lane 1.
RESHAPE_POLARS = """
geojson_point_flatten:address:address_location_lng:address_location_lat
cast_number:price,weekly_price
select:_id,name,price,weekly_price,address_location_lng,address_location_lat,amenities,reviews
"""


# Lane 4: the same canonical target pushed DOWN into a MongoDB aggregation pipeline, run
# server-side via the source's `collection:<pipeline-json>` form (omniload/source/mongodb/
# api.py) instead of a client-side --reshape. The pipeline does the value work in Mongo:
#   - $addFields flattens address.location.coordinates [lng, lat] into typed columns and
#     coerces the Decimal128 money with $toDouble (absent money -> null). $toDouble, not
#     $toDecimal, on purpose: omniload's Mongo loader stringifies Decimal128, so a
#     server-side decimal wouldn't survive as a number. Like jq/Polars, lane 4 loses
#     Decimal typing; the comparator normalizes numerics so it still matches lane 1.
#   - $project keeps the canonical columns, drops address, and $map projects each review to
#     {reviewer_id, comments}, leaving reviews/amenities as real lists. $ifNull guards make
#     the absent-address / absent-reviews row (L4) emit null/empty instead of erroring.
# What pushdown CANNOT do alone: omniload still applies TypeHintMap (which json-hints
# top-level arrays into JSON columns) unless a --reshape is active, so the arrays only
# normalize into __reviews/__amenities child tables when the pushdown is paired with a thin
# client pass (reshape_pushdown_client_passthrough). See the lane-4 tests and the plan.
PUSHDOWN_PIPELINE: list[dict] = [
    {
        "$addFields": {
            # $arrayElemAt over an absent (-> []) array yields "missing", which would omit
            # the field; the outer $ifNull pins it to an explicit null so the absent-address
            # row (L4) emits null lat/lng like lane 1 does, instead of relying on dlt to
            # backfill the column from sibling rows.
            "address_location_lng": {
                "$ifNull": [
                    {
                        "$arrayElemAt": [
                            {"$ifNull": ["$address.location.coordinates", []]},
                            0,
                        ]
                    },
                    None,
                ]
            },
            "address_location_lat": {
                "$ifNull": [
                    {
                        "$arrayElemAt": [
                            {"$ifNull": ["$address.location.coordinates", []]},
                            1,
                        ]
                    },
                    None,
                ]
            },
            "price": {"$toDouble": "$price"},
            "weekly_price": {"$toDouble": "$weekly_price"},
        }
    },
    {
        "$project": {
            "_id": 1,
            "name": 1,
            "price": 1,
            "weekly_price": 1,
            "address_location_lng": 1,
            "address_location_lat": 1,
            "amenities": 1,
            "reviews": {
                "$map": {
                    "input": {"$ifNull": ["$reviews", []]},
                    "as": "r",
                    "in": {
                        "reviewer_id": "$$r.reviewer_id",
                        "comments": "$$r.comments",
                    },
                }
            },
        }
    },
]


def pushdown_source_table(collection_path: str) -> str:
    """Build the ``collection:<pipeline-json>`` source-table string for the lane-4 pushdown.

    ``collection_path`` is the qualified ``database.collection`` (e.g. ``test_db.airbnb_x``);
    the rendered aggregation pipeline is appended after the first colon, which the MongoDB
    source recognizes as a custom aggregation query. The name must not itself contain a
    ``:`` (the source splits the source-table string on the first colon).
    """
    from bson import json_util

    return f"{collection_path}:{json_util.dumps(PUSHDOWN_PIPELINE)}"


def reshape_pushdown_client_passthrough(doc: dict) -> dict:
    """Lane 4's thin client pass: identity.

    The lane-4 aggregation already produced the canonical shape server-side (flattened
    lat/lng, ``$toDouble`` money, reviews/amenities projected as lists, ``_id`` preserved),
    so this map does nothing to the document. It exists only so that a ``--reshape`` is
    active, which makes omniload skip ``TypeHintMap`` so the top-level arrays normalize into
    ``__reviews`` / ``__amenities`` child tables instead of being json-hinted into JSON
    columns. A first-class "pushdown lane" / "skip type hints" flag would remove the need
    for this no-op (issue #112 readout).
    """
    return doc


def seed(collection, docs=None):
    """Insert the sample documents into a pymongo collection."""
    import copy

    collection.insert_many(copy.deepcopy(docs if docs is not None else SAMPLE_DOCS))
