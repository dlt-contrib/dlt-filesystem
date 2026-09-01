"""A `PyMongoArrowContext` is per batch, not per cursor.

The Arrow loaders in `source/mongodb/helpers.py` call `context.finish()` once per
wire batch. `finish()` does not reset the builders the context has accumulated, so
one context shared across batches replays the earlier batch's slots as all-null
rows. That is invisible on a collection small enough to arrive in a single batch,
which is every fixture here, and shows up in production as phantom rows or a
primary-key violation.

Mock-only unit lane: this pins the library behaviour the loaders have to work
around, so the per-batch construction cannot be "simplified" back out.
"""

import pytest

pytest.importorskip("pymongoarrow")

import bson  # noqa: E402
from bson.codec_options import CodecOptions  # noqa: E402
from pymongoarrow.context import PyMongoArrowContext  # noqa: E402


def _raw_batch(documents):
    """One `find_raw_batches`-shaped BSON stream."""
    return b"".join(bson.encode(document) for document in documents)


def _finish(context, documents):
    context.process_bson_stream(_raw_batch(documents))
    return context.finish().to_pylist()


def test_a_reused_context_replays_the_previous_batch_as_null_rows():
    """The library behaviour the loaders must not rely on."""
    context = PyMongoArrowContext(schema=None, codec_options=CodecOptions())

    assert _finish(context, [{"_id": 1}]) == [{"_id": 1}]
    assert _finish(context, [{"_id": 2}]) == [{"_id": None}, {"_id": 2}]


def test_a_context_per_batch_yields_exactly_its_own_rows():
    """What the loaders do instead."""
    rows = []
    for batch in ([{"_id": 1}], [{"_id": 2}], [{"_id": 3}]):
        context = PyMongoArrowContext(schema=None, codec_options=CodecOptions())
        rows.extend(_finish(context, batch))

    assert rows == [{"_id": 1}, {"_id": 2}, {"_id": 3}]


def test_the_arrow_loaders_build_a_context_per_batch():
    """Pin the call sites, so the fix cannot be undone without failing here."""
    import inspect

    from omniload.source.mongodb.helpers import (
        CollectionArrowLoader,
        CollectionArrowLoaderParallel,
    )

    for owner, method in (
        (CollectionArrowLoader, "load_documents"),
        (CollectionArrowLoaderParallel, "_run_batch"),
    ):
        lines = inspect.getsource(getattr(owner, method)).splitlines()
        loop = next(
            i for i, line in enumerate(lines) if line.lstrip().startswith("for ")
        )
        built = next(
            i for i, line in enumerate(lines) if "PyMongoArrowContext(" in line
        )
        assert loop < built, (
            f"{owner.__name__}.{method}: the context is built outside the batch "
            "loop, so finish() replays earlier batches as all-null rows"
        )
