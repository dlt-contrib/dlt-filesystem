"""Cycle rejection must terminate even when Python's recursive-call check does not."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    ("file_format", "payload"),
    [
        pytest.param("yaml", b"a: &a [*a]\n", id="yaml-nested-list"),
        pytest.param("yaml", b"&a {self: *a}\n", id="yaml-root-map"),
        pytest.param("yaml", b"&a [*a]\n", id="yaml-root-list"),
        pytest.param("yaml", b"&a [{parent: *a}]\n", id="yaml-indirect"),
        pytest.param("yaml", b"&a [{id: 1}, *a]\n", id="yaml-before-first-row"),
        pytest.param("yaml", b"a: &a !!omap [{self: *a}]\n", id="yaml-tuple-edge"),
        # Tags 28/29 are shareable/sharedref. No decoder hooks are needed.
        pytest.param("cbor", bytes.fromhex("d81c81d81d00"), id="cbor-root-list"),
        pytest.param("cbor", bytes.fromhex("d81ca16161d81d00"), id="cbor-root-map"),
        pytest.param("cbor", bytes.fromhex("d81c81a16161d81d00"), id="cbor-indirect"),
        pytest.param("cbor", bytes.fromhex("d81cd904d281d81d00"), id="cbor-tag-tuple"),
        pytest.param(
            "cbor", bytes.fromhex("d81cd904d2a16178d81d00"), id="cbor-tag-frozen-map"
        ),
        pytest.param(
            "cbor", bytes.fromhex("d81cd904d2a1d81d0000"), id="cbor-tag-mapping-key"
        ),
        pytest.param(
            "cbor", bytes.fromhex("d81c82a162696401d81d00"), id="cbor-before-first-row"
        ),
    ],
)
def test_cyclic_input_is_rejected_before_first_row(tmp_path, file_format, payload):
    """A hard deadline bounds the Python 3.15 CALL_EX_PY recursion-check regression.

    Even a valid first row must not escape from a document with a later back-reference.
    These are real wire payloads, not hand-built objects injected after decoding.
    """
    pytest.importorskip({"yaml": "yaml", "cbor": "cbor2"}[file_format])
    path = tmp_path / f"cyclic.{file_format}"
    path.write_bytes(payload)
    code = """
import sys
from dlt_filesystem.source.error import CyclicInputError
from dlt_filesystem.source.format.iterable_codec import read_via_iterable
from dlt_filesystem.testing.stub import FileItemStub

reader = read_via_iterable(iter([FileItemStub(sys.argv[1])]),
                           file_format=sys.argv[2], chunksize=1)
try:
    next(reader)
except CyclicInputError as error:
    assert str(error) == "Cyclic input is not supported"
else:
    raise AssertionError("cyclic input emitted a row instead of raising")
"""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code, str(path), file_format],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_validator_handles_deep_and_shared_graphs_without_recursion():
    """Validation is iterative and checks a shared subgraph once, without mutating it."""
    code = """
from dlt_filesystem.source.error import CyclicInputError
from dlt_filesystem.source.format.iterable_codec import _reject_cycles

root = []
leaf = root
for _ in range(5000):
    child = []
    leaf.append(child)
    leaf = child
_reject_cycles(root)
leaf.append(root)
try:
    _reject_cycles(root)
except CyclicInputError:
    pass
else:
    raise AssertionError("deep cycle accepted")

shared = {"blob": b"AB"}
graph = shared
for _ in range(60):
    graph = [graph, graph]
_reject_cycles(graph)
assert shared == {"blob": b"AB"}
assert graph[0] is graph[1]
"""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
