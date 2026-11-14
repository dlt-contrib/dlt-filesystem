"""Which side of the Avro writer's Polars gate this environment is on.

Polars before 2.1 mis-frames a list column holding an empty list before a non-empty
one, so the Avro writer refuses on it, and Polars 2.1 needs Python 3.11. A test that
needs the Avro writer to succeed runs only where it writes; CI's 3.10 leg resolves
Polars 2.0 and proves the refusal instead. Hand-built Avro fixtures need no gate.
"""

import polars as pl
import pytest
from verlib2 import Version

#: Polars 2.1 fixed the list framing (pola-rs/polars#30061), and the writer refuses below it.
POLARS_WRITES_AVRO = Version(pl.__version__) >= Version("2.1")

needs_the_writer = pytest.mark.skipif(
    not POLARS_WRITES_AVRO,
    reason=f"Polars {pl.__version__} predates 2.1, where the writer refuses",
)
needs_an_early_polars = pytest.mark.skipif(
    POLARS_WRITES_AVRO,
    reason=f"Polars {pl.__version__} writes Avro; the refusal is for releases before 2.1",
)
