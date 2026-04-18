"""Performance regression gates for Wave 10.

These tests are skipped from the default suite because pytest-benchmark is
run with ``--benchmark-only`` in CI. Each benchmark targets a hot path that
showed up in the Wave 5 / Wave 9 profiling: batch ID escaping for LanceDB,
``IN``-clause construction for SQLite, and blake3 digest hashing of small
payloads used by ``config.lock``.
"""

from __future__ import annotations

import pytest

from lxd.domain.ids import blake3_hex
from lxd.stores.lance_sql import in_clause as lance_in_clause
from lxd.stores.sql_helpers import in_clause

pytestmark = [pytest.mark.unit, pytest.mark.benchmark]


def test_bench_sqlite_in_clause(benchmark: pytest.FixtureRequest) -> None:
    """Placeholder generation for a 1 000-element ``IN`` clause.

    A 1k cohort matches the upper bound we pass to LanceDB during chunk
    resolution and is a realistic worst-case bucket for SQLite queries.
    """
    benchmark(lambda: in_clause(1000))  # type: ignore[operator]


def test_bench_lance_in_clause(benchmark: pytest.FixtureRequest) -> None:
    """LanceDB ``IN`` clause construction with 200 escaped chunk IDs."""
    values = [f"chunk-{i:05d}" for i in range(200)]
    benchmark(lambda: lance_in_clause("chunk_id", values))  # type: ignore[operator]


def test_bench_blake3_digest_small_payload(benchmark: pytest.FixtureRequest) -> None:
    """Blake3 digest for a typical JSON-normalised config payload.

    Tracks the cost of :func:`lxd.app.bootstrap.compute_config_digest`
    which runs on every CLI/MCP startup.
    """
    payload = "{" + ",".join(f'"{i}":"{i}"' for i in range(32)) + "}"
    benchmark(lambda: blake3_hex(payload))  # type: ignore[operator]
