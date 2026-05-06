"""Tests for `embedding_cache.lookup` defensive-copy behaviour (B-PERF-4)."""

from __future__ import annotations

from pathlib import Path

import lancedb

from lxd.ingest.embedding_cache import lookup, open_cache_table, store


def _seed(path: Path, *, vector_size: int) -> object:
    db = lancedb.connect(path)
    table = open_cache_table(db, vector_size=vector_size)
    return table


def test_lookup_returns_defensive_copies(tmp_path: Path) -> None:
    """Mutating the returned list must not corrupt subsequent cache reads."""
    table = _seed(tmp_path / "cache", vector_size=4)
    store(
        table,
        chunk_hashes=["alpha"],
        vectors=[[0.1, 0.2, 0.3, 0.4]],
        embedding_model="test-model",
        embedding_dims=4,
    )

    first = lookup(
        table,
        chunk_hashes=["alpha"],
        embedding_model="test-model",
        embedding_dims=4,
    )
    assert 0 in first.hits
    first.hits[0][0] = 99.0  # mutate the returned vector

    second = lookup(
        table,
        chunk_hashes=["alpha"],
        embedding_model="test-model",
        embedding_dims=4,
    )
    # float32 round-trip introduces ~1e-8 noise; tolerance is wide enough
    # to accept that but tight enough to detect a 99.0 leak.
    assert abs(second.hits[0][0] - 0.1) < 0.01, (
        "Mutating the first lookup's returned vector should not corrupt the cache; "
        f"saw second.hits[0][0]={second.hits[0][0]}"
    )


def test_lookup_preserves_input_order_for_misses(tmp_path: Path) -> None:
    """Miss indices reflect input order, not the deduplicated query order."""
    table = _seed(tmp_path / "cache", vector_size=4)
    store(
        table,
        chunk_hashes=["alpha"],
        vectors=[[1.0, 2.0, 3.0, 4.0]],
        embedding_model="m",
        embedding_dims=4,
    )

    result = lookup(
        table,
        chunk_hashes=["miss-1", "alpha", "miss-2", "alpha"],
        embedding_model="m",
        embedding_dims=4,
    )

    assert result.hit_count == 2  # both alpha occurrences
    assert result.miss_count == 2
    assert result.misses_indices == [0, 2]
    assert result.hits[1] == [1.0, 2.0, 3.0, 4.0]
    assert result.hits[3] == [1.0, 2.0, 3.0, 4.0]


def test_lookup_returns_floats_not_other_numeric_types(tmp_path: Path) -> None:
    """Returned vector elements are Python floats so downstream `float()` calls don't break."""
    table = _seed(tmp_path / "cache", vector_size=4)
    store(
        table,
        chunk_hashes=["a"],
        vectors=[[1.5, 2.5, 3.5, 4.5]],
        embedding_model="m",
        embedding_dims=4,
    )

    result = lookup(
        table,
        chunk_hashes=["a"],
        embedding_model="m",
        embedding_dims=4,
    )

    assert all(isinstance(x, float) for x in result.hits[0]), (
        f"Vector elements must remain Python floats; saw types "
        f"{[type(x).__name__ for x in result.hits[0]]}"
    )
