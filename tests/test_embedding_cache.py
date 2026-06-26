"""Tests for the content-addressed embedding cache.

Cache key is ``(chunk_hash, embedding_model, embedding_dims)``. Hits avoid
the API call; misses go through the live embedder and are stored back.
"""

from pathlib import Path

from lxd.ingest.embedding_cache import lookup, open_cache_table, store
from lxd.stores.lancedb import connect_lancedb


def test_cache_round_trip_hit_and_miss(tmp_path: Path) -> None:
    db = connect_lancedb(tmp_path / "lancedb")
    cache = open_cache_table(db, vector_size=3)

    # Store two entries.
    written = store(
        cache,
        chunk_hashes=["hash-a", "hash-b"],
        vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        embedding_model="test-embed",
        embedding_dims=3,
    )
    assert written == 2

    # Look up a mix: one hit, one miss.
    result = lookup(
        cache,
        chunk_hashes=["hash-a", "hash-not-stored"],
        embedding_model="test-embed",
        embedding_dims=3,
    )
    assert result.hit_count == 1
    assert result.miss_count == 1
    assert result.hits[0] == [1.0, 0.0, 0.0]
    assert result.misses_indices == [1]


def test_cache_key_is_model_specific(tmp_path: Path) -> None:
    """Same chunk_hash with a different model must miss — that's the whole
    point of the model+dims being part of the cache key."""
    db = connect_lancedb(tmp_path / "lancedb")
    cache = open_cache_table(db, vector_size=3)
    store(
        cache,
        chunk_hashes=["hash-a"],
        vectors=[[1.0, 0.0, 0.0]],
        embedding_model="model-v1",
        embedding_dims=3,
    )
    result = lookup(
        cache,
        chunk_hashes=["hash-a"],
        embedding_model="model-v2",  # different model
        embedding_dims=3,
    )
    assert result.hit_count == 0
    assert result.miss_count == 1


def test_cache_store_is_idempotent(tmp_path: Path) -> None:
    """Re-storing the same key replaces, does not duplicate."""
    db = connect_lancedb(tmp_path / "lancedb")
    cache = open_cache_table(db, vector_size=3)
    store(
        cache,
        chunk_hashes=["hash-a"],
        vectors=[[1.0, 0.0, 0.0]],
        embedding_model="m",
        embedding_dims=3,
    )
    store(
        cache,
        chunk_hashes=["hash-a"],
        vectors=[[2.0, 0.0, 0.0]],
        embedding_model="m",
        embedding_dims=3,
    )
    result = lookup(
        cache,
        chunk_hashes=["hash-a"],
        embedding_model="m",
        embedding_dims=3,
    )
    assert result.hit_count == 1
    # Latest write wins.
    assert result.hits[0] == [2.0, 0.0, 0.0]


def test_cache_lookup_with_empty_input_returns_empty(tmp_path: Path) -> None:
    db = connect_lancedb(tmp_path / "lancedb")
    cache = open_cache_table(db, vector_size=3)
    result = lookup(
        cache,
        chunk_hashes=[],
        embedding_model="m",
        embedding_dims=3,
    )
    assert result.hit_count == 0
    assert result.miss_count == 0


def test_cache_skips_vectors_with_wrong_dims(tmp_path: Path) -> None:
    """If a caller passes a vector of the wrong length, it is skipped silently
    rather than corrupting the table. The fixed-size-list schema would
    reject it anyway; we'd rather drop the bad entry than fail the whole
    batch."""
    db = connect_lancedb(tmp_path / "lancedb")
    cache = open_cache_table(db, vector_size=3)
    written = store(
        cache,
        chunk_hashes=["good", "bad"],
        vectors=[[1.0, 0.0, 0.0], [1.0, 0.0]],  # second is wrong size
        embedding_model="m",
        embedding_dims=3,
    )
    assert written == 1
    result = lookup(
        cache,
        chunk_hashes=["good", "bad"],
        embedding_model="m",
        embedding_dims=3,
    )
    assert result.hit_count == 1
    assert result.misses_indices == [1]
