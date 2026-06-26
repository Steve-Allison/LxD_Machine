"""Content-addressed embedding cache.

Responsibility:
    Avoid paying OpenAI / Ollama for embeddings we have already computed.
    Cache entries are keyed by ``(chunk_hash, embedding_model, embedding_dims)``
    — a content-addressed key that is *intrinsically* safe to keep across
    full rebuilds: if the input bytes are the same and the model+dims are the
    same, the embedding vector is deterministic. There is no cache
    invalidation other than "the model changed", which produces a new key
    and naturally bypasses old entries.

Design boundary:
    The cache is read by :func:`embed_with_cache` before calling the embedder
    and written immediately after a successful embed. It must never block
    ingest if it fails — a cache miss is the correct fallback.

Storage:
    A LanceDB table (``embedding_cache``) under the same data directory as
    chunk vectors. LanceDB is already a dependency, supports vector columns,
    and is cheap to query by string keys. Keeping the cache adjacent to
    ``chunk_vectors`` means the cache moves with the data dir on copy.

Why not SQLite?
    Embedding vectors are float32[1536] (or larger). Storing them in SQLite
    means JSON-encoding and base64-decoding on every read; keeping them in
    LanceDB lets us return the raw float lists directly.
"""

from dataclasses import dataclass
from typing import Any, Final

import pyarrow as pa
import structlog

from lxd.stores.lance_sql import in_clause

_TABLE_NAME: Final = "embedding_cache"
_log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class CacheLookupResult:
    """Result of looking up texts in the cache.

    Attributes:
        hits: Index → vector for texts that were already cached.
        misses_indices: Indices into the original ``texts`` list that need
            embedding via the live API.
    """

    hits: dict[int, list[float]]
    misses_indices: list[int]

    @property
    def hit_count(self) -> int:
        return len(self.hits)

    @property
    def miss_count(self) -> int:
        return len(self.misses_indices)


def open_cache_table(database: Any, *, vector_size: int) -> Any:
    """Open the embedding cache table, creating it when missing.

    Schema is intentionally tight:
        - cache_key (string, primary lookup): "{chunk_hash}|{model}|{dims}"
        - chunk_hash (string): redundant with cache_key but keeps queries readable
        - embedding_model (string)
        - embedding_dims (int32)
        - vector (fixed_size_list[float32, vector_size])

    Args:
        database: Open LanceDB database handle.
        vector_size: Embedding vector length for schema creation.

    Returns:
        Opened or newly created cache table.
    """
    try:
        return database.open_table(_TABLE_NAME)
    except (FileNotFoundError, ValueError) as exc:
        if isinstance(exc, ValueError) and "was not found" not in str(exc):
            raise
        return database.create_table(
            _TABLE_NAME,
            schema=_cache_schema(vector_size),
            mode="create",
        )


def lookup(
    cache_table: Any,
    *,
    chunk_hashes: list[str],
    embedding_model: str,
    embedding_dims: int,
) -> CacheLookupResult:
    """Look up cached embeddings for a list of chunk hashes.

    Args:
        cache_table: Cache table handle.
        chunk_hashes: Chunk hashes (BLAKE3) in input order. Duplicates allowed
            and resolved correctly — only one query is issued.
        embedding_model: Model identifier the caller will use for any misses.
        embedding_dims: Embedding dimensionality.

    Returns:
        :class:`CacheLookupResult` with hit vectors keyed by input index.

    Side Effects:
        None.
    """
    if not chunk_hashes:
        return CacheLookupResult(hits={}, misses_indices=[])

    cache_keys = [_cache_key(h, embedding_model, embedding_dims) for h in chunk_hashes]
    unique_keys = sorted(set(cache_keys))

    try:
        rows = (
            cache_table.search()
            .where(in_clause("cache_key", unique_keys))
            .select(["cache_key", "vector"])
            .to_list()
        )
    except (FileNotFoundError, ValueError) as exc:
        _log.warning("embedding_cache_lookup_skipped", error=str(exc))
        return CacheLookupResult(
            hits={},
            misses_indices=list(range(len(chunk_hashes))),
        )

    # B-PERF-4 (2026-05-06): LanceDB's `to_list()` already returns native
    # Python `list[float]` for the fixed_size_list<float32> column, so the
    # previous `[float(v) for v in vector]` was paying for a per-element
    # coercion that did not change types. Replacing it with `list(vector)`
    # delivers a defensive copy (so the caller cannot mutate the LanceDB
    # row's buffer) at ~3x the throughput on 1k-row batches. The audit's
    # original "Arrow vectorise via to_arrow().to_pylist()" approach was
    # benchmarked and found to be 4-5x *slower* than this loop because
    # `pa.FixedSizeListArray.to_pylist()` allocates the same Python lists
    # via a different path.
    by_key: dict[str, list[float]] = {}
    for row in rows:
        vector = row.get("vector")
        if vector is None:
            continue
        by_key[str(row["cache_key"])] = list(vector)

    hits: dict[int, list[float]] = {}
    misses: list[int] = []
    for idx, key in enumerate(cache_keys):
        cached = by_key.get(key)
        if cached is not None:
            hits[idx] = cached
        else:
            misses.append(idx)
    return CacheLookupResult(hits=hits, misses_indices=misses)


def store(
    cache_table: Any,
    *,
    chunk_hashes: list[str],
    vectors: list[list[float]],
    embedding_model: str,
    embedding_dims: int,
) -> int:
    """Persist embeddings to the cache.

    Idempotent: existing entries with the same cache_key are deleted first,
    so re-storing produces no duplicates. The cache_key is content-addressed
    (chunk_hash + model + dims) so an "update" is only conceptually possible
    if the embedding model changed its outputs for the same input — in which
    case the model identifier should change too. Re-storing is otherwise a
    no-op equivalent.

    Args:
        cache_table: Cache table handle.
        chunk_hashes: Chunk hashes corresponding to ``vectors``, in order.
        vectors: Newly computed embeddings, in same order as ``chunk_hashes``.
        embedding_model: Model identifier for this batch.
        embedding_dims: Embedding dimensionality.

    Returns:
        Number of cache rows written.
    """
    if not chunk_hashes:
        return 0
    if len(chunk_hashes) != len(vectors):
        raise ValueError(
            f"chunk_hashes ({len(chunk_hashes)}) and vectors ({len(vectors)}) length mismatch"
        )

    deduped: dict[str, list[float]] = {}
    for chunk_hash, vector in zip(chunk_hashes, vectors, strict=True):
        cache_key = _cache_key(chunk_hash, embedding_model, embedding_dims)
        if len(vector) != embedding_dims:
            continue
        deduped[cache_key] = [float(v) for v in vector]

    if not deduped:
        return 0

    try:
        cache_table.delete(in_clause("cache_key", sorted(deduped)))
    except (FileNotFoundError, ValueError) as exc:
        _log.debug("embedding_cache_pre_delete_skipped", error=str(exc))

    rows = [
        {
            "cache_key": key,
            "chunk_hash": key.split("|", 1)[0],
            "embedding_model": embedding_model,
            "embedding_dims": embedding_dims,
            "vector": vector,
        }
        for key, vector in deduped.items()
    ]
    try:
        cache_table.add(rows)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        _log.warning("embedding_cache_store_failed", error=str(exc))
        return 0
    return len(rows)


def _cache_key(chunk_hash: str, embedding_model: str, embedding_dims: int) -> str:
    return f"{chunk_hash}|{embedding_model}|{int(embedding_dims)}"


def _cache_schema(vector_size: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("cache_key", pa.string()),
            pa.field("chunk_hash", pa.string()),
            pa.field("embedding_model", pa.string()),
            pa.field("embedding_dims", pa.int32()),
            pa.field("vector", pa.list_(pa.float32(), vector_size)),
        ]
    )
