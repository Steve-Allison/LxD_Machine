"""Persist and query vector chunk records in LanceDB."""

import json
from pathlib import Path
from typing import Any, Final

import lancedb
import pyarrow as pa
import structlog

from lxd.stores.lance_sql import eq_clause, in_clause
from lxd.stores.models import ChunkRecord, VectorSearchRecord

_TABLE_NAME: Final = "chunk_vectors"
_FTS_FIELD: Final = "text"
_log = structlog.get_logger(__name__)


def connect_lancedb(path: Path) -> Any:
    """Open (and create if needed) the LanceDB database directory.

    Args:
        path: Path to the source file or storage location.

    Returns:
        Connected LanceDB database handle.
    """
    path.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(str(path))


def open_chunk_table(database: Any, *, vector_size: int) -> Any:
    """Open the chunk vector table, creating it when missing.

    Also ensures the native LanceDB FTS index over the ``text`` column is
    present so retrieval can issue BM25 queries without per-call setup.
    The index is replaced (rebuilt) on each open: native LanceDB FTS does
    not auto-include rows added after index creation, so retrieval needs a
    fresh index for fairness against the latest writes.

    Args:
        database: Open LanceDB database handle.
        vector_size: Embedding vector length for schema creation.

    Returns:
        Opened or newly created chunk table.
    """
    try:
        table = database.open_table(_TABLE_NAME)
    except (FileNotFoundError, ValueError) as exc:
        if isinstance(exc, ValueError) and not _is_missing_table_error(exc):
            raise
        table = database.create_table(
            _TABLE_NAME,
            schema=_chunk_table_schema(vector_size),
            mode="create",
        )
    refresh_fts_index(table)
    return table


def reset_chunk_table(database: Any, *, vector_size: int) -> Any:
    """Drop and recreate the chunk vector table schema.

    Args:
        database: Open LanceDB database handle.
        vector_size: Embedding vector length for schema creation.

    Returns:
        Newly created empty chunk table.
    """
    try:
        database.drop_table(_TABLE_NAME)
    except FileNotFoundError:
        pass
    except ValueError as exc:
        if not _is_missing_table_error(exc):
            raise
    table = database.create_table(
        _TABLE_NAME,
        schema=_chunk_table_schema(vector_size),
        mode="create",
    )
    refresh_fts_index(table)
    return table


def refresh_fts_index(table: Any) -> None:
    """(Re)build the native LanceDB FTS index over the ``text`` column.

    Native LanceDB FTS is incrementally appendable but does not
    auto-include rows added after index creation; ingest calls this once
    after persisting all chunks so retrieval BM25 sees every row. Calls
    are idempotent: the index is replaced in place when it already exists
    and created from scratch otherwise.

    Args:
        table: LanceDB chunk_vectors table.
    """
    table.create_fts_index(_FTS_FIELD, use_tantivy=False, replace=True)


def replace_source_chunks(
    table: Any, source_rel_path: str, chunk_records: list[ChunkRecord]
) -> None:
    """Replace all vector chunks for one source path.

    Args:
        table: LanceDB table storing chunk vectors.
        source_rel_path: Corpus-relative source path.
        chunk_records: Chunk rows to persist for a source.
    """
    delete_source(table, source_rel_path)
    if chunk_records:
        table.add([_chunk_record_to_row(record) for record in chunk_records])


def delete_source(table: Any, source_rel_path: str) -> None:
    """Apply the requested persistence operation.

    Args:
        table: LanceDB table storing chunk vectors.
        source_rel_path: Corpus-relative source path.
    """
    table.delete(eq_clause("source_rel_path", source_rel_path))


def search_chunks(
    table: Any,
    *,
    query_vector: list[float],
    domain: str | None,
    limit: int,
) -> list[VectorSearchRecord]:
    """Dense vector retrieval ordered by cosine distance.

    Args:
        table: LanceDB table storing chunk vectors.
        query_vector: Embedded query vector for nearest-neighbour search.
        domain: Optional source domain filter.
        limit: Maximum number of records to return.

    Returns:
        Vector search matches ordered by similarity. ``score`` carries the
        raw cosine distance (lower is closer); callers negate it when a
        higher-is-better ordering is needed.
    """
    query = table.search(query_vector, vector_column_name="vector").metric("cosine")
    if domain is not None:
        query = query.where(eq_clause("source_domain", domain))
    rows = query.limit(limit).to_list()
    return [
        record
        for record in (_row_to_vector_search_record(row, score_field="_distance") for row in rows)
        if record is not None
    ]


def search_chunks_fts(
    table: Any,
    *,
    query: str,
    domain: str | None,
    limit: int,
) -> list[VectorSearchRecord]:
    """BM25 full-text retrieval over the chunk ``text`` column.

    The native LanceDB FTS index is built by :func:`open_chunk_table` /
    :func:`refresh_fts_index`; queries here issue BM25 directly against
    that index — no Python-side keyword counting, no IDF estimation by
    hand, no length-normalisation guesswork.

    Args:
        table: LanceDB table storing chunk vectors.
        query: Natural-language query string.
        domain: Optional source domain filter.
        limit: Maximum number of records to return.

    Returns:
        Chunks ordered by BM25 relevance (higher score = better match).
        Returns an empty list when the query is empty or contains no
        index-matching tokens.
    """
    cleaned = query.strip()
    if not cleaned:
        return []
    fts_query = table.search(cleaned, query_type="fts")
    if domain is not None:
        fts_query = fts_query.where(eq_clause("source_domain", domain))
    rows = fts_query.limit(limit).to_list()
    return [
        record
        for record in (_row_to_vector_search_record(row, score_field="_score") for row in rows)
        if record is not None
    ]


def _row_to_vector_search_record(
    row: dict[str, Any], *, score_field: str
) -> VectorSearchRecord | None:
    score_value = row.get(score_field)
    if not isinstance(score_value, (int, float)):
        return None
    return VectorSearchRecord(
        chunk_id=str(row["chunk_id"]),
        document_id=str(row["document_id"]),
        source_rel_path=str(row["source_rel_path"]),
        source_filename=str(row["source_filename"]),
        source_type=str(row["source_type"]),
        source_domain=str(row["source_domain"]),
        source_hash=str(row["source_hash"]),
        citation_label=str(row["citation_label"]),
        chunk_index=int(row["chunk_index"]),
        chunk_occurrence=int(row["chunk_occurrence"]),
        token_count=int(row["token_count"]),
        text=str(row["text"]),
        score_hint=str(row["score_hint"]),
        metadata_json=str(row["metadata_json"]),
        score=float(score_value),
        cited_sources=_decode_string_list(row.get("cited_sources_json")),
        wiki_links=_decode_string_list(row.get("wiki_links_json")),
    )


def load_vectors_by_chunk_ids(table: Any, chunk_ids: list[str]) -> dict[str, list[float]]:
    """Return the stored embedding vector for each requested chunk_id.

    Args:
        table: LanceDB table storing chunk vectors.
        chunk_ids: Chunk identifiers to look up.

    Returns:
        Mapping of ``chunk_id`` -> ``vector``. Missing chunks are simply
        absent from the returned dict; callers must handle that case.

    Side Effects:
        None. Performs a single filtered LanceDB scan.
    """
    if not chunk_ids:
        return {}
    rows = (
        table.search()
        .where(in_clause("chunk_id", chunk_ids))
        .select(["chunk_id", "vector"])
        .to_list()
    )
    result: dict[str, list[float]] = {}
    for row in rows:
        vector = row.get("vector")
        if vector is None:
            continue
        result[str(row["chunk_id"])] = [float(v) for v in vector]
    return result


def _decode_string_list(value: object) -> tuple[str, ...]:
    """Decode a JSON-array-of-strings column tolerantly.

    LanceDB rows pre-dating the wiki swap may not carry the column at all
    (returns ``None``), or carry an empty string. Both cases are fine.
    """
    if not value or not isinstance(value, str):
        return ()
    try:
        parsed = json.loads(value)
    except TypeError, ValueError:
        return ()
    if not isinstance(parsed, list):
        return ()
    return tuple(str(item) for item in parsed if isinstance(item, str))


def _chunk_table_schema(vector_size: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("chunk_id", pa.string()),
            pa.field("document_id", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), vector_size)),
            pa.field("source_rel_path", pa.string()),
            pa.field("source_filename", pa.string()),
            pa.field("source_type", pa.string()),
            pa.field("source_domain", pa.string()),
            pa.field("source_hash", pa.string()),
            pa.field("citation_label", pa.string()),
            pa.field("chunk_index", pa.int32()),
            pa.field("chunk_occurrence", pa.int32()),
            pa.field("token_count", pa.int32()),
            pa.field("text", pa.string()),
            pa.field("score_hint", pa.string()),
            pa.field("metadata_json", pa.string()),
            pa.field("cited_sources_json", pa.string()),
            pa.field("wiki_links_json", pa.string()),
        ]
    )


def _chunk_record_to_row(record: ChunkRecord) -> dict[str, object]:
    return {
        "chunk_id": record.chunk_id,
        "document_id": record.document_id,
        "vector": [float(value) for value in record.vector],
        "source_rel_path": record.source_rel_path,
        "source_filename": record.source_filename,
        "source_type": record.source_type,
        "source_domain": record.source_domain,
        "source_hash": record.source_hash,
        "citation_label": record.citation_label,
        "chunk_index": record.chunk_index,
        "chunk_occurrence": record.chunk_occurrence,
        "token_count": record.token_count,
        "text": record.text,
        "score_hint": record.score_hint,
        "metadata_json": record.metadata_json,
        "cited_sources_json": json.dumps(list(record.cited_sources)),
        "wiki_links_json": json.dumps(list(record.wiki_links)),
    }


# ---------------------------------------------------------------------------
# Entity embeddings table
# ---------------------------------------------------------------------------

_ENTITY_TABLE_NAME: Final = "entity_embeddings"


def open_entity_table(database: Any, *, vector_size: int) -> Any:
    """Open the entity embeddings table, creating it when missing."""
    try:
        return database.open_table(_ENTITY_TABLE_NAME)
    except (FileNotFoundError, ValueError) as exc:
        if isinstance(exc, ValueError) and not _is_missing_table_error(exc):
            raise
        return database.create_table(
            _ENTITY_TABLE_NAME,
            schema=_entity_table_schema(vector_size),
            mode="create",
        )


def replace_entity_embeddings(
    table: Any,
    records: list[dict[str, object]],
) -> None:
    """Replace all entity embeddings (full rebuild).

    Each record must have: entity_id, label, community_id, vector.

    Reserved for callers that genuinely want "wipe and reload" semantics —
    test fixtures and the ``build-graph --full`` path. Routine incremental
    runs use :func:`upsert_entity_embeddings` instead so unchanged entities
    keep their existing vectors.

    The delete-before-add is the canonical "replace-all" idiom against
    LanceDB's append-only storage. When the table has no prior rows, the
    delete is a no-op but some LanceDB builds raise ``FileNotFoundError``
    (empty-fragment lookup) or ``ValueError`` (no predicate match); both are
    swallowed with a debug log so the caller sees a clean "replace" semantic.
    """
    try:
        table.delete("entity_id IS NOT NULL")
    except (FileNotFoundError, ValueError) as exc:
        _log.debug("lancedb_entity_delete_skipped", error=str(exc))
    if records:
        table.add(records)


def upsert_entity_embeddings(
    table: Any,
    records: list[dict[str, object]],
    *,
    removed_entity_ids: list[str] | None = None,
) -> None:
    """Replace only the named entity rows; leave the rest of the table alone.

    Used by the incremental ``build-graph`` path so unchanged entities keep
    their existing mean-pooled vectors and the LanceDB table is not wiped on
    every run. ``records`` are the entities whose vector is being added or
    updated; ``removed_entity_ids`` are entities that no longer qualify and
    should be evicted entirely.

    Delete-before-add is the LanceDB upsert idiom — the storage layer is
    append-only, so an in-place row update is two operations.
    """
    to_delete = list(removed_entity_ids or [])
    to_delete.extend(str(record["entity_id"]) for record in records)
    if to_delete:
        try:
            table.delete(in_clause("entity_id", to_delete))
        except (FileNotFoundError, ValueError) as exc:
            _log.debug("lancedb_entity_delete_skipped", error=str(exc))
    if records:
        table.add(records)


def search_similar_entities(
    table: Any,
    *,
    query_vector: list[float],
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Find entities nearest to a query vector."""
    rows = (
        table.search(query_vector, vector_column_name="vector")
        .metric("cosine")
        .limit(limit)
        .to_list()
    )
    results: list[dict[str, Any]] = []
    for row in rows:
        score_value = row.get("_distance")
        if not isinstance(score_value, (int, float)):
            continue
        results.append(
            {
                "entity_id": str(row["entity_id"]),
                "label": str(row["label"]),
                "community_id": int(row["community_id"])
                if row.get("community_id") is not None
                else None,
                "score": float(score_value),
            }
        )
    return results


def fetch_vectors_by_chunk_ids(
    table: Any,
    chunk_ids: list[str],
) -> dict[str, list[float]]:
    """Fetch raw vectors for specific chunk IDs from LanceDB.

    Returns a mapping of chunk_id to vector. More efficient than parsing
    JSON text from SQLite for large vector dimensions.
    """
    if not chunk_ids:
        return {}
    rows = (
        table.search()
        .where(in_clause("chunk_id", chunk_ids))
        .select(["chunk_id", "vector"])
        .limit(len(chunk_ids))
        .to_list()
    )
    result: dict[str, list[float]] = {}
    for row in rows:
        vec = row.get("vector")
        if vec is not None:
            result[str(row["chunk_id"])] = [float(v) for v in vec]
    return result


def _entity_table_schema(vector_size: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("entity_id", pa.string()),
            pa.field("label", pa.string()),
            pa.field("community_id", pa.int32()),
            pa.field("vector", pa.list_(pa.float32(), vector_size)),
        ]
    )


def _is_missing_table_error(error: ValueError) -> bool:
    return "was not found" in str(error)
