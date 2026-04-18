"""Persist and query vector chunk records in LanceDB."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa
import structlog

from lxd.stores.lance_sql import eq_clause, in_clause
from lxd.stores.models import ChunkRecord, VectorSearchRecord

_TABLE_NAME = "chunk_vectors"
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

    Args:
        database: Open LanceDB database handle.
        vector_size: Embedding vector length for schema creation.

    Returns:
        Opened or newly created chunk table.
    """
    try:
        return database.open_table(_TABLE_NAME)
    except (FileNotFoundError, ValueError) as exc:
        if isinstance(exc, ValueError) and not _is_missing_table_error(exc):
            raise
        return database.create_table(
            _TABLE_NAME,
            schema=_chunk_table_schema(vector_size),
            mode="create",
        )


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
    return database.create_table(
        _TABLE_NAME,
        schema=_chunk_table_schema(vector_size),
        mode="create",
    )


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
    """Run dense retrieval, optional rerank, and fusion.

    Args:
        table: LanceDB table storing chunk vectors.
        query_vector: Embedded query vector for nearest-neighbor search.
        domain: Optional source domain filter.
        limit: Maximum number of records to return.

    Returns:
        Vector search matches ordered by similarity.
    """
    query = table.search(query_vector, vector_column_name="vector").metric("cosine")
    if domain is not None:
        query = query.where(eq_clause("source_domain", domain))
    rows = query.limit(limit).to_list()
    records: list[VectorSearchRecord] = []
    for row in rows:
        score_value = row.get("_distance")
        if not isinstance(score_value, (int, float)):
            continue
        records.append(
            VectorSearchRecord(
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
            )
        )
    return records


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
    }


# ---------------------------------------------------------------------------
# Entity embeddings table (Phase 5)
# ---------------------------------------------------------------------------

_ENTITY_TABLE_NAME = "entity_embeddings"


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


def reset_entity_table(database: Any, *, vector_size: int) -> Any:
    """Drop and recreate the entity embeddings table."""
    try:
        database.drop_table(_ENTITY_TABLE_NAME)
    except FileNotFoundError:
        pass
    except ValueError as exc:
        if not _is_missing_table_error(exc):
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
