"""Persist and query vector chunk records in LanceDB."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import pyarrow as pa

from lxd.stores.models import ChunkRecord, VectorSearchRecord

_TABLE_NAME = "chunk_vectors"


def connect_lancedb(path: Path) -> Any:
    """Open (and create if needed) the LanceDB database directory.

    Args:
        path: Path to the source file or storage location.

    Returns:
        Connected LanceDB database handle.
    """
    import lancedb

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
    except FileNotFoundError:
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
    table.delete(f"source_rel_path = '{_escape_string_literal(source_rel_path)}'")


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
        query = query.where(f"source_domain = '{_escape_string_literal(domain)}'")
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
                source_path=str(row["source_path"]),
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


def _chunk_table_schema(vector_size: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("chunk_id", pa.string()),
            pa.field("document_id", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), vector_size)),
            pa.field("source_path", pa.string()),
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
        "source_path": record.source_path,
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
    except FileNotFoundError:
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
    """
    with contextlib.suppress(Exception):
        table.delete("entity_id IS NOT NULL")
    if records:
        table.add(records)


def search_similar_entities(
    table: Any,
    *,
    query_vector: list[float],
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Find entities nearest to a query vector."""
    rows = table.search(query_vector, vector_column_name="vector").limit(limit).to_list()
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


def _entity_table_schema(vector_size: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("entity_id", pa.string()),
            pa.field("label", pa.string()),
            pa.field("community_id", pa.int32()),
            pa.field("vector", pa.list_(pa.float32(), vector_size)),
        ]
    )


def _escape_string_literal(value: str) -> str:
    return value.replace("'", "''")


def _is_missing_table_error(error: ValueError) -> bool:
    return "was not found" in str(error)
