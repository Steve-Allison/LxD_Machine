"""Persist and query vector chunk records in LanceDB."""

import json
from pathlib import Path
from typing import Any, Final

import lancedb
import pyarrow as pa
from lancedb.index import FTS, BTree
from lancedb.rerankers import RRFReranker

from lxd.stores.lance_sql import eq_clause, in_clause
from lxd.stores.models import ChunkRecord, VectorSearchRecord

_TABLE_NAME: Final = "chunk_vectors"
_FTS_FIELD: Final = "text"
_CHUNK_SCALAR_INDEX_COLUMNS: Final = ("source_rel_path", "chunk_id", "source_domain")


def ensure_scalar_index(table: Any, column: str) -> None:
    """Create a BTREE scalar index on ``column`` if one does not already exist.

    Idempotent: safe to call on every table open. Uses the native LanceDB
    ``create_index(config=BTree())`` API introduced in 0.25; the pre-0.25
    ``create_scalar_index`` shim is deprecated.

    Args:
        table: LanceDB table handle.
        column: Column name to index.
    """
    index_name = f"{column}_idx"
    existing_names = {getattr(index, "name", None) for index in table.list_indices()}
    if index_name in existing_names:
        return
    table.create_index(column, config=BTree(), name=index_name)


def connect_lancedb(path: Path) -> Any:
    """Open (and create if needed) the LanceDB database directory.

    Args:
        path: Path to the source file or storage location.

    Returns:
        Connected LanceDB database handle.
    """
    path.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(str(path))


def open_chunk_table(
    database: Any, *, vector_size: int, refresh_fts: bool = False
) -> Any:
    """Open the chunk vector table, creating it when missing.

    On the **read** path (default), ensure the native LanceDB FTS index
    over ``text`` exists but do not rebuild it — ingest refreshes FTS once
    after writes. On the **write** path, pass ``refresh_fts=True`` (or call
    :func:`refresh_fts_index` after mutations) so BM25 sees newly added rows.
    Native LanceDB FTS does not auto-include rows added after index creation.

    Args:
        database: Open LanceDB database handle.
        vector_size: Embedding vector length for schema creation.
        refresh_fts: When True, rebuild the FTS index (write/ingest path).

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
        # Fresh table has no FTS yet — create it once.
        refresh_fts_index(table)
    else:
        if refresh_fts:
            refresh_fts_index(table)
        else:
            ensure_fts_index(table)
    for column in _CHUNK_SCALAR_INDEX_COLUMNS:
        ensure_scalar_index(table, column)
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
    for column in _CHUNK_SCALAR_INDEX_COLUMNS:
        ensure_scalar_index(table, column)
    return table


def ensure_fts_index(table: Any) -> None:
    """Create the native FTS index over ``text`` if it is absent.

    Idempotent for the read path: an existing index is left untouched so
    retrieval does not pay a full Tantivy rebuild on every query. Call
    :func:`refresh_fts_index` after ingest writes so BM25 sees new rows.
    """
    fts_index_name = f"{_FTS_FIELD}_fts_idx"
    existing_names = {getattr(index, "name", None) for index in table.list_indices()}
    if fts_index_name in existing_names:
        return
    table.create_index(
        _FTS_FIELD,
        config=FTS(with_position=False),
        name=fts_index_name,
        replace=False,
    )


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
    fts_index_name = f"{_FTS_FIELD}_fts_idx"
    table.create_index(
        _FTS_FIELD,
        config=FTS(with_position=False),
        name=fts_index_name,
        replace=True,
    )


def load_source_chunk_rows(table: Any, source_rel_path: str) -> list[dict[str, object]]:
    """Snapshot all LanceDB rows for one source path (pre-write compensate).

    Returns raw table rows suitable for :func:`restore_source_chunk_rows`.
    An empty list means the source had no vectors (first ingest / cleared).
    """
    rows = (
        table.search()
        .where(eq_clause("source_rel_path", source_rel_path))
        .select(
            [
                "chunk_id",
                "document_id",
                "vector",
                "source_rel_path",
                "source_filename",
                "source_type",
                "source_domain",
                "source_hash",
                "citation_label",
                "chunk_index",
                "chunk_occurrence",
                "token_count",
                "text",
                "score_hint",
                "metadata_json",
                "cited_sources_json",
                "wiki_links_json",
            ]
        )
        .to_list()
    )
    snapshot: list[dict[str, object]] = []
    for row in rows:
        vector = row.get("vector")
        if vector is None:
            continue
        snapshot.append(
            {
                "chunk_id": str(row["chunk_id"]),
                "document_id": str(row["document_id"]),
                "vector": [float(v) for v in vector],
                "source_rel_path": str(row["source_rel_path"]),
                "source_filename": str(row["source_filename"]),
                "source_type": str(row["source_type"]),
                "source_domain": str(row["source_domain"]),
                "source_hash": str(row["source_hash"]),
                "citation_label": str(row["citation_label"]),
                "chunk_index": int(row["chunk_index"]),
                "chunk_occurrence": int(row["chunk_occurrence"]),
                "token_count": int(row["token_count"]),
                "text": str(row["text"]),
                "score_hint": str(row["score_hint"]),
                "metadata_json": str(row["metadata_json"]),
                "cited_sources_json": str(row.get("cited_sources_json") or "[]"),
                "wiki_links_json": str(row.get("wiki_links_json") or "[]"),
            }
        )
    return snapshot


def restore_source_chunk_rows(
    table: Any, source_rel_path: str, snapshot: list[dict[str, object]]
) -> None:
    """Restore a source path to a previously captured LanceDB snapshot.

    Deletes any current rows for ``source_rel_path``, then re-adds the
    snapshot. An empty snapshot leaves the path empty — the correct
    compensate outcome for a first-ingest failure.
    """
    delete_source(table, source_rel_path)
    if snapshot:
        table.add(snapshot)


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


def search_chunks_hybrid(
    table: Any,
    *,
    query: str,
    query_vector: list[float],
    domain: str | None,
    limit: int,
) -> list[VectorSearchRecord]:
    """Hybrid dense + BM25 retrieval fused inside LanceDB via RRF.

    ``Table.search(query_type="hybrid")`` runs the dense k-NN and the BM25
    FTS index in one query and fuses them with the passed reranker
    (Reciprocal Rank Fusion here). Returns a single ordered list keyed on
    ``_relevance_score`` — the per-lane ranks are collapsed inside the
    engine and are not surfaced to Python callers.

    This is an alternative to the two-query + Python-side fuse path that
    ``search_chunks`` + ``search_chunks_fts`` provide separately. Callers
    that need independent per-lane weights (e.g. the current 5-lane RRF
    in :mod:`lxd.retrieval.query_pipeline`) cannot use this shape; those
    that just want dense+BM25 fused with default RRF can.
    """
    cleaned = query.strip()
    if not cleaned:
        # Fall back to dense-only when the query is empty — hybrid with an
        # empty text query is undefined at the engine level.
        return search_chunks(
            table, query_vector=query_vector, domain=domain, limit=limit
        )
    hybrid = (
        table.search(query_type="hybrid")
        .vector(query_vector)
        .text(cleaned)
    )
    if domain is not None:
        hybrid = hybrid.where(eq_clause("source_domain", domain))
    rows = hybrid.rerank(RRFReranker()).limit(limit).to_list()
    return [
        record
        for record in (
            _row_to_vector_search_record(row, score_field="_relevance_score") for row in rows
        )
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
        .limit(len(chunk_ids))
        .to_list()
    )
    result: dict[str, list[float]] = {}
    for row in rows:
        vector = row.get("vector")
        if vector is None:
            continue
        result[str(row["chunk_id"])] = [float(v) for v in vector]
    return result


# Back-compat alias — prefer :func:`load_vectors_by_chunk_ids`.
fetch_vectors_by_chunk_ids = load_vectors_by_chunk_ids


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
        table = database.open_table(_ENTITY_TABLE_NAME)
    except (FileNotFoundError, ValueError) as exc:
        if isinstance(exc, ValueError) and not _is_missing_table_error(exc):
            raise
        table = database.create_table(
            _ENTITY_TABLE_NAME,
            schema=_entity_table_schema(vector_size),
            mode="create",
        )
    ensure_scalar_index(table, "entity_id")
    return table


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

    Uses the native LanceDB ``merge_insert`` upsert with
    ``when_not_matched_by_source_delete`` so the replace is a single atomic
    operation: rows present in ``records`` are updated or inserted; rows
    absent from ``records`` are evicted; the empty-table case is a
    no-op-plus-insert with no exception path to smooth over.
    """
    if not records:
        # Full wipe when the caller says "no entities at all" — a legitimate
        # corner (empty ontology, initial bootstrap). Fall through to a
        # merge_insert of an empty batch by way of an empty schema-shaped
        # frame so the delete side still runs.
        (
            table.merge_insert("entity_id")
            .when_not_matched_by_source_delete()
            .execute(_empty_entity_frame(table))
        )
        return
    (
        table.merge_insert("entity_id")
        .when_matched_update_all()
        .when_not_matched_insert_all()
        .when_not_matched_by_source_delete()
        .execute(records)
    )


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

    The record set uses the native LanceDB ``merge_insert`` upsert so each
    row is atomically updated-or-inserted in one call — no delete-before-add
    round-trip and no defensive try/except to smooth over an empty-fragment
    quirk. Removed entities are evicted via an explicit ``delete`` because
    they belong to a disjoint keyset the merge cannot see.
    """
    if removed_entity_ids:
        table.delete(in_clause("entity_id", removed_entity_ids))
    if not records:
        return
    (
        table.merge_insert("entity_id")
        .when_matched_update_all()
        .when_not_matched_insert_all()
        .execute(records)
    )


def _empty_entity_frame(table: Any) -> pa.Table:
    """Return an empty Arrow table matching the entity schema."""
    return pa.Table.from_pylist([], schema=table.schema)


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
