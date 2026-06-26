"""Row-to-record adapters for the ``lxd.stores.sqlite`` query layer.

Responsibility:
    Translate raw :class:`sqlite3.Row` instances into the typed records
    declared in :mod:`lxd.stores.models`. Keeping these adapters separate
    from the query functions keeps column-name churn localised and makes
    the query file easier to scan.

Design boundary:
    Private to ``lxd.stores``. External callers must use the typed APIs in
    :mod:`lxd.stores.sqlite` instead of importing row helpers directly.

Key constraints:
    All helpers are pure functions of a single row; they must not execute
    SQL, mutate state, or hold references to the connection.
"""

import json
import sqlite3
from typing import Any

from lxd.domain.ids import blake3_hex
from lxd.stores.models import (
    CanonicalRelationRecord,
    ChunkRecord,
    ClaimRecord,
    CommunityReportRecord,
    EntityProfileRecord,
    ManifestRecord,
    MentionRecord,
)


def optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def row_value(row: sqlite3.Row | None, key: str) -> int:
    if row is None:
        return 0
    value: Any = row[key]
    if value is None:
        return 0
    return int(value)


def mention_id(record: MentionRecord) -> str:
    return blake3_hex(record.entity_id, record.chunk_id, str(record.start_char))


def claim_from_row(row: sqlite3.Row) -> ClaimRecord:
    return ClaimRecord(
        claim_id=str(row["claim_id"]),
        chunk_id=str(row["chunk_id"]),
        document_id=str(row["document_id"]),
        source_rel_path=str(row["source_rel_path"]),
        claim_text=str(row["claim_text"]),
        subject_entity_id=optional_str(row["subject_entity_id"]),
        object_entity_id=optional_str(row["object_entity_id"]),
        claim_type=str(row["claim_type"]),
        confidence=float(row["confidence"]),
        extraction_model=str(row["extraction_model"]),
        extracted_at=str(row["extracted_at"]),
    )


def entity_profile_from_row(row: sqlite3.Row) -> EntityProfileRecord:
    return EntityProfileRecord(
        entity_id=str(row["entity_id"]),
        label=str(row["label"]),
        entity_type=str(row["entity_type"]),
        domain=str(row["domain"]),
        aliases_json=str(row["aliases_json"]),
        deterministic_summary=str(row["deterministic_summary"]),
        llm_summary=optional_str(row["llm_summary"]),
        chunk_count=int(row["chunk_count"]),
        doc_count=int(row["doc_count"]),
        mention_count=int(row["mention_count"]),
        claim_count=int(row["claim_count"]),
        top_predicates_json=str(row["top_predicates_json"]),
        top_claims_json=str(row["top_claims_json"]),
        pagerank=float(row["pagerank"]),
        betweenness=float(row["betweenness"]),
        closeness=float(row["closeness"]),
        in_degree=int(row["in_degree"]),
        out_degree=int(row["out_degree"]),
        eigenvector=float(row["eigenvector"]),
        community_id=int(row["community_id"]) if row["community_id"] is not None else None,
        source_hash=str(row["source_hash"]),
        generated_at=str(row["generated_at"]),
    )


def community_report_from_row(row: sqlite3.Row) -> CommunityReportRecord:
    row_keys = set(row.keys())
    parent_raw = row["parent_community_id"] if "parent_community_id" in row_keys else None
    return CommunityReportRecord(
        community_id=int(row["community_id"]),
        community_level=int(row["community_level"]),
        parent_community_id=int(parent_raw) if parent_raw is not None else None,
        member_count=int(row["member_count"]),
        member_entity_ids_json=str(row["member_entity_ids_json"]),
        deterministic_summary=str(row["deterministic_summary"]),
        llm_summary=optional_str(row["llm_summary"]),
        top_entities_json=str(row["top_entities_json"]),
        top_claims_json=str(row["top_claims_json"]),
        intra_community_edge_count=int(row["intra_community_edge_count"]),
        source_hash=str(row["source_hash"]),
        generated_at=str(row["generated_at"]),
    )


def canonical_relation_from_row(row: sqlite3.Row) -> CanonicalRelationRecord:
    return CanonicalRelationRecord(
        relation_id=str(row["relation_id"]),
        subject_entity_id=str(row["subject_entity_id"]),
        predicate=str(row["predicate"]),
        object_entity_id=str(row["object_entity_id"]),
        support_count=int(row["support_count"]),
        avg_confidence=float(row["avg_confidence"]),
        min_confidence=float(row["min_confidence"]),
        max_confidence=float(row["max_confidence"]),
        first_seen_at=str(row["first_seen_at"]),
        last_seen_at=str(row["last_seen_at"]),
    )


def manifest_from_row(row: sqlite3.Row) -> ManifestRecord:
    return ManifestRecord(
        source_rel_path=str(row["source_rel_path"]),
        absolute_path=str(row["absolute_path"]),
        source_type=str(row["source_type"]),
        source_domain=str(row["source_domain"]),
        document_id=optional_str(row["document_id"]),
        file_size_bytes=int(row["file_size_bytes"]),
        content_hash=str(row["blake3_hash"]),
        parent_source_rel_path=optional_str(row["parent_source_rel_path"]),
        chunk_count=int(row["chunk_count"]),
        last_seen_at=str(row["last_seen_at"]),
        last_processed_at=optional_str(row["last_processed_at"]),
        last_committed_at=optional_str(row["last_committed_at"]),
        error_message=optional_str(row["error_message"]),
        lifecycle_status=str(row["lifecycle_status"]),
        retrieval_status=str(row["retrieval_status"]),
    )


def chunk_from_row(row: sqlite3.Row) -> ChunkRecord:
    """Materialise a :class:`ChunkRecord` from a SQLite ``chunk_rows`` row.

    Constraints:
        SQLite no longer stores the embedding vector (LanceDB is canonical
        as of schema v2); the returned record therefore carries an empty
        ``vector``. Callers that need vectors must hydrate them from LanceDB
        via :func:`lxd.stores.lancedb.load_vectors_by_chunk_ids`.

    The ``cited_sources_json`` and ``wiki_links_json`` columns are read
    defensively — rows that pre-date schema v6 default to empty tuples.
    """
    return ChunkRecord(
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
        chunk_hash=str(row["chunk_hash"]),
        score_hint=str(row["score_hint"]),
        metadata_json=str(row["metadata_json"]),
        vector=[],
        embedding_model=str(row["embedding_model"]),
        embedding_dims=int(row["embedding_dims"]),
        cited_sources=_parse_string_list(row, "cited_sources_json"),
        wiki_links=_parse_string_list(row, "wiki_links_json"),
    )


def _parse_string_list(row: sqlite3.Row, column: str) -> tuple[str, ...]:
    """Parse a JSON array of strings, returning an empty tuple on error or absence."""
    try:
        raw = row[column]
    except KeyError, IndexError:
        return ()
    if not raw:
        return ()
    try:
        parsed = json.loads(raw)
    except TypeError, ValueError:
        return ()
    if not isinstance(parsed, list):
        return ()
    return tuple(str(item) for item in parsed if isinstance(item, str))
