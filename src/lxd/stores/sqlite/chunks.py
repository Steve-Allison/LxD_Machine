"""Chunk rows, mention rows, extracted relations, and per-chunk centrality signals."""

import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

from lxd.stores._sqlite_rows import chunk_from_row, mention_id
from lxd.stores.models import (
    ChunkCentralitySignals,
    ChunkRecord,
    EntityMentionResult,
    ExtractedRelationRecord,
    MentionRecord,
)


def replace_source_chunks(
    connection: sqlite3.Connection,
    *,
    source_rel_path: str,
    chunk_records: list[ChunkRecord],
    mention_records: list[MentionRecord],
    relation_records: list[ExtractedRelationRecord] | None = None,
) -> None:
    """Replace all vector chunks for one source path."""
    with connection:
        connection.execute("DELETE FROM chunk_rows WHERE source_rel_path = ?", (source_rel_path,))
        connection.execute("DELETE FROM mention_rows WHERE source_rel_path = ?", (source_rel_path,))
        if chunk_records:
            connection.executemany(
                """
                INSERT INTO chunk_rows (
                    chunk_id,
                    document_id,
                    source_rel_path,
                    source_filename,
                    source_type,
                    source_domain,
                    source_hash,
                    citation_label,
                    chunk_index,
                    chunk_occurrence,
                    token_count,
                    text,
                    chunk_hash,
                    score_hint,
                    metadata_json,
                    embedding_model,
                    embedding_dims,
                    cited_sources_json,
                    wiki_links_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        record.chunk_id,
                        record.document_id,
                        record.source_rel_path,
                        record.source_filename,
                        record.source_type,
                        record.source_domain,
                        record.source_hash,
                        record.citation_label,
                        record.chunk_index,
                        record.chunk_occurrence,
                        record.token_count,
                        record.text,
                        record.chunk_hash,
                        record.score_hint,
                        record.metadata_json,
                        record.embedding_model,
                        record.embedding_dims,
                        json.dumps(list(record.cited_sources)),
                        json.dumps(list(record.wiki_links)),
                    )
                    for record in chunk_records
                ],
            )
        if mention_records:
            rel_path = chunk_records[0].source_rel_path if chunk_records else source_rel_path
            source_domain = chunk_records[0].source_domain if chunk_records else ""
            source_filename = Path(rel_path).name if rel_path else ""
            connection.executemany(
                """
                INSERT INTO mention_rows (
                    mention_id,
                    entity_id,
                    term_source,
                    source_domain,
                    source_rel_path,
                    source_filename,
                    chunk_id,
                    surface_form,
                    start_char,
                    end_char
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        mention_id(record),
                        record.entity_id,
                        record.term_source,
                        source_domain,
                        rel_path,
                        source_filename,
                        record.chunk_id,
                        record.surface_form,
                        record.start_char,
                        record.end_char,
                    )
                    for record in mention_records
                ],
            )
        if relation_records:
            connection.executemany(
                """
                INSERT OR IGNORE INTO extracted_relations (
                    relation_id,
                    chunk_id,
                    document_id,
                    source_rel_path,
                    subject_entity_id,
                    predicate,
                    object_entity_id,
                    confidence,
                    extraction_model,
                    extracted_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        record.relation_id,
                        record.chunk_id,
                        record.document_id,
                        record.source_rel_path,
                        record.subject_entity_id,
                        record.predicate,
                        record.object_entity_id,
                        record.confidence,
                        record.extraction_model,
                        record.extracted_at,
                    )
                    for record in relation_records
                ],
            )


def load_chunk_records_for_source(
    connection: sqlite3.Connection, source_rel_path: str
) -> list[ChunkRecord]:
    """Load persisted chunk records for a source path."""
    rows = connection.execute(
        """
        SELECT
            chunk_id,
            document_id,
            source_rel_path,
            source_filename,
            source_type,
            source_domain,
            source_hash,
            citation_label,
            chunk_index,
            chunk_occurrence,
            token_count,
            text,
            chunk_hash,
            score_hint,
            metadata_json,
            embedding_model,
            embedding_dims,
            cited_sources_json,
            wiki_links_json
        FROM chunk_rows
        WHERE source_rel_path = ?
        ORDER BY chunk_index
        """,
        (source_rel_path,),
    ).fetchall()
    return [chunk_from_row(row) for row in rows]


def load_mentions_for_source(
    connection: sqlite3.Connection, source_rel_path: str
) -> dict[str, list[MentionRecord]]:
    """Load persisted mentions grouped by chunk ID for a source."""
    rows = connection.execute(
        """
        SELECT
            chunk_id,
            entity_id,
            term_source,
            surface_form,
            start_char,
            end_char
        FROM mention_rows
        WHERE source_rel_path = ?
        ORDER BY chunk_id, start_char, end_char, entity_id
        """,
        (source_rel_path,),
    ).fetchall()
    grouped: dict[str, list[MentionRecord]] = defaultdict(list)
    for row in rows:
        record = MentionRecord(
            chunk_id=str(row["chunk_id"]),
            entity_id=str(row["entity_id"]),
            term_source=str(row["term_source"]),
            surface_form=str(row["surface_form"]),
            start_char=int(row["start_char"]),
            end_char=int(row["end_char"]),
        )
        grouped[record.chunk_id].append(record)
    return dict(grouped)


def find_chunks_by_entity_mentions(
    connection: sqlite3.Connection,
    entity_ids: list[str],
    *,
    limit: int = 50,
) -> list[EntityMentionResult]:
    """Find chunks matching one or more entity mentions."""
    if not entity_ids:
        return []
    placeholders = ",".join("?" * len(entity_ids))
    rows = connection.execute(
        f"""
        WITH matched AS (
            SELECT chunk_id, COUNT(DISTINCT entity_id) AS entity_match_count
            FROM mention_rows
            WHERE entity_id IN ({placeholders})
            GROUP BY chunk_id
        )
        SELECT
            c.chunk_id,
            c.document_id,
            c.source_rel_path,
            c.citation_label,
            c.chunk_index,
            c.text,
            c.score_hint,
            c.metadata_json,
            m.entity_match_count
        FROM chunk_rows c
        JOIN matched m ON c.chunk_id = m.chunk_id
        ORDER BY m.entity_match_count DESC, c.chunk_index ASC
        LIMIT ?
        """,
        (*entity_ids, limit * 4),
    ).fetchall()
    total = len(entity_ids)
    seen_sources: set[str] = set()
    results: list[EntityMentionResult] = []
    for row in rows:
        source_rel_path = str(row["source_rel_path"])
        if source_rel_path in seen_sources:
            continue
        seen_sources.add(source_rel_path)
        results.append(
            EntityMentionResult(
                chunk_id=str(row["chunk_id"]),
                document_id=str(row["document_id"]),
                source_rel_path=source_rel_path,
                citation_label=str(row["citation_label"]),
                chunk_index=int(row["chunk_index"]),
                text=str(row["text"]),
                score_hint=str(row["score_hint"]),
                metadata_json=str(row["metadata_json"]),
                entity_match_count=int(row["entity_match_count"]),
                total_entity_ids=total,
            )
        )
        if len(results) >= limit:
            break
    return results


def load_corpus_related_entity_ids(
    connection: sqlite3.Connection,
    entity_ids: list[str],
    *,
    min_confidence: float = 0.5,
    max_results: int = 20,
) -> list[str]:
    """Return entity IDs strongly related to ``entity_ids`` via extracted corpus relations."""
    if not entity_ids:
        return []
    placeholders = ",".join("?" * len(entity_ids))
    rows = connection.execute(
        f"""
        SELECT subject_entity_id, object_entity_id, confidence
        FROM extracted_relations
        WHERE (subject_entity_id IN ({placeholders}) OR object_entity_id IN ({placeholders}))
          AND confidence >= ?
        ORDER BY confidence DESC
        LIMIT ?
        """,
        [*entity_ids, *entity_ids, min_confidence, max_results * 4],
    ).fetchall()
    seed_set = set(entity_ids)
    related: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for candidate in (str(row["subject_entity_id"]), str(row["object_entity_id"])):
            if candidate not in seed_set and candidate not in seen:
                seen.add(candidate)
                related.append(candidate)
                if len(related) >= max_results:
                    return related
    return related


def load_corpus_relations_for_entity(
    connection: sqlite3.Connection,
    entity_id: str,
    *,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Return extracted corpus relations where ``entity_id`` appears as subject or object."""
    rows = connection.execute(
        """
        SELECT subject_entity_id, predicate, object_entity_id, confidence,
               extraction_model, source_rel_path, chunk_id
        FROM extracted_relations
        WHERE subject_entity_id = ? OR object_entity_id = ?
        ORDER BY confidence DESC
        LIMIT ?
        """,
        (entity_id, entity_id, limit),
    ).fetchall()
    return [
        {
            "subject": str(row["subject_entity_id"]),
            "predicate": str(row["predicate"]),
            "object": str(row["object_entity_id"]),
            "confidence": float(row["confidence"]),
            "source_rel_path": str(row["source_rel_path"]),
            "chunk_id": str(row["chunk_id"]),
        }
        for row in rows
    ]


def load_chunk_record_by_id(
    connection: sqlite3.Connection, chunk_id: str
) -> ChunkRecord | None:
    """Load a single persisted chunk record by ID, or ``None`` if absent.

    Used by the graph-as-retrieval-lane path to hydrate the chunk text
    and source metadata backing a claim, and by the query pipeline to
    append a claim-linked chunk that fell outside the dense/rerank
    prefix.
    """
    row = connection.execute(
        """
        SELECT
            chunk_id,
            document_id,
            source_rel_path,
            source_filename,
            source_type,
            source_domain,
            source_hash,
            citation_label,
            chunk_index,
            chunk_occurrence,
            token_count,
            text,
            chunk_hash,
            score_hint,
            metadata_json,
            embedding_model,
            embedding_dims,
            cited_sources_json,
            wiki_links_json
        FROM chunk_rows
        WHERE chunk_id = ?
        """,
        (chunk_id,),
    ).fetchone()
    if row is None:
        return None
    return chunk_from_row(row)


def load_relation_chunk_ids(
    connection: sqlite3.Connection,
    entity_ids: list[str],
) -> set[str]:
    """Return chunk IDs that contain an extracted relation involving any of ``entity_ids``."""
    if not entity_ids:
        return set()
    placeholders = ",".join("?" * len(entity_ids))
    rows = connection.execute(
        f"""
        SELECT DISTINCT chunk_id
        FROM extracted_relations
        WHERE subject_entity_id IN ({placeholders})
           OR object_entity_id IN ({placeholders})
        """,
        [*entity_ids, *entity_ids],
    ).fetchall()
    return {str(row["chunk_id"]) for row in rows}


def load_chunk_centrality_signals(
    connection: sqlite3.Connection, chunk_ids: list[str]
) -> dict[str, ChunkCentralitySignals]:
    """Return ``chunk_id -> ChunkCentralitySignals`` for the given chunks.

    Resolves each chunk's mentioned entities, joins to ``entity_profiles``
    (built by ``pixi run build-graph``), and returns:

    * ``max_pagerank`` — the highest PageRank across the chunk's mentioned
      entities, treated as the chunk's "graph importance" signal.
    * ``community_ids`` — distinct community ids the chunk's entities
      belong to, sorted ascending. Used for community-aware
      diversification at retrieval time.

    Returns an empty dict if ``chunk_ids`` is empty. Chunks with no rows
    in ``entity_profiles`` (graph not yet built) are silently absent from
    the result; callers default-fill with :class:`ChunkCentralitySignals`.
    """
    if not chunk_ids:
        return {}
    placeholders = ",".join("?" * len(chunk_ids))
    rows = connection.execute(
        f"""
        SELECT
            m.chunk_id AS chunk_id,
            MAX(p.pagerank) AS max_pagerank,
            GROUP_CONCAT(DISTINCT p.community_id) AS community_ids
        FROM mention_rows m
        JOIN entity_profiles p ON m.entity_id = p.entity_id
        WHERE m.chunk_id IN ({placeholders})
        GROUP BY m.chunk_id
        """,
        list(chunk_ids),
    ).fetchall()
    result: dict[str, ChunkCentralitySignals] = {}
    for row in rows:
        chunk_id = str(row["chunk_id"])
        max_pr_value = row["max_pagerank"]
        max_pr = float(max_pr_value) if max_pr_value is not None else 0.0
        raw_comm = row["community_ids"]
        community_ids: tuple[int, ...] = ()
        if isinstance(raw_comm, str) and raw_comm:
            community_ids = tuple(
                sorted(
                    {
                        int(token)
                        for token in raw_comm.split(",")
                        if token.strip().lstrip("-").isdigit()
                    }
                )
            )
        result[chunk_id] = ChunkCentralitySignals(
            max_pagerank=max_pr,
            community_ids=community_ids,
        )
    return result


def load_all_extracted_relations(
    connection: sqlite3.Connection,
) -> list[ExtractedRelationRecord]:
    """Load all rows from extracted_relations."""
    rows = connection.execute(
        """
        SELECT relation_id, chunk_id, document_id, source_rel_path,
               subject_entity_id, predicate, object_entity_id,
               confidence, extraction_model, extracted_at
        FROM extracted_relations
        ORDER BY subject_entity_id, predicate, object_entity_id
        """
    ).fetchall()
    return [
        ExtractedRelationRecord(
            relation_id=str(row["relation_id"]),
            chunk_id=str(row["chunk_id"]),
            document_id=str(row["document_id"]),
            source_rel_path=str(row["source_rel_path"]),
            subject_entity_id=str(row["subject_entity_id"]),
            predicate=str(row["predicate"]),
            object_entity_id=str(row["object_entity_id"]),
            confidence=float(row["confidence"]),
            extraction_model=str(row["extraction_model"]),
            extracted_at=str(row["extracted_at"]),
        )
        for row in rows
    ]


def load_entity_mention_stats(
    connection: sqlite3.Connection,
) -> dict[str, dict[str, int]]:
    """Load per-entity mention statistics (chunk_count, doc_count, mention_count)."""
    rows = connection.execute(
        """
        SELECT
            m.entity_id,
            COUNT(DISTINCT m.chunk_id) AS chunk_count,
            COUNT(DISTINCT c.source_rel_path) AS doc_count,
            COUNT(*) AS mention_count
        FROM mention_rows m
        JOIN chunk_rows c ON m.chunk_id = c.chunk_id
        GROUP BY m.entity_id
        """
    ).fetchall()
    return {
        str(row["entity_id"]): {
            "chunk_count": int(row["chunk_count"]),
            "doc_count": int(row["doc_count"]),
            "mention_count": int(row["mention_count"]),
        }
        for row in rows
    }


def load_chunk_ids_for_entity(
    connection: sqlite3.Connection, entity_id: str, *, limit: int = 100
) -> list[str]:
    """Return chunk IDs mentioning an entity, ordered by mention frequency."""
    rows = connection.execute(
        """
        SELECT chunk_id, COUNT(*) AS cnt
        FROM mention_rows
        WHERE entity_id = ?
        GROUP BY chunk_id
        ORDER BY cnt DESC
        LIMIT ?
        """,
        (entity_id, limit),
    ).fetchall()
    return [str(row["chunk_id"]) for row in rows]
