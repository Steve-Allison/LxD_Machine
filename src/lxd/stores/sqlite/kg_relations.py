"""Canonical relations, relation evidence, graph build state, graph metadata."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from lxd.stores._sqlite_rows import canonical_relation_from_row, optional_str, row_value
from lxd.stores.models import (
    CanonicalRelationRecord,
    GraphBuildStateRecord,
    RelationEvidenceRecord,
)


def replace_canonical_relations(
    connection: sqlite3.Connection, records: list[CanonicalRelationRecord]
) -> None:
    """Truncate and rebuild the canonical relations table."""
    with connection:
        connection.execute("DELETE FROM relations")
        if records:
            connection.executemany(
                """
                INSERT INTO relations (
                    relation_id, subject_entity_id, predicate, object_entity_id,
                    support_count, avg_confidence, min_confidence, max_confidence,
                    first_seen_at, last_seen_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        r.relation_id,
                        r.subject_entity_id,
                        r.predicate,
                        r.object_entity_id,
                        r.support_count,
                        r.avg_confidence,
                        r.min_confidence,
                        r.max_confidence,
                        r.first_seen_at,
                        r.last_seen_at,
                    )
                    for r in records
                ],
            )


def load_canonical_relation(
    connection: sqlite3.Connection, relation_id: str
) -> CanonicalRelationRecord | None:
    """Load a single canonical relation by ID."""
    row = connection.execute(
        "SELECT * FROM relations WHERE relation_id = ?", (relation_id,)
    ).fetchone()
    if row is None:
        return None
    return canonical_relation_from_row(row)


def load_relations_for_entity(
    connection: sqlite3.Connection,
    entity_id: str,
    *,
    limit: int = 50,
) -> list[CanonicalRelationRecord]:
    """Load canonical relations where entity appears as subject or object."""
    rows = connection.execute(
        """
        SELECT * FROM relations
        WHERE subject_entity_id = ? OR object_entity_id = ?
        ORDER BY support_count DESC
        LIMIT ?
        """,
        (entity_id, entity_id, limit),
    ).fetchall()
    return [canonical_relation_from_row(row) for row in rows]


def load_top_predicates_for_entity(
    connection: sqlite3.Connection,
    entity_id: str,
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Return top predicates for an entity by frequency."""
    rows = connection.execute(
        """
        SELECT predicate, COUNT(*) AS cnt
        FROM relations
        WHERE subject_entity_id = ? OR object_entity_id = ?
        GROUP BY predicate
        ORDER BY cnt DESC
        LIMIT ?
        """,
        (entity_id, entity_id, limit),
    ).fetchall()
    return [{"predicate": str(row["predicate"]), "count": int(row["cnt"])} for row in rows]


def replace_relation_evidence(
    connection: sqlite3.Connection, records: list[RelationEvidenceRecord]
) -> None:
    """Truncate and rebuild the relation evidence table."""
    with connection:
        connection.execute("DELETE FROM relation_evidence")
        if records:
            connection.executemany(
                """
                INSERT INTO relation_evidence (
                    evidence_id, relation_id, chunk_id,
                    surface_subject, surface_object, evidence_text,
                    confidence, extraction_model, extracted_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        r.evidence_id,
                        r.relation_id,
                        r.chunk_id,
                        r.surface_subject,
                        r.surface_object,
                        r.evidence_text,
                        r.confidence,
                        r.extraction_model,
                        r.extracted_at,
                    )
                    for r in records
                ],
            )


def load_evidence_for_relation(
    connection: sqlite3.Connection, relation_id: str
) -> list[RelationEvidenceRecord]:
    """Load all evidence records for a canonical relation."""
    rows = connection.execute(
        """
        SELECT * FROM relation_evidence
        WHERE relation_id = ?
        ORDER BY confidence DESC
        """,
        (relation_id,),
    ).fetchall()
    return [
        RelationEvidenceRecord(
            evidence_id=str(row["evidence_id"]),
            relation_id=str(row["relation_id"]),
            chunk_id=str(row["chunk_id"]),
            surface_subject=str(row["surface_subject"]),
            surface_object=str(row["surface_object"]),
            evidence_text=str(row["evidence_text"]),
            confidence=float(row["confidence"]),
            extraction_model=str(row["extraction_model"]),
            extracted_at=str(row["extracted_at"]),
        )
        for row in rows
    ]


def count_canonical_relations(connection: sqlite3.Connection) -> int:
    """Return total canonical relation count."""
    row = connection.execute("SELECT COUNT(*) AS cnt FROM relations").fetchone()
    return int(row_value(row, "cnt"))


def count_relation_evidence(connection: sqlite3.Connection) -> int:
    """Return total relation evidence count."""
    row = connection.execute("SELECT COUNT(*) AS cnt FROM relation_evidence").fetchone()
    return int(row_value(row, "cnt"))


def begin_graph_build(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    started_at: str,
    graph_version: int,
) -> None:
    """Insert initial graph build state row."""
    with connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO graph_build_state (
                run_id, started_at, finished_at, status, current_phase, graph_version,
                relations_consolidated, evidence_rows_built, claims_extracted,
                entity_profiles_built, communities_detected, community_reports_built,
                centrality_computed, entity_embeddings_computed, llm_enrichment_count,
                notes_json
            )
            VALUES (?, ?, NULL, 'running', 'pending', ?, 0, 0, 0, 0, 0, 0, 0, 0, 0, '[]')
            """,
            (run_id, started_at, graph_version),
        )


def update_graph_build_phase(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    current_phase: str,
    **counters: int,
) -> None:
    """Update the current phase and counter columns on a graph build."""
    set_clauses = ["current_phase = ?"]
    params: list[Any] = [current_phase]
    valid_columns = {
        "relations_consolidated",
        "evidence_rows_built",
        "claims_extracted",
        "entity_profiles_built",
        "communities_detected",
        "community_reports_built",
        "centrality_computed",
        "entity_embeddings_computed",
        "llm_enrichment_count",
    }
    for key, value in counters.items():
        if key in valid_columns:
            set_clauses.append(f"{key} = ?")
            params.append(value)
    params.append(run_id)
    with connection:
        connection.execute(
            f"UPDATE graph_build_state SET {', '.join(set_clauses)} WHERE run_id = ?",
            params,
        )


def finish_graph_build(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    finished_at: str,
    status: str,
    notes: list[str],
) -> None:
    """Finalise a graph build run."""
    with connection:
        connection.execute(
            """
            UPDATE graph_build_state
            SET finished_at = ?, status = ?, notes_json = ?
            WHERE run_id = ?
            """,
            (finished_at, status, json.dumps(notes, separators=(",", ":")), run_id),
        )


def load_latest_graph_build_state(
    connection: sqlite3.Connection,
) -> GraphBuildStateRecord | None:
    """Load the most recent graph build state row."""
    row = connection.execute(
        "SELECT * FROM graph_build_state ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return None
    return GraphBuildStateRecord(
        run_id=str(row["run_id"]),
        started_at=str(row["started_at"]),
        finished_at=optional_str(row["finished_at"]),
        status=str(row["status"]),
        current_phase=str(row["current_phase"]),
        graph_version=int(row["graph_version"]),
        relations_consolidated=int(row["relations_consolidated"]),
        evidence_rows_built=int(row["evidence_rows_built"]),
        claims_extracted=int(row["claims_extracted"]),
        entity_profiles_built=int(row["entity_profiles_built"]),
        communities_detected=int(row["communities_detected"]),
        community_reports_built=int(row["community_reports_built"]),
        centrality_computed=int(row["centrality_computed"]),
        entity_embeddings_computed=int(row["entity_embeddings_computed"]),
        llm_enrichment_count=int(row["llm_enrichment_count"]),
        notes_json=str(row["notes_json"]),
    )


def upsert_graph_metadata(
    connection: sqlite3.Connection,
    key: str,
    value: str,
    updated_at: str,
) -> None:
    """Insert or update a graph metadata key-value entry."""
    with connection:
        connection.execute(
            """
            INSERT INTO graph_metadata (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
            """,
            (key, value, updated_at),
        )


def load_graph_metadata(connection: sqlite3.Connection) -> dict[str, str]:
    """Load all graph metadata key-value pairs."""
    rows = connection.execute("SELECT key, value FROM graph_metadata").fetchall()
    return {str(row["key"]): str(row["value"]) for row in rows}


def load_graph_version(connection: sqlite3.Connection) -> int:
    """Load the current graph version number, defaulting to 0."""
    row = connection.execute(
        "SELECT value FROM graph_metadata WHERE key = 'graph_version'"
    ).fetchone()
    if row is None:
        return 0
    return int(row["value"])
