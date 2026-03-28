"""Consolidate extracted relations into canonical triples and build evidence provenance."""

from __future__ import annotations

import sqlite3
from collections import defaultdict

import structlog

from lxd.domain.ids import blake3_hex
from lxd.stores.models import (
    CanonicalRelationRecord,
    ExtractedRelationRecord,
    RelationEvidenceRecord,
)
from lxd.stores.sqlite import (
    load_all_extracted_relations,
    replace_canonical_relations,
    replace_relation_evidence,
)

_log = structlog.get_logger(__name__)


def consolidate_relations(connection: sqlite3.Connection) -> tuple[int, int]:
    """Consolidate extracted_relations into canonical relations + evidence.

    Reads all rows from extracted_relations, groups by (subject, predicate, object),
    and produces one canonical relation per unique triple with aggregated statistics.
    Also builds per-chunk evidence records with surface forms and source text.

    Returns:
        Tuple of (canonical_relation_count, evidence_record_count).
    """
    extracted = load_all_extracted_relations(connection)
    if not extracted:
        _log.info("consolidate_relations: no extracted relations found")
        replace_canonical_relations(connection, [])
        replace_relation_evidence(connection, [])
        return 0, 0

    # Step 1: group by (subject, predicate, object)
    groups: dict[tuple[str, str, str], list[ExtractedRelationRecord]] = defaultdict(list)
    for row in extracted:
        key = (row.subject_entity_id, row.predicate, row.object_entity_id)
        groups[key].append(row)

    canonical_records: list[CanonicalRelationRecord] = []
    evidence_records: list[RelationEvidenceRecord] = []

    # Step 2: build canonical relations and evidence
    for (subject, predicate, obj), rows in groups.items():
        relation_id = blake3_hex(subject, predicate, obj)
        confidences = [r.confidence for r in rows]
        timestamps = [r.extracted_at for r in rows]

        canonical_records.append(
            CanonicalRelationRecord(
                relation_id=relation_id,
                subject_entity_id=subject,
                predicate=predicate,
                object_entity_id=obj,
                support_count=len(rows),
                avg_confidence=sum(confidences) / len(confidences),
                min_confidence=min(confidences),
                max_confidence=max(confidences),
                first_seen_at=min(timestamps),
                last_seen_at=max(timestamps),
            )
        )

        # Build evidence records — one per (relation × chunk) pair
        chunk_groups: dict[str, list[ExtractedRelationRecord]] = defaultdict(list)
        for row in rows:
            chunk_groups[row.chunk_id].append(row)

        for chunk_id, chunk_rows in chunk_groups.items():
            best = max(chunk_rows, key=lambda r: r.confidence)
            evidence_id = blake3_hex(relation_id, chunk_id)

            # Look up chunk text and surface forms
            chunk_text, surface_subject, surface_object = _lookup_evidence_context(
                connection, chunk_id, subject, obj
            )

            evidence_records.append(
                RelationEvidenceRecord(
                    evidence_id=evidence_id,
                    relation_id=relation_id,
                    chunk_id=chunk_id,
                    surface_subject=surface_subject,
                    surface_object=surface_object,
                    evidence_text=chunk_text,
                    confidence=best.confidence,
                    extraction_model=best.extraction_model,
                    extracted_at=best.extracted_at,
                )
            )

    # Step 3: truncate and rebuild both tables
    replace_canonical_relations(connection, canonical_records)
    replace_relation_evidence(connection, evidence_records)

    _log.info(
        "consolidate_relations complete",
        canonical_relations=len(canonical_records),
        evidence_records=len(evidence_records),
    )
    return len(canonical_records), len(evidence_records)


def _lookup_evidence_context(
    connection: sqlite3.Connection,
    chunk_id: str,
    subject_entity_id: str,
    object_entity_id: str,
) -> tuple[str, str, str]:
    """Look up chunk text and surface forms for evidence records.

    Returns:
        Tuple of (chunk_text, surface_subject, surface_object).
    """
    # Get chunk text
    chunk_row = connection.execute(
        "SELECT text FROM chunk_rows WHERE chunk_id = ?", (chunk_id,)
    ).fetchone()
    chunk_text = str(chunk_row["text"]) if chunk_row else ""

    # Get surface forms from mention_rows
    mention_rows = connection.execute(
        """
        SELECT entity_id, surface_form
        FROM mention_rows
        WHERE chunk_id = ? AND entity_id IN (?, ?)
        ORDER BY start_char
        """,
        (chunk_id, subject_entity_id, object_entity_id),
    ).fetchall()

    surface_subject = subject_entity_id
    surface_object = object_entity_id
    for mention in mention_rows:
        entity_id = str(mention["entity_id"])
        surface = str(mention["surface_form"])
        if entity_id == subject_entity_id:
            surface_subject = surface
        elif entity_id == object_entity_id:
            surface_object = surface

    return chunk_text, surface_subject, surface_object
