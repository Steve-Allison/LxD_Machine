"""Claims insert / load / count."""

import sqlite3

from lxd.stores._sqlite_rows import claim_from_row, row_value
from lxd.stores.models import ClaimRecord


def insert_claims(connection: sqlite3.Connection, records: list[ClaimRecord]) -> int:
    """Insert claim records, skipping duplicates."""
    if not records:
        return 0
    with connection:
        connection.executemany(
            """
            INSERT OR IGNORE INTO claims (
                claim_id, chunk_id, document_id, source_rel_path,
                claim_text, subject_entity_id, object_entity_id,
                claim_type, confidence, extraction_model, extracted_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    r.claim_id,
                    r.chunk_id,
                    r.document_id,
                    r.source_rel_path,
                    r.claim_text,
                    r.subject_entity_id,
                    r.object_entity_id,
                    r.claim_type,
                    r.confidence,
                    r.extraction_model,
                    r.extracted_at,
                )
                for r in records
            ],
        )
    return len(records)


def load_claims_for_entities(
    connection: sqlite3.Connection,
    entity_ids: list[str],
    *,
    limit: int = 50,
) -> list[ClaimRecord]:
    """Load claims linked to any of the given entity IDs, ranked by confidence."""
    if not entity_ids:
        return []
    placeholders = ",".join("?" * len(entity_ids))
    rows = connection.execute(
        f"""
        SELECT * FROM claims
        WHERE subject_entity_id IN ({placeholders})
           OR object_entity_id IN ({placeholders})
        ORDER BY confidence DESC
        LIMIT ?
        """,
        [*entity_ids, *entity_ids, limit],
    ).fetchall()
    return [claim_from_row(row) for row in rows]


def load_claims_for_chunk(connection: sqlite3.Connection, chunk_id: str) -> list[ClaimRecord]:
    """Load all claims extracted from a specific chunk."""
    rows = connection.execute(
        "SELECT * FROM claims WHERE chunk_id = ? ORDER BY confidence DESC",
        (chunk_id,),
    ).fetchall()
    return [claim_from_row(row) for row in rows]


def count_claims(connection: sqlite3.Connection) -> int:
    """Return total claim count."""
    row = connection.execute("SELECT COUNT(*) AS cnt FROM claims").fetchone()
    return int(row_value(row, "cnt"))


def load_chunk_ids_with_claims(connection: sqlite3.Connection) -> set[str]:
    """Return chunk IDs that already have claims extracted."""
    rows = connection.execute("SELECT DISTINCT chunk_id FROM claims").fetchall()
    return {str(row["chunk_id"]) for row in rows}
