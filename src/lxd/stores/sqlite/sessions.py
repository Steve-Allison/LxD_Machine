"""Learner-brief session state: ``sessions`` + ``session_turns`` rows.

Ephemeral product state for the Phase 3 SOTA roadmap (multi-turn design and
answer requests carrying an audience/modality/Bloom-target/constraints
brief). Not corpus data — exempt from the "MCP is read-only w.r.t. the
corpus" rule.
"""

import sqlite3

from lxd.stores._sqlite_rows import session_from_row, session_turn_from_row
from lxd.stores.models import SessionRecord, SessionTurnRecord


def upsert_session_brief(connection: sqlite3.Connection, record: SessionRecord) -> None:
    """Insert or update the brief fields for a session.

    Only the brief columns (``audience`` / ``modality`` / ``bloom_target`` /
    ``constraints_text``) and timestamps are written here;
    ``last_artefact_json`` is managed separately via
    :func:`update_last_artefact` so a brief update never clobbers the most
    recent artefact reference.
    """
    with connection:
        connection.execute(
            """
            INSERT INTO sessions (
                session_id, audience, modality, bloom_target, constraints_text,
                created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                audience = excluded.audience,
                modality = excluded.modality,
                bloom_target = excluded.bloom_target,
                constraints_text = excluded.constraints_text,
                updated_at = excluded.updated_at
            """,
            (
                record.session_id,
                record.audience,
                record.modality,
                record.bloom_target,
                record.constraints_text,
                record.created_at,
                record.updated_at,
            ),
        )


def load_session(connection: sqlite3.Connection, session_id: str) -> SessionRecord | None:
    """Load a session's brief + last-artefact state by ID. ``None`` if unknown."""
    row = connection.execute(
        "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()
    if row is None:
        return None
    return session_from_row(row)


def append_turn(connection: sqlite3.Connection, record: SessionTurnRecord) -> None:
    """Append one turn (user question or assistant artefact reference) to a session.

    ``INSERT OR IGNORE`` on the deterministic ``turn_id`` primary key makes
    this safe to call more than once for the same logical turn (e.g. a
    retried MCP call) without raising or duplicating rows.
    """
    with connection:
        connection.execute(
            """
            INSERT OR IGNORE INTO session_turns (
                turn_id, session_id, role, content_json, created_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                record.turn_id,
                record.session_id,
                record.role,
                record.content_json,
                record.created_at,
            ),
        )


def load_turns(
    connection: sqlite3.Connection, session_id: str, *, limit: int = 50
) -> list[SessionTurnRecord]:
    """Load a session's turns in chronological order, capped at ``limit``."""
    rows = connection.execute(
        """
        SELECT * FROM session_turns
        WHERE session_id = ?
        ORDER BY created_at ASC,
                 CASE role WHEN 'user' THEN 0 WHEN 'assistant' THEN 1 ELSE 2 END,
                 turn_id ASC
        LIMIT ?
        """,
        (session_id, limit),
    ).fetchall()
    return [session_turn_from_row(row) for row in rows]


def update_last_artefact(
    connection: sqlite3.Connection,
    *,
    session_id: str,
    last_artefact_json: str,
    updated_at: str,
) -> None:
    """Persist the most recent artefact reference for a session.

    No-ops (0 rows affected) if ``session_id`` has no ``sessions`` row yet —
    callers are expected to have called :func:`upsert_session_brief` first.
    """
    with connection:
        connection.execute(
            """
            UPDATE sessions
            SET last_artefact_json = ?, updated_at = ?
            WHERE session_id = ?
            """,
            (last_artefact_json, updated_at, session_id),
        )
