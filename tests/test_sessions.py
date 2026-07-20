"""Tests for the sessions/session_turns schema (migration 0010) and store helpers."""

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from lxd.stores.models import SessionRecord, SessionTurnRecord
from lxd.stores.schema import CURRENT_SCHEMA_VERSION, ensure_schema, get_schema_version
from lxd.stores.sqlite.sessions import (
    append_turn,
    load_session,
    load_turns,
    update_last_artefact,
    upsert_session_brief,
)


@pytest.fixture
def db(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    """Open an isolated, migrated SQLite database for each test."""
    connection = sqlite3.connect(tmp_path / "lxd.sqlite3")
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON;")
    ensure_schema(connection)
    try:
        yield connection
    finally:
        connection.close()


def _brief_record(
    session_id: str = "s1",
    *,
    audience: str | None = "new admins",
    modality: str | None = "ILT",
    bloom_target: str | None = "apply",
    constraints: str | None = "1 day budget",
    created_at: str = "2026-01-01T00:00:00+00:00",
    updated_at: str = "2026-01-01T00:00:00+00:00",
) -> SessionRecord:
    return SessionRecord(
        session_id=session_id,
        audience=audience,
        modality=modality,
        bloom_target=bloom_target,
        constraints_text=constraints,
        created_at=created_at,
        updated_at=updated_at,
    )


def test_migration_0010_lands_sessions_tables_at_current_version(
    db: sqlite3.Connection,
) -> None:
    assert get_schema_version(db) == CURRENT_SCHEMA_VERSION
    tables = {
        row["name"]
        for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name IN ('sessions', 'session_turns')"
        ).fetchall()
    }
    assert tables == {"sessions", "session_turns"}


def test_migration_0010_is_idempotent(db: sqlite3.Connection) -> None:
    ensure_schema(db)  # second call must be a clean no-op
    assert get_schema_version(db) == CURRENT_SCHEMA_VERSION


def test_upsert_and_load_session_round_trips_brief_fields(db: sqlite3.Connection) -> None:
    upsert_session_brief(db, _brief_record())
    loaded = load_session(db, "s1")
    assert loaded is not None
    assert loaded.session_id == "s1"
    assert loaded.audience == "new admins"
    assert loaded.modality == "ILT"
    assert loaded.bloom_target == "apply"
    assert loaded.constraints_text == "1 day budget"
    assert loaded.last_artefact_json == "{}"


def test_load_session_returns_none_for_unknown_id(db: sqlite3.Connection) -> None:
    assert load_session(db, "does-not-exist") is None


def test_upsert_session_brief_updates_fields_without_touching_last_artefact(
    db: sqlite3.Connection,
) -> None:
    upsert_session_brief(db, _brief_record())
    update_last_artefact(
        db,
        session_id="s1",
        last_artefact_json='{"citations": ["a"]}',
        updated_at="2026-01-02T00:00:00+00:00",
    )
    upsert_session_brief(
        db,
        _brief_record(audience="revised audience", updated_at="2026-01-03T00:00:00+00:00"),
    )
    loaded = load_session(db, "s1")
    assert loaded is not None
    assert loaded.audience == "revised audience"
    assert loaded.last_artefact_json == '{"citations": ["a"]}', (
        "a brief-only upsert must not clobber the last-artefact reference"
    )


def test_update_last_artefact_is_noop_for_unknown_session(db: sqlite3.Connection) -> None:
    """No FK-less parent row exists yet — the UPDATE affects zero rows, not an error."""
    update_last_artefact(
        db, session_id="ghost", last_artefact_json="{}", updated_at="2026-01-01T00:00:00+00:00"
    )
    assert load_session(db, "ghost") is None


def test_append_turn_persists_role_and_content(db: sqlite3.Connection) -> None:
    upsert_session_brief(db, _brief_record())
    append_turn(
        db,
        SessionTurnRecord(
            turn_id="t1",
            session_id="s1",
            role="user",
            content_json='{"question": "what is ADDIE?"}',
            created_at="2026-01-01T00:00:01+00:00",
        ),
    )
    append_turn(
        db,
        SessionTurnRecord(
            turn_id="t2",
            session_id="s1",
            role="assistant",
            content_json='{"answer_status": "answered", "citations": ["a"]}',
            created_at="2026-01-01T00:00:02+00:00",
        ),
    )
    turns = load_turns(db, "s1")
    assert [t.turn_id for t in turns] == ["t1", "t2"]
    assert turns[0].role == "user"
    assert turns[1].role == "assistant"
    assert turns[1].content_json == '{"answer_status": "answered", "citations": ["a"]}'


def test_append_turn_is_idempotent_on_turn_id(db: sqlite3.Connection) -> None:
    upsert_session_brief(db, _brief_record())
    record = SessionTurnRecord(
        turn_id="t1",
        session_id="s1",
        role="user",
        content_json="{}",
        created_at="2026-01-01T00:00:01+00:00",
    )
    append_turn(db, record)
    append_turn(db, record)  # retried call must not duplicate or raise
    turns = load_turns(db, "s1")
    assert len(turns) == 1


def test_append_turn_requires_a_parent_session_row(db: sqlite3.Connection) -> None:
    """The FK is enforced — a turn cannot be appended for an unknown session."""
    with pytest.raises(sqlite3.IntegrityError):
        append_turn(
            db,
            SessionTurnRecord(
                turn_id="t1",
                session_id="ghost",
                role="user",
                content_json="{}",
                created_at="2026-01-01T00:00:01+00:00",
            ),
        )


def test_load_turns_respects_limit(db: sqlite3.Connection) -> None:
    upsert_session_brief(db, _brief_record())
    for i in range(5):
        append_turn(
            db,
            SessionTurnRecord(
                turn_id=f"t{i}",
                session_id="s1",
                role="user",
                content_json="{}",
                created_at=f"2026-01-01T00:00:{i:02d}+00:00",
            ),
        )
    turns = load_turns(db, "s1", limit=2)
    assert [t.turn_id for t in turns] == ["t0", "t1"]


def test_update_last_artefact_persists_and_bumps_updated_at(db: sqlite3.Connection) -> None:
    upsert_session_brief(db, _brief_record())
    update_last_artefact(
        db,
        session_id="s1",
        last_artefact_json='{"citations": ["x", "y"]}',
        updated_at="2026-02-01T00:00:00+00:00",
    )
    loaded = load_session(db, "s1")
    assert loaded is not None
    assert loaded.last_artefact_json == '{"citations": ["x", "y"]}'
    assert loaded.updated_at == "2026-02-01T00:00:00+00:00"
