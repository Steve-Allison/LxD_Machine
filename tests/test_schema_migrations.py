"""Tests for schema migrations 4 (ghost-FK repair) and 5 (telemetry columns),
plus the post-migration integrity verification.

These cover the regression that triggered the 2026-05-01 ingest failure: a
prior legacy migration left the FK definitions of three child tables pointing
at a transient ``chunk_rows_v2_legacy`` table, so every per-file delete
cascade failed with ``no such table: main.chunk_rows_v2_legacy``.
"""

from __future__ import annotations

import sqlite3

import pytest

from lxd.stores._base_ddl import BASE_SCHEMA_DDL
from lxd.stores.schema import (
    CURRENT_SCHEMA_VERSION,
    SchemaIntegrityError,
    ensure_schema,
    get_schema_version,
    verify_schema_integrity,
)


def _open_inmem() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON;")
    return connection


def _disable_fks(connection: sqlite3.Connection) -> None:
    """Mirror the production migration: SQLite tolerates orphan FK *references*
    (FK pointing at a nonexistent table) only when ``foreign_keys=OFF`` at
    DDL time. The bug we're testing was created in exactly that environment.
    """
    connection.execute("PRAGMA foreign_keys=OFF;")


def test_ensure_schema_on_fresh_db_lands_at_current_version() -> None:
    connection = _open_inmem()
    try:
        ensure_schema(connection)
        assert get_schema_version(connection) == CURRENT_SCHEMA_VERSION
    finally:
        connection.close()


def test_ensure_schema_is_idempotent() -> None:
    connection = _open_inmem()
    try:
        ensure_schema(connection)
        ensure_schema(connection)  # second call must be a clean no-op
        assert get_schema_version(connection) == CURRENT_SCHEMA_VERSION
    finally:
        connection.close()


def test_migration_0004_repairs_ghost_fk_in_extracted_relations() -> None:
    """Simulate the live-DB state: extracted_relations has FK referencing
    ``chunk_rows_v2_legacy``. After migration 4 it must reference
    ``chunk_rows`` and the integrity check must pass."""
    connection = _open_inmem()
    try:
        # Start from the canonical baseline so all required tables exist,
        # then run migrations 1-3 first so the DB is in a production-like
        # shape before we inject the ghost FK.
        ensure_schema(connection)
        # Poison extracted_relations with the ghost FK shape, and rewind
        # user_version to 3 so migration 4 runs again.
        _disable_fks(connection)
        connection.executescript(
            """
            DROP TABLE extracted_relations;
            CREATE TABLE extracted_relations (
                relation_id TEXT PRIMARY KEY,
                chunk_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                source_rel_path TEXT NOT NULL,
                subject_entity_id TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object_entity_id TEXT NOT NULL,
                confidence REAL NOT NULL,
                extraction_model TEXT NOT NULL,
                extracted_at TEXT NOT NULL,
                FOREIGN KEY(chunk_id) REFERENCES "chunk_rows_v2_legacy"(chunk_id)
                    ON DELETE CASCADE
            );
            """
        )
        connection.execute("PRAGMA user_version = 3;")

        sql_before = connection.execute(
            "SELECT sql FROM sqlite_master WHERE name='extracted_relations'"
        ).fetchone()["sql"]
        assert "chunk_rows_v2_legacy" in sql_before

        ensure_schema(connection)

        sql_after = connection.execute(
            "SELECT sql FROM sqlite_master WHERE name='extracted_relations'"
        ).fetchone()["sql"]
        assert "chunk_rows_v2_legacy" not in sql_after
        assert "chunk_rows" in sql_after
        assert get_schema_version(connection) == CURRENT_SCHEMA_VERSION
    finally:
        connection.close()


def test_migration_0004_preserves_existing_data() -> None:
    """The repair must copy every row of the affected child tables."""
    connection = _open_inmem()
    try:
        ensure_schema(connection)
        # Seed parent rows so child FK rows have valid targets when
        # foreign_keys is re-enabled at integrity-check time.
        connection.executescript(
            """
            INSERT INTO corpus_manifest VALUES
                ('p1', '/p1', 'markdown', 'd', 'd1', 'h', 0,
                 NULL, 'complete', 'searchable', 1,
                 '2026-01-01', '2026-01-01', '2026-01-01', NULL);
            INSERT INTO chunk_rows VALUES
                ('c1', 'd1', 'p1', 'p1.md', 'markdown', 'd', 'h',
                 'p1#1', 0, 0, 1, 'text', 'h1', 'normal', '{}', 'm', 1);
            """
        )
        _disable_fks(connection)
        # Override the three child tables with the ghost-FK shape so we can
        # test the repair path on a populated set.
        connection.executescript(
            """
            DROP TABLE extracted_relations;
            DROP TABLE claims;
            DROP TABLE relation_evidence;
            CREATE TABLE extracted_relations (
                relation_id TEXT PRIMARY KEY, chunk_id TEXT NOT NULL,
                document_id TEXT NOT NULL, source_rel_path TEXT NOT NULL,
                subject_entity_id TEXT NOT NULL, predicate TEXT NOT NULL,
                object_entity_id TEXT NOT NULL, confidence REAL NOT NULL,
                extraction_model TEXT NOT NULL, extracted_at TEXT NOT NULL,
                FOREIGN KEY(chunk_id) REFERENCES "chunk_rows_v2_legacy"(chunk_id)
                    ON DELETE CASCADE
            );
            CREATE TABLE claims (
                claim_id TEXT PRIMARY KEY, chunk_id TEXT NOT NULL,
                document_id TEXT NOT NULL, source_rel_path TEXT NOT NULL,
                claim_text TEXT NOT NULL, subject_entity_id TEXT,
                object_entity_id TEXT,
                claim_type TEXT NOT NULL DEFAULT 'assertion',
                confidence REAL NOT NULL, extraction_model TEXT NOT NULL,
                extracted_at TEXT NOT NULL,
                FOREIGN KEY(chunk_id) REFERENCES "chunk_rows_v2_legacy"(chunk_id)
                    ON DELETE CASCADE
            );
            CREATE TABLE relation_evidence (
                evidence_id TEXT PRIMARY KEY, relation_id TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                surface_subject TEXT NOT NULL, surface_object TEXT NOT NULL,
                evidence_text TEXT NOT NULL, confidence REAL NOT NULL,
                extraction_model TEXT NOT NULL, extracted_at TEXT NOT NULL,
                FOREIGN KEY(relation_id) REFERENCES relations(relation_id) ON DELETE CASCADE,
                FOREIGN KEY(chunk_id) REFERENCES "chunk_rows_v2_legacy"(chunk_id)
                    ON DELETE CASCADE
            );

            INSERT INTO extracted_relations VALUES
                ('r1', 'c1', 'd1', 'p1', 's1', 'P1', 'o1', 0.9, 'm', '2026-01-01');
            INSERT INTO claims VALUES
                ('cl1', 'c1', 'd1', 'p1', 'text', 's1', 'o1', 'assertion', 0.9, 'm', '2026-01-01');
            INSERT INTO relations VALUES
                ('rel1', 's1', 'P1', 'o1', 1, 0.9, 0.9, 0.9, '2026-01-01', '2026-01-01');
            INSERT INTO relation_evidence VALUES
                ('e1', 'rel1', 'c1', 's', 'o', 'text', 0.9, 'm', '2026-01-01');
            """
        )
        connection.execute("PRAGMA user_version = 3;")

        ensure_schema(connection)

        # All rows must have survived the rebuild.
        assert connection.execute("SELECT relation_id FROM extracted_relations").fetchall() == [
            ("r1",)
        ] or list(
            row["relation_id"]
            for row in connection.execute("SELECT relation_id FROM extracted_relations")
        ) == ["r1"]
        rows = connection.execute("SELECT claim_id FROM claims").fetchall()
        assert [str(r["claim_id"]) for r in rows] == ["cl1"]
        rows = connection.execute("SELECT evidence_id FROM relation_evidence").fetchall()
        assert [str(r["evidence_id"]) for r in rows] == ["e1"]
    finally:
        connection.close()


def test_verify_schema_integrity_catches_ghost_fk() -> None:
    """The integrity check must surface a ghost FK as a hard error."""
    connection = _open_inmem()
    try:
        connection.executescript(BASE_SCHEMA_DDL)
        _disable_fks(connection)
        connection.executescript(
            """
            DROP TABLE extracted_relations;
            CREATE TABLE extracted_relations (
                relation_id TEXT PRIMARY KEY, chunk_id TEXT NOT NULL,
                document_id TEXT NOT NULL, source_rel_path TEXT NOT NULL,
                subject_entity_id TEXT NOT NULL, predicate TEXT NOT NULL,
                object_entity_id TEXT NOT NULL, confidence REAL NOT NULL,
                extraction_model TEXT NOT NULL, extracted_at TEXT NOT NULL,
                FOREIGN KEY(chunk_id) REFERENCES "definitely_not_a_table"(chunk_id)
            );
            INSERT INTO extracted_relations VALUES
                ('r1', 'c-orphan', 'd', 'p', 's', 'P', 'o', 0.5, 'm', '2026-01-01');
            """
        )
        with pytest.raises(SchemaIntegrityError) as exc_info:
            verify_schema_integrity(connection)
        assert "extracted_relations" in str(exc_info.value)
    finally:
        connection.close()


def test_migration_0005_adds_telemetry_columns() -> None:
    connection = _open_inmem()
    try:
        ensure_schema(connection)
        cols = {
            row["name"] for row in connection.execute("PRAGMA table_info(ingest_runs);").fetchall()
        }
        assert "embedding_tokens" in cols
        assert "llm_tokens" in cols
        assert "estimated_cost_usd" in cols
        assert "embedding_cache_hits" in cols
        assert "embedding_cache_misses" in cols
    finally:
        connection.close()
