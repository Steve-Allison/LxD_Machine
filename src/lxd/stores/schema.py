"""Own SQLite schema DDL, migrations, and versioned upgrades.

Responsibility:
    Define the authoritative set of SQLite tables, indexes, and numbered
    migrations that advance on-disk state from any historic shape to the
    current one. All schema evolution goes through :func:`ensure_schema`.

Design boundary:
    Callers should treat SQLite schema as opaque and only interact via this
    module's ``ensure_schema`` entrypoint and the high-level CRUD helpers in
    ``lxd.stores.sqlite``. The schema version is tracked with SQLite's
    built-in ``PRAGMA user_version`` so we never need a bespoke metadata row.

Key constraints:
    * Migrations are pure SQL/Python functions keyed by *target* version; they
      run in ascending order and inside a transaction.
    * ``ensure_schema`` is idempotent and safe to call from any process that
      opens the database.
    * Legacy migrations (pre-versioning) remain in ``sqlite.py`` for now and
      are invoked from version 0 -> 1, but new changes must land here as
      numbered migrations.
    * Every call to ``ensure_schema`` finishes with an integrity verification
      pass — ``PRAGMA foreign_key_check`` plus a tables/columns presence
      check — so a half-migrated DB cannot proceed silently into ingest.
"""

import shutil
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from lxd.stores._base_ddl import BASE_SCHEMA_DDL

Migration = Callable[[sqlite3.Connection], None]

CURRENT_SCHEMA_VERSION: Final = 9


class SchemaIntegrityError(sqlite3.DatabaseError):
    """Raised when post-migration verification finds a broken schema.

    This is a hard stop: the DB is not safe to write to. Surface to the user
    immediately rather than letting downstream writes fail mid-batch and burn
    upstream API spend.
    """


def ensure_schema(connection: sqlite3.Connection) -> None:
    """Ensure ``connection`` is at the current schema version.

    Creates missing tables/indexes, runs any unapplied migrations in ascending
    order, stamps ``PRAGMA user_version``, then verifies the resulting schema
    is internally consistent (FKs reference real tables, expected tables and
    columns are present).

    Args:
        connection: Open SQLite connection. ``PRAGMA foreign_keys`` and the
            other connection-scoped pragmas configured in
            :func:`lxd.stores.sqlite.connect_sqlite` must already be applied.

    Side Effects:
        Executes DDL inside a transaction, bumps ``user_version`` to
        :data:`CURRENT_SCHEMA_VERSION` after all pending migrations succeed,
        and may write a timestamped backup of the SQLite file when migrations
        actually run.

    Raises:
        SchemaIntegrityError: Post-migration verification failed (ghost FK,
            missing table, missing column).
        sqlite3.DatabaseError: Propagated from DDL or migration steps.
    """
    current = _read_user_version(connection)
    pending = [target for target in sorted(_MIGRATIONS) if target > current]
    if pending:
        _backup_database_for_migration(connection, from_version=current, to_version=pending[-1])
    with connection:
        connection.executescript(BASE_SCHEMA_DDL)
        current = _read_user_version(connection)
        for target in sorted(_MIGRATIONS):
            if target <= current:
                continue
            _MIGRATIONS[target](connection)
            _write_user_version(connection, target)
            current = target
        if current < CURRENT_SCHEMA_VERSION:
            _write_user_version(connection, CURRENT_SCHEMA_VERSION)
    _verify_schema_integrity(connection)


def get_schema_version(connection: sqlite3.Connection) -> int:
    """Return the current on-disk schema version from ``PRAGMA user_version``.

    Args:
        connection: Open SQLite connection.

    Returns:
        Non-negative integer stored in ``PRAGMA user_version``. Fresh databases
        return ``0``.
    """
    return _read_user_version(connection)


def verify_schema_integrity(connection: sqlite3.Connection) -> None:
    """Public hook for callers that want an integrity check without migrating.

    Used by the preflight command and tests.
    """
    _verify_schema_integrity(connection)


def _read_user_version(connection: sqlite3.Connection) -> int:
    row = connection.execute("PRAGMA user_version;").fetchone()
    if row is None:
        return 0
    value = row[0] if not isinstance(row, sqlite3.Row) else row["user_version"]
    return int(value or 0)


def _write_user_version(connection: sqlite3.Connection, version: int) -> None:
    if version < 0:
        raise ValueError("schema version must be non-negative")
    connection.execute(f"PRAGMA user_version = {int(version)};")


def _backup_database_for_migration(
    connection: sqlite3.Connection,
    *,
    from_version: int,
    to_version: int,
) -> None:
    """Snapshot the SQLite file before destructive migrations run.

    On-disk databases **must** be backed up successfully before migrations
    proceed — a failed ``copy2`` raises :class:`SchemaIntegrityError` so we
    never run destructive DDL without a restore point. In-memory / unnamed
    databases (no file path) skip the backup; there is nothing to snapshot.

    The backup lives next to the database with a timestamp suffix.
    """
    try:
        db_path_row = connection.execute("PRAGMA database_list;").fetchone()
    except sqlite3.DatabaseError as exc:
        raise SchemaIntegrityError(
            f"Refusing migration v{from_version}→v{to_version}: "
            f"could not resolve database path for backup ({exc})."
        ) from exc
    if db_path_row is None:
        raise SchemaIntegrityError(
            f"Refusing migration v{from_version}→v{to_version}: "
            "PRAGMA database_list returned no rows."
        )
    file_path_str = (
        db_path_row[2] if not isinstance(db_path_row, sqlite3.Row) else db_path_row["file"]
    )
    if not file_path_str:
        # :memory: / temporary connections — nothing durable to back up.
        return
    db_path = Path(file_path_str)
    if not db_path.exists():
        raise SchemaIntegrityError(
            f"Refusing migration v{from_version}→v{to_version}: "
            f"database file does not exist at {db_path}."
        )
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    backup_path = db_path.with_suffix(
        f".pre-migration-v{from_version}-to-v{to_version}-{timestamp}.sqlite3.bak"
    )
    try:
        shutil.copy2(db_path, backup_path)
    except OSError as exc:
        raise SchemaIntegrityError(
            f"Refusing migration v{from_version}→v{to_version}: "
            f"could not write backup to {backup_path} ({exc})."
        ) from exc


def _migration_0001_baseline(connection: sqlite3.Connection) -> None:
    """Baseline migration: promote existing store to versioned layout."""
    del connection


def _migration_0002_drop_chunk_vector_json(connection: sqlite3.Connection) -> None:
    """Drop the legacy ``chunk_rows.vector_json`` column."""
    row = connection.execute("PRAGMA table_info(chunk_rows);").fetchall()
    columns = {str(info[1]) for info in row} if row else set()
    if "vector_json" in columns:
        connection.execute("ALTER TABLE chunk_rows DROP COLUMN vector_json;")


def _migration_0003_llm_jobs(connection: sqlite3.Connection) -> None:
    """Introduce a persistent LLM job queue."""
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS llm_jobs (
            job_id       TEXT PRIMARY KEY,
            kind         TEXT NOT NULL,
            corpus_id    TEXT NOT NULL DEFAULT 'default',
            status       TEXT NOT NULL
                         CHECK (status IN ('queued','running','succeeded','failed','cancelled')),
            payload_json TEXT NOT NULL,
            result_json  TEXT,
            error        TEXT,
            attempts     INTEGER NOT NULL DEFAULT 0,
            created_at   TEXT NOT NULL,
            updated_at   TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_llm_jobs_status
            ON llm_jobs(corpus_id, status, updated_at);
        """
    )


def _migration_0004_repair_ghost_fks(connection: sqlite3.Connection) -> None:
    """Repair FK references that point at the ghost ``chunk_rows_v2_legacy`` table.

    Background: migration ``_migrate_absolute_path_pks`` (the legacy ad-hoc
    migration that ran before numbered migrations were introduced) renamed
    ``chunk_rows`` to ``chunk_rows_v2_legacy`` while three child tables held
    FK references to ``chunk_rows``. SQLite, with the default
    ``PRAGMA legacy_alter_table=OFF``, rewrote those FK references to point
    at the renamed table — and that pointer never got rewritten back when
    ``chunk_rows_v2_legacy`` was dropped. Every child-table cascade now fails
    with ``no such table: main.chunk_rows_v2_legacy``.

    The fix:
        For each affected child table — ``extracted_relations``, ``claims``,
        ``relation_evidence`` — inspect ``sqlite_master.sql``. If it still
        references the ghost table, recreate the table with the correct DDL
        and copy the data through. Tables that do not reference the ghost
        (e.g. fresh DBs created from ``BASE_SCHEMA_DDL``) are left alone.

    Idempotent: each child table is checked individually; the migration is a
    no-op when no ghost references remain.
    """
    affected = ("extracted_relations", "claims", "relation_evidence")
    repairs_needed = []
    for table in affected:
        row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name = ?",
            (table,),
        ).fetchone()
        if row is None:
            continue
        sql = row[0] if not isinstance(row, sqlite3.Row) else row["sql"]
        if sql is None:
            continue
        if "chunk_rows_v2_legacy" in sql:
            repairs_needed.append(table)

    if not repairs_needed:
        return

    connection.execute("PRAGMA foreign_keys=OFF;")
    try:
        if "extracted_relations" in repairs_needed:
            _rebuild_extracted_relations(connection)
        if "claims" in repairs_needed:
            _rebuild_claims(connection)
        if "relation_evidence" in repairs_needed:
            _rebuild_relation_evidence(connection)
    finally:
        connection.execute("PRAGMA foreign_keys=ON;")


def _rebuild_extracted_relations(connection: sqlite3.Connection) -> None:
    connection.execute(
        "ALTER TABLE extracted_relations RENAME TO _extracted_relations_fk_repair_tmp;"
    )
    connection.executescript(
        """
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
            FOREIGN KEY(chunk_id) REFERENCES chunk_rows(chunk_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_extracted_relations_subject
            ON extracted_relations(subject_entity_id);
        CREATE INDEX IF NOT EXISTS idx_extracted_relations_object
            ON extracted_relations(object_entity_id);

        INSERT INTO extracted_relations (
            relation_id, chunk_id, document_id, source_rel_path,
            subject_entity_id, predicate, object_entity_id,
            confidence, extraction_model, extracted_at
        )
        SELECT
            relation_id, chunk_id, document_id, source_rel_path,
            subject_entity_id, predicate, object_entity_id,
            confidence, extraction_model, extracted_at
        FROM _extracted_relations_fk_repair_tmp;

        DROP TABLE _extracted_relations_fk_repair_tmp;
        """
    )


def _rebuild_claims(connection: sqlite3.Connection) -> None:
    connection.execute("ALTER TABLE claims RENAME TO _claims_fk_repair_tmp;")
    connection.executescript(
        """
        CREATE TABLE claims (
            claim_id TEXT PRIMARY KEY,
            chunk_id TEXT NOT NULL,
            document_id TEXT NOT NULL,
            source_rel_path TEXT NOT NULL,
            claim_text TEXT NOT NULL,
            subject_entity_id TEXT,
            object_entity_id TEXT,
            claim_type TEXT NOT NULL DEFAULT 'assertion',
            confidence REAL NOT NULL,
            extraction_model TEXT NOT NULL,
            extracted_at TEXT NOT NULL,
            FOREIGN KEY(chunk_id) REFERENCES chunk_rows(chunk_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_claims_subject ON claims(subject_entity_id);
        CREATE INDEX IF NOT EXISTS idx_claims_object  ON claims(object_entity_id);
        CREATE INDEX IF NOT EXISTS idx_claims_chunk   ON claims(chunk_id);
        CREATE INDEX IF NOT EXISTS idx_claims_document ON claims(document_id);

        INSERT INTO claims (
            claim_id, chunk_id, document_id, source_rel_path, claim_text,
            subject_entity_id, object_entity_id, claim_type, confidence,
            extraction_model, extracted_at
        )
        SELECT
            claim_id, chunk_id, document_id, source_rel_path, claim_text,
            subject_entity_id, object_entity_id, claim_type, confidence,
            extraction_model, extracted_at
        FROM _claims_fk_repair_tmp;

        DROP TABLE _claims_fk_repair_tmp;
        """
    )


def _rebuild_relation_evidence(connection: sqlite3.Connection) -> None:
    connection.execute("ALTER TABLE relation_evidence RENAME TO _relation_evidence_fk_repair_tmp;")
    connection.executescript(
        """
        CREATE TABLE relation_evidence (
            evidence_id TEXT PRIMARY KEY,
            relation_id TEXT NOT NULL,
            chunk_id TEXT NOT NULL,
            surface_subject TEXT NOT NULL,
            surface_object TEXT NOT NULL,
            evidence_text TEXT NOT NULL,
            confidence REAL NOT NULL,
            extraction_model TEXT NOT NULL,
            extracted_at TEXT NOT NULL,
            FOREIGN KEY(relation_id) REFERENCES relations(relation_id) ON DELETE CASCADE,
            FOREIGN KEY(chunk_id) REFERENCES chunk_rows(chunk_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_relation_evidence_relation
            ON relation_evidence(relation_id);
        CREATE INDEX IF NOT EXISTS idx_relation_evidence_chunk
            ON relation_evidence(chunk_id);

        INSERT INTO relation_evidence (
            evidence_id, relation_id, chunk_id, surface_subject, surface_object,
            evidence_text, confidence, extraction_model, extracted_at
        )
        SELECT
            evidence_id, relation_id, chunk_id, surface_subject, surface_object,
            evidence_text, confidence, extraction_model, extracted_at
        FROM _relation_evidence_fk_repair_tmp;

        DROP TABLE _relation_evidence_fk_repair_tmp;
        """
    )


def _migration_0005_ingest_run_telemetry(connection: sqlite3.Connection) -> None:
    """Add cost / token telemetry columns to ``ingest_runs``.

    Allows post-hoc cost analysis without re-running ingest. Columns are
    nullable so historic rows remain valid; ingest writes them when known.
    """
    row = connection.execute("PRAGMA table_info(ingest_runs);").fetchall()
    columns = {str(info[1]) for info in row} if row else set()
    if "embedding_tokens" not in columns:
        connection.execute("ALTER TABLE ingest_runs ADD COLUMN embedding_tokens INTEGER;")
    if "llm_tokens" not in columns:
        connection.execute("ALTER TABLE ingest_runs ADD COLUMN llm_tokens INTEGER;")
    if "estimated_cost_usd" not in columns:
        connection.execute("ALTER TABLE ingest_runs ADD COLUMN estimated_cost_usd REAL;")
    if "embedding_cache_hits" not in columns:
        connection.execute("ALTER TABLE ingest_runs ADD COLUMN embedding_cache_hits INTEGER;")
    if "embedding_cache_misses" not in columns:
        connection.execute("ALTER TABLE ingest_runs ADD COLUMN embedding_cache_misses INTEGER;")


def _migration_0006_chunk_rows_wiki_metadata(connection: sqlite3.Connection) -> None:
    """Add page-level wiki metadata columns to ``chunk_rows``.

    Captures the Sources line and ``[[slug]]`` cross-references parsed from
    the wiki frontmatter so retrieval can surface citations and traverse the
    page graph without re-parsing files. JSON arrays of strings; '[]' for
    historic rows that pre-date the wiki swap.
    """
    row = connection.execute("PRAGMA table_info(chunk_rows);").fetchall()
    columns = {str(info[1]) for info in row} if row else set()
    if "cited_sources_json" not in columns:
        connection.execute(
            "ALTER TABLE chunk_rows ADD COLUMN cited_sources_json TEXT NOT NULL DEFAULT '[]';"
        )
    if "wiki_links_json" not in columns:
        connection.execute(
            "ALTER TABLE chunk_rows ADD COLUMN wiki_links_json TEXT NOT NULL DEFAULT '[]';"
        )


def _migration_0007_circuit_breaker_state(connection: sqlite3.Connection) -> None:
    """Create the persistent circuit-breaker state table.

    The in-memory ``SystemicErrorCircuitBreaker`` reset its counter on
    every process start: a crashed run mid-trip would re-spend on the
    same systemic failure pattern. The persistent breaker reads its
    state from this table on construct, so a crashed pid that already
    saw 2 consecutive systemic failures resumes at 2 — a single new
    failure trips the breaker rather than starting fresh from zero.
    """
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS circuit_breaker_state (
            scope TEXT PRIMARY KEY,
            consecutive_failures INTEGER NOT NULL DEFAULT 0,
            last_error_class TEXT,
            last_error_message TEXT,
            last_error_type TEXT,
            last_failure_at TEXT,
            last_success_at TEXT,
            tripped_at TEXT
        );
        """
    )


def _migration_0008_hierarchical_communities(connection: sqlite3.Connection) -> None:
    """Migrate community tables to composite PK on ``(community_id, community_level)``.

    The pre-v8 schema had a single ``community_id PRIMARY KEY`` on
    ``community_reports`` and a single ``entity_id PRIMARY KEY`` on
    ``entity_communities``, which made hierarchical (multi-level) clustering
    impossible — the same numeric community_id could not exist at multiple
    levels, and a single entity could not belong to communities at multiple
    resolutions.

    Adds:
      - composite PK ``(community_id, community_level)`` on
        ``community_reports`` plus a ``parent_community_id`` column to
        anchor the hierarchy.
      - composite PK ``(entity_id, community_level)`` on
        ``entity_communities`` so each entity holds one assignment per level.
      - new indexes for level-scoped queries.

    Destructive: drops existing community rows. The knowledge graph build
    is rebuildable end-to-end via ``pixi run build-graph --full`` so this is
    recoverable; the migration framework's auto-backup gives the operator a
    rollback target.
    """
    connection.executescript(
        """
        DROP TABLE IF EXISTS entity_communities;
        DROP TABLE IF EXISTS community_reports;

        CREATE TABLE entity_communities (
            entity_id TEXT NOT NULL,
            community_id INTEGER NOT NULL,
            community_level INTEGER NOT NULL DEFAULT 0,
            modularity_class TEXT,
            assigned_at TEXT NOT NULL,
            PRIMARY KEY (entity_id, community_level)
        );
        CREATE INDEX idx_entity_communities_community_level
            ON entity_communities(community_id, community_level);
        CREATE INDEX idx_entity_communities_level
            ON entity_communities(community_level);

        CREATE TABLE community_reports (
            community_id INTEGER NOT NULL,
            community_level INTEGER NOT NULL DEFAULT 0,
            parent_community_id INTEGER,
            member_count INTEGER NOT NULL,
            member_entity_ids_json TEXT NOT NULL,
            deterministic_summary TEXT NOT NULL,
            llm_summary TEXT,
            top_entities_json TEXT NOT NULL DEFAULT '[]',
            top_claims_json TEXT NOT NULL DEFAULT '[]',
            intra_community_edge_count INTEGER NOT NULL DEFAULT 0,
            source_hash TEXT NOT NULL,
            generated_at TEXT NOT NULL,
            PRIMARY KEY (community_id, community_level)
        );
        CREATE INDEX idx_community_reports_level
            ON community_reports(community_level);
        CREATE INDEX idx_community_reports_parent
            ON community_reports(parent_community_id, community_level);
        """
    )


def _migration_0009_entity_embedding_state(connection: sqlite3.Connection) -> None:
    """Create ``entity_embedding_state`` for incremental entity-embedding skip.

    Before this migration ``_compute_entity_embeddings`` unconditionally
    dropped and rebuilt the LanceDB entity table on every ``build-graph``
    run, re-mean-pooling every qualifying entity's chunk vectors regardless
    of whether anything had actually changed. This table holds the per-entity
    source_hash (sorted chunk_ids + embedding model identity); a matching
    hash lets the compute step skip the entity entirely.

    Idempotent CREATE IF NOT EXISTS — composes cleanly with the base DDL on
    fresh databases.
    """
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS entity_embedding_state (
            entity_id TEXT PRIMARY KEY,
            source_hash TEXT NOT NULL,
            chunk_count INTEGER NOT NULL,
            embedding_model TEXT NOT NULL,
            embedding_dims INTEGER NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
    )


_MIGRATIONS: dict[int, Migration] = {
    1: _migration_0001_baseline,
    2: _migration_0002_drop_chunk_vector_json,
    3: _migration_0003_llm_jobs,
    4: _migration_0004_repair_ghost_fks,
    5: _migration_0005_ingest_run_telemetry,
    6: _migration_0006_chunk_rows_wiki_metadata,
    7: _migration_0007_circuit_breaker_state,
    8: _migration_0008_hierarchical_communities,
    9: _migration_0009_entity_embedding_state,
}


# ---------------------------------------------------------------------------
# Post-migration integrity verification
# ---------------------------------------------------------------------------


# Tables and the columns we hard-depend on at runtime. Keep this list tight —
# every entry is a runtime contract, not exhaustive coverage of the schema.
_REQUIRED_COLUMNS: dict[str, frozenset[str]] = {
    "corpus_manifest": frozenset({"source_rel_path", "blake3_hash", "lifecycle_status"}),
    "chunk_rows": frozenset(
        {
            "chunk_id",
            "source_rel_path",
            "chunk_hash",
            "embedding_model",
            "embedding_dims",
            "cited_sources_json",
            "wiki_links_json",
        }
    ),
    "mention_rows": frozenset({"mention_id", "chunk_id", "entity_id"}),
    "extracted_relations": frozenset({"relation_id", "chunk_id"}),
    "claims": frozenset({"claim_id", "chunk_id"}),
    "relation_evidence": frozenset({"evidence_id", "relation_id", "chunk_id"}),
    "ingest_runs": frozenset({"run_id", "status"}),
    "llm_jobs": frozenset({"job_id", "status"}),
    "circuit_breaker_state": frozenset({"scope", "consecutive_failures"}),
}


def _verify_schema_integrity(connection: sqlite3.Connection) -> None:
    """Hard stop if migrations left the DB in a half-state.

    Runs three checks:
    1. ``PRAGMA foreign_key_check`` — surfaces orphaned FK rows AND ghost FK
       references. A ghost FK (table renamed/dropped under a child) shows up
       as a row in this pragma's output even when no data is orphaned, because
       SQLite tries to look up the parent.
    2. Required tables are present.
    3. Required columns are present in each required table.

    Any violation raises :class:`SchemaIntegrityError`. The error message
    enumerates everything found wrong so the operator sees the full picture
    on one screen.
    """
    issues: list[str] = []

    fk_rows = connection.execute("PRAGMA foreign_key_check;").fetchall()
    if fk_rows:
        seen_fk: set[str] = set()
        for fk_row in fk_rows:
            child = fk_row[0] if not isinstance(fk_row, sqlite3.Row) else fk_row["table"]
            parent = fk_row[2] if not isinstance(fk_row, sqlite3.Row) else fk_row["parent"]
            label = f"{child} -> {parent}"
            if label in seen_fk:
                continue
            seen_fk.add(label)
            issues.append(f"foreign_key_check violation: {label}")

    table_rows = connection.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
    existing_tables = {
        (row[0] if not isinstance(row, sqlite3.Row) else row["name"]) for row in table_rows
    }
    for table in _REQUIRED_COLUMNS:
        if table not in existing_tables:
            issues.append(f"missing required table: {table}")

    for table, required in _REQUIRED_COLUMNS.items():
        if table not in existing_tables:
            continue
        info = connection.execute(f"PRAGMA table_info({table});").fetchall()
        cols = {str(row[1]) for row in info}
        missing = required - cols
        if missing:
            issues.append(
                f"table {table} is missing required column(s): {', '.join(sorted(missing))}"
            )

    if issues:
        raise SchemaIntegrityError(
            "Schema integrity check failed after migrations:\n  - "
            + "\n  - ".join(issues)
            + "\nThis is a hard stop — refuse to write until repaired."
        )
