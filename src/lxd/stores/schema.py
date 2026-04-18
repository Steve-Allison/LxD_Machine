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
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable

from lxd.stores._base_ddl import BASE_SCHEMA_DDL

Migration = Callable[[sqlite3.Connection], None]

CURRENT_SCHEMA_VERSION = 3


def ensure_schema(connection: sqlite3.Connection) -> None:
    """Ensure ``connection`` is at the current schema version.

    Creates missing tables/indexes, runs any unapplied migrations in ascending
    order, and stamps ``PRAGMA user_version`` on success. Idempotent: calling
    it on an up-to-date database is a cheap no-op (a single pragma read).

    Args:
        connection: Open SQLite connection. ``PRAGMA foreign_keys`` and the
            other connection-scoped pragmas configured in
            :func:`lxd.stores.sqlite.connect_sqlite` must already be applied.

    Side Effects:
        Executes DDL inside a transaction, and bumps ``user_version`` to
        :data:`CURRENT_SCHEMA_VERSION` after all pending migrations succeed.

    Raises:
        sqlite3.DatabaseError: Propagated from DDL or migration steps.
    """
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


def get_schema_version(connection: sqlite3.Connection) -> int:
    """Return the current on-disk schema version from ``PRAGMA user_version``.

    Args:
        connection: Open SQLite connection.

    Returns:
        Non-negative integer stored in ``PRAGMA user_version``. Fresh databases
        return ``0``.
    """
    return _read_user_version(connection)


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


def _migration_0001_baseline(connection: sqlite3.Connection) -> None:
    """Baseline migration: promote existing store to versioned layout.

    This is a no-op in terms of DDL because the baseline tables are created
    by :data:`BASE_SCHEMA_DDL` above; the migration exists so that legacy
    databases (which report ``user_version = 0``) are explicitly stamped
    with version 1 after first contact.
    """
    del connection


def _migration_0002_drop_chunk_vector_json(connection: sqlite3.Connection) -> None:
    """Drop the legacy ``chunk_rows.vector_json`` column.

    LanceDB is the canonical store for chunk embedding vectors; maintaining a
    parallel JSON-encoded copy in SQLite wastes disk, slows writes, and
    allows the two stores to drift. This migration drops the column on any
    existing database where it still exists. Requires SQLite >= 3.35.
    """
    row = connection.execute("PRAGMA table_info(chunk_rows);").fetchall()
    columns = {str(info[1]) for info in row} if row else set()
    if "vector_json" in columns:
        connection.execute("ALTER TABLE chunk_rows DROP COLUMN vector_json;")


def _migration_0003_llm_jobs(connection: sqlite3.Connection) -> None:
    """Introduce a persistent LLM job queue.

    Creates ``llm_jobs``, an idempotent work ledger used by the relation
    extraction and claims pipelines for OpenAI batch submissions, Ollama
    long-running calls, and any future structured-output workload. The
    columns are intentionally minimal so downstream tools can attach their
    own JSON payloads without requiring further migrations.

    Columns:
        job_id: ULID/blake3-derived identifier supplied by the caller.
        kind: Logical job category (e.g. ``claims.openai_batch``).
        corpus_id: Tenancy marker; defaults to ``'default'`` to preserve
            single-tenant behaviour.
        status: One of ``queued|running|succeeded|failed|cancelled``.
        payload_json: JSON blob owned by the producer; opaque to the queue.
        result_json: JSON blob or NULL until the job completes.
        error: Short human-readable failure message.
        attempts: Retry counter maintained by the executor.
        created_at / updated_at: ISO-8601 UTC timestamps.

    The ``(corpus_id, status, updated_at)`` index supports efficient
    "next queued job" lookups per tenant without full table scans.
    """
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


_MIGRATIONS: dict[int, Migration] = {
    1: _migration_0001_baseline,
    2: _migration_0002_drop_chunk_vector_json,
    3: _migration_0003_llm_jobs,
}
