"""Connect / build-paths / initialize-schema / reset for the SQLite store."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from lxd.stores.models import StorePaths
from lxd.stores.schema import ensure_schema

_SQLITE_FILENAME = "lxd.sqlite3"


def connect_sqlite(path: Path) -> sqlite3.Connection:
    """Open SQLite storage and apply connection settings.

    Applies tuned PRAGMAs on every fresh connection:

    - ``journal_mode=WAL`` — concurrent readers alongside a single writer.
    - ``synchronous=NORMAL`` — safe under WAL, meaningfully faster than FULL.
    - ``foreign_keys=ON`` — enforce declared FKs (off by default in sqlite3).
    - ``busy_timeout=5000`` — tolerate brief lock contention in place of
      immediate ``database is locked`` errors.
    - ``temp_store=MEMORY`` — keep transient B-trees in RAM, not tempfiles.
    - ``cache_size=-65536`` — ~64 MiB per-connection page cache.

    Args:
        path: Path to the source file or storage location.

    Returns:
        Configured SQLite connection.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL;")
    connection.execute("PRAGMA synchronous=NORMAL;")
    connection.execute("PRAGMA foreign_keys=ON;")
    connection.execute("PRAGMA busy_timeout=5000;")
    connection.execute("PRAGMA temp_store=MEMORY;")
    connection.execute("PRAGMA cache_size=-65536;")
    return connection


def build_store_paths(data_path: Path) -> StorePaths:
    """Resolve SQLite and LanceDB paths under the data directory."""
    return StorePaths(sqlite_path=data_path / _SQLITE_FILENAME, lancedb_path=data_path / "lancedb")


def assert_no_v2_legacy_tables(connection: sqlite3.Connection) -> None:
    """Refuse to proceed if any ``*_v2_legacy`` table is present.

    These tables only exist as the smoking gun of a half-finished pre-v0
    migration. The numbered migration system in :mod:`lxd.stores.schema`
    cannot reason about them and downstream writes will fail mid-batch if
    we silently continue. Raising here surfaces the corruption loudly.
    """
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%_v2_legacy'"
    ).fetchall()
    leftover = sorted(str(row["name"]) for row in rows)
    if leftover:
        raise sqlite3.DatabaseError(
            "Legacy migration partially applied; unexpected tables present: "
            f"{', '.join(leftover)}. Restore from a pre-migration backup or "
            "manually resolve the leftover tables before re-running ingest."
        )


def initialize_schema(connection: sqlite3.Connection) -> None:
    """Create and migrate required SQLite tables.

    Order of operations:

    1. Refuse to proceed if any ``*_v2_legacy`` table is present (smoking
       gun of an interrupted pre-v0 migration).
    2. Delegate to :func:`lxd.stores.schema.ensure_schema` to create missing
       baseline tables and run pending numbered migrations with
       ``PRAGMA user_version`` stamped on success.
    3. Ensure runtime indexes that may live outside the DDL.
    """
    with connection:
        assert_no_v2_legacy_tables(connection)
    ensure_schema(connection)
    with connection:
        _ensure_indexes(connection)


def reset_store(connection: sqlite3.Connection) -> None:
    """Delete persisted ingest data across managed tables."""
    with connection:
        connection.execute("DELETE FROM asset_links")
        connection.execute("DELETE FROM ontology_sources")
        connection.execute("DELETE FROM ontology_snapshot")
        connection.execute("DELETE FROM ingest_config")
        connection.execute("DELETE FROM extracted_relations")
        connection.execute("DELETE FROM mention_rows")
        connection.execute("DELETE FROM chunk_rows")
        connection.execute("DELETE FROM corpus_manifest")


def _ensure_indexes(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE INDEX IF NOT EXISTS idx_corpus_manifest_blake3_hash
        ON corpus_manifest(blake3_hash);

        CREATE INDEX IF NOT EXISTS idx_corpus_manifest_document_id
        ON corpus_manifest(document_id);

        CREATE INDEX IF NOT EXISTS idx_chunk_rows_source_rel_path
        ON chunk_rows(source_rel_path);

        CREATE INDEX IF NOT EXISTS idx_chunk_rows_document_id
        ON chunk_rows(document_id);

        CREATE INDEX IF NOT EXISTS idx_chunk_rows_source_domain
        ON chunk_rows(source_domain);
        """
    )
