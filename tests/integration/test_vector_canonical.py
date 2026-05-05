"""Regression tests for Wave 3: LanceDB is the canonical chunk-vector store.

Covers:

1. Migration 0002 drops ``chunk_rows.vector_json`` on existing databases and
   re-running ``ensure_schema`` is a no-op.
2. ``load_chunk_records_for_source`` no longer surfaces vectors (empty list)
   and schema does not contain the ``vector_json`` column.
3. ``load_vectors_by_chunk_ids`` returns the vectors persisted to LanceDB.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lxd.stores.connection import open_store_connection
from lxd.stores.lancedb import (
    connect_lancedb,
    load_vectors_by_chunk_ids,
    open_chunk_table,
)
from lxd.stores.lancedb import (
    replace_source_chunks as replace_vector_source_chunks,
)
from lxd.stores.models import ChunkRecord, ManifestRecord
from lxd.stores.schema import CURRENT_SCHEMA_VERSION, ensure_schema, get_schema_version
from lxd.stores.sqlite.chunks import load_chunk_records_for_source, replace_source_chunks
from lxd.stores.sqlite.connection import build_store_paths, connect_sqlite
from lxd.stores.sqlite.manifest import upsert_manifest_record


def _chunk(chunk_id: str, *, source_rel_path: str, vector: list[float]) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        document_id="doc-1",
        source_rel_path=source_rel_path,
        source_filename=Path(source_rel_path).name,
        source_type="markdown",
        source_domain="guides",
        source_hash="hash-doc",
        citation_label=source_rel_path,
        chunk_index=0,
        chunk_occurrence=0,
        token_count=4,
        text="demo chunk",
        chunk_hash="chunk-hash",
        score_hint="neutral",
        metadata_json="{}",
        vector=vector,
        embedding_model="mxbai",
        embedding_dims=len(vector),
    )


def _manifest(source_rel_path: str) -> ManifestRecord:
    return ManifestRecord(
        source_rel_path=source_rel_path,
        absolute_path=f"/tmp/{source_rel_path}",
        source_type="markdown",
        source_domain="guides",
        document_id="doc-1",
        file_size_bytes=10,
        content_hash="hash-doc",
        parent_source_rel_path=None,
        chunk_count=1,
        last_seen_at="2026-03-27T00:00:00+00:00",
        last_processed_at="2026-03-27T00:00:00+00:00",
        last_committed_at="2026-03-27T00:00:00+00:00",
        error_message=None,
    )


def test_migration_drops_vector_json_on_legacy_database(tmp_path: Path) -> None:
    """Pre-v2 databases with ``vector_json`` must have the column dropped."""
    store_paths = build_store_paths(tmp_path)
    connection = connect_sqlite(store_paths.sqlite_path)
    try:
        connection.executescript(
            """
            CREATE TABLE chunk_rows (
                chunk_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                source_rel_path TEXT NOT NULL,
                source_filename TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_domain TEXT NOT NULL,
                source_hash TEXT NOT NULL,
                citation_label TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                chunk_occurrence INTEGER NOT NULL,
                token_count INTEGER NOT NULL,
                text TEXT NOT NULL,
                chunk_hash TEXT NOT NULL,
                score_hint TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                vector_json TEXT NOT NULL,
                embedding_model TEXT NOT NULL,
                embedding_dims INTEGER NOT NULL
            );
            """
        )
        connection.execute("PRAGMA user_version = 1;")
        connection.commit()

        ensure_schema(connection)

        columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(chunk_rows);").fetchall()
        }
        assert "vector_json" not in columns
        assert get_schema_version(connection) == CURRENT_SCHEMA_VERSION

        ensure_schema(connection)
        assert get_schema_version(connection) == CURRENT_SCHEMA_VERSION
    finally:
        connection.close()


def test_vector_round_trip_via_lancedb(tmp_path: Path) -> None:
    """Vectors live in LanceDB; SQLite returns them empty and hydration works."""
    store_paths = build_store_paths(tmp_path)
    source_rel_path = "Guides/vec.md"
    chunk = _chunk("chunk-1", source_rel_path=source_rel_path, vector=[0.11, 0.22, 0.33])
    manifest = _manifest(source_rel_path)

    with open_store_connection(store_paths.sqlite_path) as connection:
        upsert_manifest_record(connection, manifest)
        replace_source_chunks(
            connection,
            source_rel_path=source_rel_path,
            chunk_records=[chunk],
            mention_records=[],
            relation_records=[],
        )

        columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(chunk_rows);").fetchall()
        }
        assert "vector_json" not in columns

        loaded = load_chunk_records_for_source(connection, source_rel_path)
        assert len(loaded) == 1
        assert loaded[0].vector == []

    db = connect_lancedb(store_paths.lancedb_path)
    table = open_chunk_table(db, vector_size=len(chunk.vector))
    replace_vector_source_chunks(table, source_rel_path, [chunk])
    vectors_by_id = load_vectors_by_chunk_ids(table, ["chunk-1", "missing"])
    assert vectors_by_id["chunk-1"] == pytest.approx([0.11, 0.22, 0.33])
    assert "missing" not in vectors_by_id
