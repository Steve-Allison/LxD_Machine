"""Legacy schema migrations for pre-versioning SQLite databases.

These migrations bridge databases created before the numbered migration
system (see :mod:`lxd.stores.schema`) into the shape that the current
baseline DDL and numbered migrations expect. They are idempotent: each
step checks for the legacy shape and returns early when the migration
has already been applied.

Invoked exclusively by :func:`lxd.stores.sqlite.initialize_schema`.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from pathlib import Path

from lxd.domain.ids import blake3_hex


def migrate_legacy_schema(connection: sqlite3.Connection) -> None:
    _migrate_legacy_corpus_manifest(connection)
    _migrate_legacy_chunk_rows(connection)
    _migrate_legacy_mention_rows(connection)
    _migrate_legacy_ontology_snapshot(connection)
    _migrate_legacy_ingest_runs(connection)
    _migrate_absolute_path_pks(connection)


def _migrate_legacy_corpus_manifest(connection: sqlite3.Connection) -> None:
    if not _table_exists(connection, "corpus_manifest"):
        return
    columns = _table_columns(connection, "corpus_manifest")
    if "blake3_hash" in columns:
        return

    legacy_rows = connection.execute(
        """
        SELECT
            source_rel_path,
            absolute_path,
            source_type,
            source_domain,
            file_size_bytes,
            content_hash,
            last_ingested_at
        FROM corpus_manifest
        ORDER BY source_rel_path
        """
    ).fetchall()
    chunk_counts = {
        str(row["source_rel_path"]): int(row["chunk_count"])
        for row in connection.execute(
            """
            SELECT source_rel_path, COUNT(*) AS chunk_count
            FROM chunk_rows
            GROUP BY source_rel_path
            """
        ).fetchall()
    }

    connection.execute("ALTER TABLE corpus_manifest RENAME TO corpus_manifest_legacy")
    connection.execute(
        """
        CREATE TABLE corpus_manifest (
            file_path TEXT PRIMARY KEY,
            file_rel_path TEXT NOT NULL,
            source_type TEXT NOT NULL,
            source_domain TEXT NOT NULL,
            document_id TEXT,
            blake3_hash TEXT NOT NULL,
            file_size_bytes INTEGER NOT NULL,
            parent_source_path TEXT,
            lifecycle_status TEXT NOT NULL,
            retrieval_status TEXT NOT NULL,
            chunk_count INTEGER NOT NULL DEFAULT 0,
            last_seen_at TEXT NOT NULL,
            last_processed_at TEXT,
            last_committed_at TEXT,
            error_message TEXT
        )
        """
    )
    connection.executemany(
        """
        INSERT INTO corpus_manifest (
            file_path,
            file_rel_path,
            source_type,
            source_domain,
            document_id,
            blake3_hash,
            file_size_bytes,
            parent_source_path,
            lifecycle_status,
            retrieval_status,
            chunk_count,
            last_seen_at,
            last_processed_at,
            last_committed_at,
            error_message
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                str(row["absolute_path"]),
                str(row["source_rel_path"]),
                str(row["source_type"]),
                str(row["source_domain"]),
                None
                if str(row["source_type"]) == "image_png"
                else blake3_hex(str(row["source_rel_path"])),
                str(row["content_hash"]),
                int(row["file_size_bytes"]),
                None,
                "complete",
                "asset_only" if str(row["source_type"]) == "image_png" else "searchable",
                chunk_counts.get(str(row["source_rel_path"]), 0),
                str(row["last_ingested_at"]),
                str(row["last_ingested_at"]),
                str(row["last_ingested_at"]),
                None,
            )
            for row in legacy_rows
        ],
    )
    connection.execute("DROP TABLE corpus_manifest_legacy")


def _migrate_legacy_chunk_rows(connection: sqlite3.Connection) -> None:
    if not _table_exists(connection, "chunk_rows"):
        return
    columns = _table_columns(connection, "chunk_rows")
    if "document_id" in columns and "metadata_json" in columns:
        return

    legacy_rows = connection.execute(
        """
        SELECT
            chunk_id,
            source_rel_path,
            source_type,
            source_domain,
            citation_label,
            chunk_index,
            text,
            chunk_hash,
            score_hint,
            vector_json,
            embedding_model,
            embedding_dims
        FROM chunk_rows
        ORDER BY source_rel_path, chunk_index
        """
    ).fetchall()
    manifest_rows = connection.execute(
        """
        SELECT
            file_rel_path,
            file_path,
            document_id,
            blake3_hash
        FROM corpus_manifest
        """
    ).fetchall()
    manifest_by_rel_path = {str(row["file_rel_path"]): row for row in manifest_rows}

    connection.execute("ALTER TABLE chunk_rows RENAME TO chunk_rows_legacy")
    connection.execute(
        """
        CREATE TABLE chunk_rows (
            chunk_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            source_rel_path TEXT NOT NULL,
            source_path TEXT NOT NULL,
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
            embedding_dims INTEGER NOT NULL,
            FOREIGN KEY(source_path) REFERENCES corpus_manifest(file_path) ON DELETE CASCADE
        )
        """
    )
    occurrence_by_document_hash: dict[tuple[str, str], int] = defaultdict(int)
    rows_to_insert: list[tuple[object, ...]] = []
    for row in legacy_rows:
        source_rel_path = str(row["source_rel_path"])
        manifest = manifest_by_rel_path.get(source_rel_path)
        source_path = str(manifest["file_path"]) if manifest is not None else source_rel_path
        document_id = (
            str(manifest["document_id"])
            if manifest is not None and manifest["document_id"] is not None
            else blake3_hex(source_rel_path)
        )
        source_hash = str(manifest["blake3_hash"]) if manifest is not None else ""
        chunk_hash = str(row["chunk_hash"])
        occurrence_key = (document_id, chunk_hash)
        chunk_occurrence = occurrence_by_document_hash[occurrence_key]
        occurrence_by_document_hash[occurrence_key] += 1
        text = str(row["text"])
        rows_to_insert.append(
            (
                str(row["chunk_id"]),
                document_id,
                source_rel_path,
                source_path,
                Path(source_rel_path).name,
                str(row["source_type"]),
                str(row["source_domain"]),
                source_hash,
                str(row["citation_label"]),
                int(row["chunk_index"]),
                chunk_occurrence,
                len(text.split()),
                text,
                chunk_hash,
                str(row["score_hint"]),
                "{}",
                str(row["vector_json"]),
                str(row["embedding_model"]),
                int(row["embedding_dims"]),
            )
        )
    connection.executemany(
        """
        INSERT INTO chunk_rows (
            chunk_id,
            document_id,
            source_rel_path,
            source_path,
            source_filename,
            source_type,
            source_domain,
            source_hash,
            citation_label,
            chunk_index,
            chunk_occurrence,
            token_count,
            text,
            chunk_hash,
            score_hint,
            metadata_json,
            vector_json,
            embedding_model,
            embedding_dims
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows_to_insert,
    )
    connection.execute("DROP TABLE chunk_rows_legacy")


def _migrate_legacy_mention_rows(connection: sqlite3.Connection) -> None:
    if not _table_exists(connection, "mention_rows"):
        return
    columns = _table_columns(connection, "mention_rows")
    required_columns = {"mention_id", "term_source", "source_domain"}
    if required_columns.issubset(columns):
        return

    legacy_rows = connection.execute(
        """
        SELECT
            chunk_id,
            entity_id,
            term_source,
            surface_form,
            start_char,
            end_char
        FROM mention_rows
        ORDER BY chunk_id, start_char, end_char
        """
    ).fetchall()
    chunk_rows = connection.execute(
        """
        SELECT
            chunk_id,
            source_domain,
            source_path,
            source_filename
        FROM chunk_rows
        """
    ).fetchall()
    chunk_by_id = {str(row["chunk_id"]): row for row in chunk_rows}

    connection.execute("ALTER TABLE mention_rows RENAME TO mention_rows_legacy")
    connection.execute(
        """
        CREATE TABLE mention_rows (
            mention_id TEXT PRIMARY KEY,
            entity_id TEXT NOT NULL,
            term_source TEXT NOT NULL,
            source_domain TEXT NOT NULL,
            source_path TEXT NOT NULL,
            source_filename TEXT NOT NULL,
            chunk_id TEXT NOT NULL,
            surface_form TEXT NOT NULL,
            start_char INTEGER NOT NULL,
            end_char INTEGER NOT NULL,
            FOREIGN KEY(chunk_id) REFERENCES chunk_rows(chunk_id) ON DELETE CASCADE,
            FOREIGN KEY(source_path) REFERENCES corpus_manifest(file_path) ON DELETE CASCADE
        )
        """
    )
    connection.executemany(
        """
        INSERT INTO mention_rows (
            mention_id,
            entity_id,
            term_source,
            source_domain,
            source_path,
            source_filename,
            chunk_id,
            surface_form,
            start_char,
            end_char
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                blake3_hex(str(row["entity_id"]), str(row["chunk_id"]), str(row["start_char"])),
                str(row["entity_id"]),
                str(row["term_source"]),
                str(chunk_by_id[str(row["chunk_id"])]["source_domain"]),
                str(chunk_by_id[str(row["chunk_id"])]["source_path"]),
                str(chunk_by_id[str(row["chunk_id"])]["source_filename"]),
                str(row["chunk_id"]),
                str(row["surface_form"]),
                int(row["start_char"]),
                int(row["end_char"]),
            )
            for row in legacy_rows
            if str(row["chunk_id"]) in chunk_by_id
        ],
    )
    connection.execute("DROP TABLE mention_rows_legacy")


def _migrate_legacy_ontology_snapshot(connection: sqlite3.Connection) -> None:
    if not _table_exists(connection, "ontology_snapshot"):
        return
    columns = _table_columns(connection, "ontology_snapshot")
    additions = {
        "coverage_path_count": "INTEGER NOT NULL DEFAULT 0",
        "graph_relation_count": "INTEGER NOT NULL DEFAULT 0",
        "validation_issue_count": "INTEGER NOT NULL DEFAULT 0",
        "validation_issues_json": "TEXT NOT NULL DEFAULT '[]'",
    }
    for column_name, column_sql in additions.items():
        if column_name in columns:
            continue
        connection.execute(f"ALTER TABLE ontology_snapshot ADD COLUMN {column_name} {column_sql}")


def _migrate_legacy_ingest_runs(connection: sqlite3.Connection) -> None:
    if not _table_exists(connection, "ingest_runs"):
        return
    columns = _table_columns(connection, "ingest_runs")
    if (
        "mode" in columns
        and "files_total" in columns
        and "notes" in columns
        and "searchable_files_rebuilt" in columns
        and "asset_files_processed" in columns
        and "unchanged_files_skipped" in columns
        and "failed_files" in columns
    ):
        return

    if "mode" in columns and "files_total" in columns and "notes" in columns:
        legacy_rows = connection.execute(
            """
            SELECT
                run_id,
                started_at,
                finished_at,
                mode,
                status,
                files_total,
                files_completed,
                searchable_files_completed,
                asset_files_completed,
                chunks_written,
                notes
            FROM ingest_runs
            ORDER BY started_at
            """
        ).fetchall()
        connection.execute("ALTER TABLE ingest_runs RENAME TO ingest_runs_legacy")
        connection.execute(
            """
            CREATE TABLE ingest_runs (
                run_id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                mode TEXT NOT NULL,
                status TEXT NOT NULL,
                files_total INTEGER NOT NULL,
                files_completed INTEGER NOT NULL,
                searchable_files_rebuilt INTEGER NOT NULL,
                asset_files_processed INTEGER NOT NULL,
                unchanged_files_skipped INTEGER NOT NULL,
                failed_files INTEGER NOT NULL,
                chunks_written INTEGER NOT NULL,
                notes TEXT NOT NULL
            )
            """
        )
        connection.executemany(
            """
            INSERT INTO ingest_runs (
                run_id,
                started_at,
                finished_at,
                mode,
                status,
                files_total,
                files_completed,
                searchable_files_rebuilt,
                asset_files_processed,
                unchanged_files_skipped,
                failed_files,
                chunks_written,
                notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    str(row["run_id"]),
                    str(row["started_at"]),
                    str(row["finished_at"]) if row["finished_at"] is not None else None,
                    str(row["mode"]),
                    str(row["status"]),
                    int(row["files_total"]),
                    int(row["files_completed"]),
                    int(row["searchable_files_completed"]),
                    int(row["asset_files_completed"]),
                    0,
                    0,
                    int(row["chunks_written"]),
                    str(row["notes"]),
                )
                for row in legacy_rows
            ],
        )
        connection.execute("DROP TABLE ingest_runs_legacy")
        return

    legacy_rows = connection.execute(
        """
        SELECT
            run_id,
            started_at,
            completed_at,
            status,
            corpus_file_count,
            text_file_count,
            asset_file_count,
            chunk_count,
            warning_json
        FROM ingest_runs
        ORDER BY started_at
        """
    ).fetchall()

    connection.execute("ALTER TABLE ingest_runs RENAME TO ingest_runs_legacy")
    connection.execute(
        """
        CREATE TABLE ingest_runs (
            run_id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            mode TEXT NOT NULL,
            status TEXT NOT NULL,
            files_total INTEGER NOT NULL,
            files_completed INTEGER NOT NULL,
            searchable_files_rebuilt INTEGER NOT NULL,
            asset_files_processed INTEGER NOT NULL,
            unchanged_files_skipped INTEGER NOT NULL,
            failed_files INTEGER NOT NULL,
            chunks_written INTEGER NOT NULL,
            notes TEXT NOT NULL
        )
        """
    )
    connection.executemany(
        """
        INSERT INTO ingest_runs (
            run_id,
            started_at,
            finished_at,
            mode,
            status,
            files_total,
            files_completed,
            searchable_files_rebuilt,
            asset_files_processed,
            unchanged_files_skipped,
            failed_files,
            chunks_written,
            notes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                str(row["run_id"]),
                str(row["started_at"]),
                str(row["completed_at"]) if row["completed_at"] is not None else None,
                "legacy",
                str(row["status"]),
                int(row["corpus_file_count"]),
                int(row["corpus_file_count"]),
                int(row["text_file_count"]),
                int(row["asset_file_count"]),
                0,
                0,
                int(row["chunk_count"]),
                str(row["warning_json"]),
            )
            for row in legacy_rows
        ],
    )
    connection.execute("DROP TABLE ingest_runs_legacy")


def _migrate_absolute_path_pks(connection: sqlite3.Connection) -> None:
    """Migrate from absolute-path PKs/FKs to relative-path PKs/FKs for portability.

    Idempotency / partial-state guard: in a previous version this function
    only checked ``corpus_manifest`` for the ``source_rel_path`` column. If a
    crash between the corpus_manifest rewrite and the chunk_rows rewrite ever
    left the DB half-migrated, the next run would treat the migration as
    "already applied" and skip the remaining tables. We now also refuse to
    proceed if any ``*_v2_legacy`` table is present (a clear marker that a
    prior run did not finish) — that's a hard stop with a useful message,
    not a silent skip.
    """
    if not _table_exists(connection, "corpus_manifest"):
        return

    leftover = _find_leftover_legacy_tables(connection)
    if leftover:
        raise sqlite3.DatabaseError(
            "Legacy migration partially applied; unexpected tables present: "
            f"{', '.join(leftover)}. Restore from a pre-migration backup or "
            "manually resolve the leftover tables before re-running ingest."
        )

    columns = _table_columns(connection, "corpus_manifest")
    if "source_rel_path" in columns and "absolute_path" in columns:
        return
    if "file_path" not in columns:
        return

    connection.execute("PRAGMA foreign_keys=OFF")

    # --- corpus_manifest: file_path PK → source_rel_path PK ---
    legacy_manifest = connection.execute(
        """
        SELECT file_path, file_rel_path, source_type, source_domain, document_id,
               blake3_hash, file_size_bytes, parent_source_path,
               lifecycle_status, retrieval_status, chunk_count,
               last_seen_at, last_processed_at, last_committed_at, error_message
        FROM corpus_manifest ORDER BY file_rel_path
        """
    ).fetchall()
    abs_to_rel: dict[str, str] = {
        str(row["file_path"]): str(row["file_rel_path"]) for row in legacy_manifest
    }
    connection.execute("ALTER TABLE corpus_manifest RENAME TO corpus_manifest_v2_legacy")
    connection.execute(
        """
        CREATE TABLE corpus_manifest (
            source_rel_path TEXT PRIMARY KEY,
            absolute_path TEXT NOT NULL,
            source_type TEXT NOT NULL,
            source_domain TEXT NOT NULL,
            document_id TEXT,
            blake3_hash TEXT NOT NULL,
            file_size_bytes INTEGER NOT NULL,
            parent_source_rel_path TEXT,
            lifecycle_status TEXT NOT NULL,
            retrieval_status TEXT NOT NULL,
            chunk_count INTEGER NOT NULL DEFAULT 0,
            last_seen_at TEXT NOT NULL,
            last_processed_at TEXT,
            last_committed_at TEXT,
            error_message TEXT
        )
        """
    )
    connection.executemany(
        """
        INSERT INTO corpus_manifest (
            source_rel_path, absolute_path, source_type, source_domain, document_id,
            blake3_hash, file_size_bytes, parent_source_rel_path,
            lifecycle_status, retrieval_status, chunk_count,
            last_seen_at, last_processed_at, last_committed_at, error_message
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                str(row["file_rel_path"]),
                str(row["file_path"]),
                str(row["source_type"]),
                str(row["source_domain"]),
                row["document_id"],
                str(row["blake3_hash"]),
                int(row["file_size_bytes"]),
                abs_to_rel.get(str(row["parent_source_path"]))
                if row["parent_source_path"]
                else None,
                str(row["lifecycle_status"]),
                str(row["retrieval_status"]),
                int(row["chunk_count"]),
                str(row["last_seen_at"]),
                row["last_processed_at"],
                row["last_committed_at"],
                row["error_message"],
            )
            for row in legacy_manifest
        ],
    )
    connection.execute("DROP TABLE corpus_manifest_v2_legacy")

    # --- chunk_rows: remove source_path, FK → source_rel_path ---
    chunk_columns = _table_columns(connection, "chunk_rows")
    if "source_path" in chunk_columns:
        legacy_chunks = connection.execute(
            """
            SELECT chunk_id, document_id, source_rel_path, source_filename,
                   source_type, source_domain, source_hash, citation_label,
                   chunk_index, chunk_occurrence, token_count, text, chunk_hash,
                   score_hint, metadata_json, vector_json, embedding_model, embedding_dims
            FROM chunk_rows ORDER BY source_rel_path, chunk_index
            """
        ).fetchall()
        connection.execute("ALTER TABLE chunk_rows RENAME TO chunk_rows_v2_legacy")
        connection.execute(
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
                embedding_dims INTEGER NOT NULL,
                FOREIGN KEY(source_rel_path) REFERENCES corpus_manifest(source_rel_path)
                    ON DELETE CASCADE
            )
            """
        )
        connection.executemany(
            """
            INSERT INTO chunk_rows (
                chunk_id, document_id, source_rel_path, source_filename,
                source_type, source_domain, source_hash, citation_label,
                chunk_index, chunk_occurrence, token_count, text, chunk_hash,
                score_hint, metadata_json, vector_json, embedding_model, embedding_dims
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    str(row["chunk_id"]),
                    str(row["document_id"]),
                    str(row["source_rel_path"]),
                    str(row["source_filename"]),
                    str(row["source_type"]),
                    str(row["source_domain"]),
                    str(row["source_hash"]),
                    str(row["citation_label"]),
                    int(row["chunk_index"]),
                    int(row["chunk_occurrence"]),
                    int(row["token_count"]),
                    str(row["text"]),
                    str(row["chunk_hash"]),
                    str(row["score_hint"]),
                    str(row["metadata_json"]),
                    str(row["vector_json"]),
                    str(row["embedding_model"]),
                    int(row["embedding_dims"]),
                )
                for row in legacy_chunks
            ],
        )
        connection.execute("DROP TABLE chunk_rows_v2_legacy")

    # --- mention_rows: source_path → source_rel_path ---
    mention_columns = _table_columns(connection, "mention_rows")
    if "source_path" in mention_columns and "source_rel_path" not in mention_columns:
        legacy_mentions = connection.execute(
            """
            SELECT mention_id, entity_id, term_source, source_domain,
                   source_path, source_filename, chunk_id, surface_form,
                   start_char, end_char
            FROM mention_rows
            """
        ).fetchall()
        connection.execute("ALTER TABLE mention_rows RENAME TO mention_rows_v2_legacy")
        connection.execute(
            """
            CREATE TABLE mention_rows (
                mention_id TEXT PRIMARY KEY,
                entity_id TEXT NOT NULL,
                term_source TEXT NOT NULL,
                source_domain TEXT NOT NULL,
                source_rel_path TEXT NOT NULL,
                source_filename TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                surface_form TEXT NOT NULL,
                start_char INTEGER NOT NULL,
                end_char INTEGER NOT NULL,
                FOREIGN KEY(chunk_id) REFERENCES chunk_rows(chunk_id) ON DELETE CASCADE,
                FOREIGN KEY(source_rel_path) REFERENCES corpus_manifest(source_rel_path)
                    ON DELETE CASCADE
            )
            """
        )
        connection.executemany(
            """
            INSERT INTO mention_rows (
                mention_id, entity_id, term_source, source_domain,
                source_rel_path, source_filename, chunk_id, surface_form,
                start_char, end_char
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    str(row["mention_id"]),
                    str(row["entity_id"]),
                    str(row["term_source"]),
                    str(row["source_domain"]),
                    abs_to_rel.get(str(row["source_path"]), str(row["source_path"])),
                    str(row["source_filename"]),
                    str(row["chunk_id"]),
                    str(row["surface_form"]),
                    int(row["start_char"]),
                    int(row["end_char"]),
                )
                for row in legacy_mentions
            ],
        )
        connection.execute("DROP TABLE mention_rows_v2_legacy")

    # --- asset_links: asset_path PK → asset_rel_path, parent_source_path → rel ---
    if _table_exists(connection, "asset_links"):
        asset_columns = _table_columns(connection, "asset_links")
        if "asset_path" in asset_columns and "parent_source_path" in asset_columns:
            legacy_assets = connection.execute(
                """
                SELECT asset_rel_path, asset_filename, source_domain,
                       parent_source_path, parent_document_id,
                       page_no, asset_index, link_method, blake3_hash, last_committed_at
                FROM asset_links
                """
            ).fetchall()
            connection.execute("ALTER TABLE asset_links RENAME TO asset_links_v2_legacy")
            connection.execute(
                """
                CREATE TABLE asset_links (
                    asset_rel_path TEXT PRIMARY KEY,
                    asset_filename TEXT NOT NULL,
                    source_domain TEXT NOT NULL,
                    parent_source_rel_path TEXT,
                    parent_document_id TEXT,
                    page_no INTEGER,
                    asset_index INTEGER,
                    link_method TEXT NOT NULL,
                    blake3_hash TEXT NOT NULL,
                    last_committed_at TEXT NOT NULL,
                    FOREIGN KEY(asset_rel_path) REFERENCES corpus_manifest(source_rel_path)
                        ON DELETE CASCADE
                )
                """
            )
            connection.executemany(
                """
                INSERT INTO asset_links (
                    asset_rel_path, asset_filename, source_domain,
                    parent_source_rel_path, parent_document_id,
                    page_no, asset_index, link_method, blake3_hash, last_committed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        str(row["asset_rel_path"]),
                        str(row["asset_filename"]),
                        str(row["source_domain"]),
                        abs_to_rel.get(str(row["parent_source_path"]))
                        if row["parent_source_path"]
                        else None,
                        row["parent_document_id"],
                        row["page_no"],
                        row["asset_index"],
                        str(row["link_method"]),
                        str(row["blake3_hash"]),
                        str(row["last_committed_at"]),
                    )
                    for row in legacy_assets
                ],
            )
            connection.execute("DROP TABLE asset_links_v2_legacy")

    # --- ontology_sources: file_path PK → file_rel_path PK ---
    if _table_exists(connection, "ontology_sources"):
        onto_columns = _table_columns(connection, "ontology_sources")
        if "file_path" in onto_columns:
            legacy_onto = connection.execute(
                "SELECT file_rel_path, blake3_hash, last_seen_at FROM ontology_sources"
            ).fetchall()
            connection.execute("ALTER TABLE ontology_sources RENAME TO ontology_sources_v2_legacy")
            connection.execute(
                """
                CREATE TABLE ontology_sources (
                    file_rel_path TEXT PRIMARY KEY,
                    blake3_hash TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL
                )
                """
            )
            connection.executemany(
                "INSERT INTO ontology_sources (file_rel_path, blake3_hash, last_seen_at)"
                " VALUES (?, ?, ?)",
                [
                    (str(row["file_rel_path"]), str(row["blake3_hash"]), str(row["last_seen_at"]))
                    for row in legacy_onto
                ],
            )
            connection.execute("DROP TABLE ontology_sources_v2_legacy")

    connection.execute("PRAGMA foreign_keys=ON")


def _table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    row = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _table_columns(connection: sqlite3.Connection, table_name: str) -> set[str]:
    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {str(row["name"]) for row in rows}


def _find_leftover_legacy_tables(connection: sqlite3.Connection) -> list[str]:
    """Return any ``*_v2_legacy`` tables left behind by an interrupted migration.

    These should never exist after a clean run. If they do, a previous
    migration crashed mid-way; the right answer is to surface that loudly
    rather than silently skip and let the user discover the half-state when
    a write fails downstream.
    """
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%_v2_legacy'"
    ).fetchall()
    return sorted(str(row["name"]) for row in rows)
