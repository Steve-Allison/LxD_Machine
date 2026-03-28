"""Persist ingest state, manifests, and ontology snapshots in SQLite."""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

from lxd.domain.ids import blake3_hex
from lxd.stores.models import (
    AssetLinkRecord,
    CanonicalRelationRecord,
    ChunkRecord,
    ClaimRecord,
    CommunityReportRecord,
    CorpusStatusSummary,
    EntityCommunityRecord,
    EntityMentionResult,
    EntityProfileRecord,
    ExtractedRelationRecord,
    GraphBuildStateRecord,
    IngestConfigSnapshotRecord,
    ManifestRecord,
    MentionRecord,
    OntologySnapshotRecord,
    OntologySourceRecord,
    RelationEvidenceRecord,
    StorePaths,
)

_SQLITE_FILENAME = "lxd.sqlite3"


def connect_sqlite(path: Path) -> sqlite3.Connection:
    """Open SQLite storage and apply connection settings.

    Args:
        path: Path to the source file or storage location.

    Returns:
        Configured SQLite connection.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL;")
    connection.execute("PRAGMA foreign_keys=ON;")
    return connection


def build_store_paths(data_path: Path) -> StorePaths:
    """Resolve SQLite and LanceDB paths under the data directory.

    Args:
        data_path: Root data directory for local stores.

    Returns:
        Resolved SQLite and LanceDB store paths.
    """
    return StorePaths(sqlite_path=data_path / _SQLITE_FILENAME, lancedb_path=data_path / "lancedb")


def initialize_schema(connection: sqlite3.Connection) -> None:
    """Create and migrate required SQLite tables.

    Args:
        connection: Open SQLite connection.
    """
    with connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS corpus_manifest (
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
            );

            CREATE TABLE IF NOT EXISTS chunk_rows (
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
            );

            CREATE TABLE IF NOT EXISTS asset_links (
                asset_path TEXT PRIMARY KEY,
                asset_rel_path TEXT NOT NULL,
                asset_filename TEXT NOT NULL,
                source_domain TEXT NOT NULL,
                parent_source_path TEXT,
                parent_document_id TEXT,
                page_no INTEGER,
                asset_index INTEGER,
                link_method TEXT NOT NULL,
                blake3_hash TEXT NOT NULL,
                last_committed_at TEXT NOT NULL,
                FOREIGN KEY(asset_path) REFERENCES corpus_manifest(file_path) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS mention_rows (
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
            );

            CREATE INDEX IF NOT EXISTS idx_mention_rows_entity_id
            ON mention_rows(entity_id);

            CREATE TABLE IF NOT EXISTS ontology_sources (
                file_path TEXT PRIMARY KEY,
                file_rel_path TEXT NOT NULL,
                blake3_hash TEXT NOT NULL,
                last_seen_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS ontology_snapshot (
                snapshot_id TEXT PRIMARY KEY CHECK (snapshot_id = 'current'),
                ontology_root TEXT NOT NULL,
                blake3_hash TEXT NOT NULL,
                matcher_termset_hash TEXT NOT NULL,
                matcher_term_count INTEGER NOT NULL,
                source_file_count INTEGER NOT NULL,
                entity_file_count INTEGER NOT NULL,
                entity_count INTEGER NOT NULL,
                coverage_path_count INTEGER NOT NULL DEFAULT 0,
                graph_relation_count INTEGER NOT NULL DEFAULT 0,
                validation_issue_count INTEGER NOT NULL DEFAULT 0,
                validation_issues_json TEXT NOT NULL DEFAULT '[]',
                last_loaded_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS extracted_relations (
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

            CREATE TABLE IF NOT EXISTS ingest_config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS ingest_runs (
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
            );

            -- Knowledge Graph tables (Phase 5)

            CREATE TABLE IF NOT EXISTS claims (
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
            CREATE INDEX IF NOT EXISTS idx_claims_object ON claims(object_entity_id);
            CREATE INDEX IF NOT EXISTS idx_claims_chunk ON claims(chunk_id);
            CREATE INDEX IF NOT EXISTS idx_claims_document ON claims(document_id);

            CREATE TABLE IF NOT EXISTS entity_profiles (
                entity_id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                domain TEXT NOT NULL DEFAULT '',
                aliases_json TEXT NOT NULL DEFAULT '[]',
                deterministic_summary TEXT NOT NULL,
                llm_summary TEXT,
                chunk_count INTEGER NOT NULL,
                doc_count INTEGER NOT NULL,
                mention_count INTEGER NOT NULL,
                claim_count INTEGER NOT NULL DEFAULT 0,
                top_predicates_json TEXT NOT NULL DEFAULT '[]',
                top_claims_json TEXT NOT NULL DEFAULT '[]',
                pagerank REAL NOT NULL DEFAULT 0.0,
                betweenness REAL NOT NULL DEFAULT 0.0,
                closeness REAL NOT NULL DEFAULT 0.0,
                in_degree INTEGER NOT NULL DEFAULT 0,
                out_degree INTEGER NOT NULL DEFAULT 0,
                eigenvector REAL NOT NULL DEFAULT 0.0,
                community_id INTEGER,
                source_hash TEXT NOT NULL,
                generated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS entity_communities (
                entity_id TEXT PRIMARY KEY,
                community_id INTEGER NOT NULL,
                community_level INTEGER NOT NULL DEFAULT 0,
                modularity_class TEXT,
                assigned_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_entity_communities_community_id
            ON entity_communities(community_id);

            CREATE TABLE IF NOT EXISTS community_reports (
                community_id INTEGER PRIMARY KEY,
                community_level INTEGER NOT NULL DEFAULT 0,
                member_count INTEGER NOT NULL,
                member_entity_ids_json TEXT NOT NULL,
                deterministic_summary TEXT NOT NULL,
                llm_summary TEXT,
                top_entities_json TEXT NOT NULL DEFAULT '[]',
                top_claims_json TEXT NOT NULL DEFAULT '[]',
                intra_community_edge_count INTEGER NOT NULL DEFAULT 0,
                source_hash TEXT NOT NULL,
                generated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS relations (
                relation_id TEXT PRIMARY KEY,
                subject_entity_id TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object_entity_id TEXT NOT NULL,
                support_count INTEGER NOT NULL DEFAULT 0,
                avg_confidence REAL NOT NULL DEFAULT 0.0,
                min_confidence REAL NOT NULL DEFAULT 0.0,
                max_confidence REAL NOT NULL DEFAULT 0.0,
                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_relations_spo
            ON relations(subject_entity_id, predicate, object_entity_id);
            CREATE INDEX IF NOT EXISTS idx_relations_subject ON relations(subject_entity_id);
            CREATE INDEX IF NOT EXISTS idx_relations_object ON relations(object_entity_id);

            CREATE TABLE IF NOT EXISTS relation_evidence (
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

            CREATE TABLE IF NOT EXISTS graph_build_state (
                run_id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                status TEXT NOT NULL,
                current_phase TEXT NOT NULL DEFAULT 'pending',
                graph_version INTEGER NOT NULL,
                relations_consolidated INTEGER NOT NULL DEFAULT 0,
                evidence_rows_built INTEGER NOT NULL DEFAULT 0,
                claims_extracted INTEGER NOT NULL DEFAULT 0,
                entity_profiles_built INTEGER NOT NULL DEFAULT 0,
                communities_detected INTEGER NOT NULL DEFAULT 0,
                community_reports_built INTEGER NOT NULL DEFAULT 0,
                centrality_computed INTEGER NOT NULL DEFAULT 0,
                entity_embeddings_computed INTEGER NOT NULL DEFAULT 0,
                llm_enrichment_count INTEGER NOT NULL DEFAULT 0,
                notes_json TEXT NOT NULL DEFAULT '[]'
            );

            CREATE TABLE IF NOT EXISTS graph_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        _migrate_legacy_schema(connection)
        _ensure_indexes(connection)


def reset_store(connection: sqlite3.Connection) -> None:
    """Delete persisted ingest data across managed tables.

    Args:
        connection: Open SQLite connection.
    """
    with connection:
        connection.execute("DELETE FROM asset_links")
        connection.execute("DELETE FROM ontology_sources")
        connection.execute("DELETE FROM ontology_snapshot")
        connection.execute("DELETE FROM ingest_config")
        connection.execute("DELETE FROM extracted_relations")
        connection.execute("DELETE FROM mention_rows")
        connection.execute("DELETE FROM chunk_rows")
        connection.execute("DELETE FROM corpus_manifest")


def begin_ingest_run(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    started_at: str,
    mode: str,
    files_total: int,
) -> None:
    """Insert the initial ingest run row.

    Args:
        connection: Open SQLite connection.
        run_id: Ingest run identifier.
        started_at: UTC timestamp when the run started.
        mode: Ingest mode label (for example, full or incremental).
        files_total: Number of files planned for this run.
    """
    with connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO ingest_runs (
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
            VALUES (?, ?, NULL, ?, 'running', ?, 0, 0, 0, 0, 0, 0, '[]')
            """,
            (run_id, started_at, mode, files_total),
        )


def finish_ingest_run(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    finished_at: str,
    status: str,
    files_completed: int,
    searchable_files_rebuilt: int,
    asset_files_processed: int,
    unchanged_files_skipped: int,
    failed_files: int,
    chunks_written: int,
    notes: list[str],
) -> None:
    """Finalize ingest run status and counters.

    Args:
        connection: Open SQLite connection.
        run_id: Ingest run identifier.
        finished_at: UTC timestamp when the run finished.
        status: Final ingest run status label.
        files_completed: Number of files processed so far.
        searchable_files_rebuilt: Count of searchable sources rebuilt.
        asset_files_processed: Count of asset-only files processed.
        unchanged_files_skipped: Count of unchanged files skipped.
        failed_files: Count of files that failed processing.
        chunks_written: Count of chunks written in this run.
        notes: Progress or warning notes to persist.
    """
    with connection:
        connection.execute(
            """
            UPDATE ingest_runs
            SET finished_at = ?,
                status = ?,
                files_completed = ?,
                searchable_files_rebuilt = ?,
                asset_files_processed = ?,
                unchanged_files_skipped = ?,
                failed_files = ?,
                chunks_written = ?,
                notes = ?
            WHERE run_id = ?
            """,
            (
                finished_at,
                status,
                files_completed,
                searchable_files_rebuilt,
                asset_files_processed,
                unchanged_files_skipped,
                failed_files,
                chunks_written,
                json.dumps(notes, separators=(",", ":")),
                run_id,
            ),
        )


def update_ingest_run_progress(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    files_completed: int,
    searchable_files_rebuilt: int,
    asset_files_processed: int,
    unchanged_files_skipped: int,
    failed_files: int,
    chunks_written: int,
    notes: list[str],
) -> None:
    """Persist incremental ingest progress counters.

    Args:
        connection: Open SQLite connection.
        run_id: Ingest run identifier.
        files_completed: Number of files processed so far.
        searchable_files_rebuilt: Count of searchable sources rebuilt.
        asset_files_processed: Count of asset-only files processed.
        unchanged_files_skipped: Count of unchanged files skipped.
        failed_files: Count of files that failed processing.
        chunks_written: Count of chunks written in this run.
        notes: Progress or warning notes to persist.
    """
    with connection:
        connection.execute(
            """
            UPDATE ingest_runs
            SET files_completed = ?,
                searchable_files_rebuilt = ?,
                asset_files_processed = ?,
                unchanged_files_skipped = ?,
                failed_files = ?,
                chunks_written = ?,
                notes = ?
            WHERE run_id = ?
            """,
            (
                files_completed,
                searchable_files_rebuilt,
                asset_files_processed,
                unchanged_files_skipped,
                failed_files,
                chunks_written,
                json.dumps(notes, separators=(",", ":")),
                run_id,
            ),
        )


def load_manifest_index(connection: sqlite3.Connection) -> dict[str, ManifestRecord]:
    """Load manifest records keyed by relative path.

    Args:
        connection: Open SQLite connection.

    Returns:
        Manifest records keyed by relative source path.
    """
    rows = connection.execute(
        """
        SELECT
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
        FROM corpus_manifest
        """
    ).fetchall()
    return {record.source_rel_path: record for record in (_manifest_from_row(row) for row in rows)}


def load_manifest_by_content_hash(
    connection: sqlite3.Connection,
) -> dict[str, list[ManifestRecord]]:
    """Load manifest records grouped by content hash.

    Args:
        connection: Open SQLite connection.

    Returns:
        Manifest records grouped by content hash.
    """
    rows = connection.execute(
        """
        SELECT
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
        FROM corpus_manifest
        ORDER BY file_rel_path
        """
    ).fetchall()
    grouped: dict[str, list[ManifestRecord]] = defaultdict(list)
    for row in rows:
        record = _manifest_from_row(row)
        grouped[record.content_hash].append(record)
    return dict(grouped)


def load_manifest_by_absolute_path(
    connection: sqlite3.Connection, absolute_path: str
) -> ManifestRecord | None:
    """Load one manifest record by absolute path.

    Args:
        connection: Open SQLite connection.
        absolute_path: Absolute source file path.

    Returns:
        Matching manifest record, if present.
    """
    row = connection.execute(
        """
        SELECT
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
        FROM corpus_manifest
        WHERE file_path = ?
        """,
        (absolute_path,),
    ).fetchone()
    if row is None:
        return None
    return _manifest_from_row(row)


def upsert_manifest_record(connection: sqlite3.Connection, record: ManifestRecord) -> None:
    """Insert or update a corpus manifest record.

    Args:
        connection: Open SQLite connection.
        record: Record instance to persist.
    """
    with connection:
        connection.execute(
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
            ON CONFLICT(file_path) DO UPDATE SET
                file_rel_path = excluded.file_rel_path,
                source_type = excluded.source_type,
                source_domain = excluded.source_domain,
                document_id = excluded.document_id,
                blake3_hash = excluded.blake3_hash,
                file_size_bytes = excluded.file_size_bytes,
                parent_source_path = excluded.parent_source_path,
                lifecycle_status = excluded.lifecycle_status,
                retrieval_status = excluded.retrieval_status,
                chunk_count = excluded.chunk_count,
                last_seen_at = excluded.last_seen_at,
                last_processed_at = excluded.last_processed_at,
                last_committed_at = excluded.last_committed_at,
                error_message = excluded.error_message
            """,
            (
                record.absolute_path,
                record.source_rel_path,
                record.source_type,
                record.source_domain,
                record.document_id,
                record.content_hash,
                record.file_size_bytes,
                record.parent_source_path,
                record.lifecycle_status,
                record.retrieval_status,
                record.chunk_count,
                record.last_seen_at,
                record.last_processed_at,
                record.last_committed_at,
                record.error_message,
            ),
        )


def upsert_asset_link(
    connection: sqlite3.Connection, absolute_asset_path: str, record: AssetLinkRecord
) -> None:
    """Insert or update an asset-to-parent linkage record.

    Args:
        connection: Open SQLite connection.
        absolute_asset_path: Absolute asset file path.
        record: Record instance to persist.
    """
    with connection:
        connection.execute(
            """
            INSERT INTO asset_links (
                asset_path,
                asset_rel_path,
                asset_filename,
                source_domain,
                parent_source_path,
                parent_document_id,
                page_no,
                asset_index,
                link_method,
                blake3_hash,
                last_committed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(asset_path) DO UPDATE SET
                asset_rel_path = excluded.asset_rel_path,
                asset_filename = excluded.asset_filename,
                source_domain = excluded.source_domain,
                parent_source_path = excluded.parent_source_path,
                parent_document_id = excluded.parent_document_id,
                page_no = excluded.page_no,
                asset_index = excluded.asset_index,
                link_method = excluded.link_method,
                blake3_hash = excluded.blake3_hash,
                last_committed_at = excluded.last_committed_at
            """,
            (
                absolute_asset_path,
                record.asset_rel_path,
                record.asset_filename,
                record.source_domain,
                record.parent_source_path,
                record.parent_document_id,
                record.page_no,
                record.asset_index,
                record.link_method,
                record.blake3_hash,
                record.last_committed_at,
            ),
        )


def replace_ontology_sources(
    connection: sqlite3.Connection, records: list[OntologySourceRecord]
) -> None:
    """Replace persisted ontology source records.

    Args:
        connection: Open SQLite connection.
        records: Records to replace in the target table.
    """
    with connection:
        connection.execute("DELETE FROM ontology_sources")
        if records:
            connection.executemany(
                """
                INSERT INTO ontology_sources (file_path, file_rel_path, blake3_hash, last_seen_at)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        record.file_path,
                        record.file_rel_path,
                        record.blake3_hash,
                        record.last_seen_at,
                    )
                    for record in records
                ],
            )


def replace_ontology_snapshot(
    connection: sqlite3.Connection, record: OntologySnapshotRecord
) -> None:
    """Replace the persisted ontology snapshot row.

    Args:
        connection: Open SQLite connection.
        record: Record instance to persist.
    """
    with connection:
        connection.execute("DELETE FROM ontology_snapshot")
        connection.execute(
            """
            INSERT INTO ontology_snapshot (
                snapshot_id,
                ontology_root,
                blake3_hash,
                matcher_termset_hash,
                matcher_term_count,
                source_file_count,
                entity_file_count,
                entity_count,
                coverage_path_count,
                graph_relation_count,
                validation_issue_count,
                validation_issues_json,
                last_loaded_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.snapshot_id,
                record.ontology_root,
                record.snapshot_hash,
                record.matcher_termset_hash,
                record.matcher_term_count,
                record.source_file_count,
                record.entity_file_count,
                record.entity_count,
                record.coverage_path_count,
                record.graph_relation_count,
                record.validation_issue_count,
                record.validation_issues_json,
                record.last_loaded_at,
            ),
        )


def replace_ingest_config_snapshot(
    connection: sqlite3.Connection, records: list[IngestConfigSnapshotRecord]
) -> None:
    """Replace persisted ingest config key-value rows.

    Args:
        connection: Open SQLite connection.
        records: Records to replace in the target table.
    """
    with connection:
        connection.execute("DELETE FROM ingest_config")
        if records:
            connection.executemany(
                "INSERT INTO ingest_config (key, value) VALUES (?, ?)",
                [(record.key, record.value) for record in records],
            )


def list_allowed_domains(connection: sqlite3.Connection) -> set[str]:
    """List available source domains from committed manifest rows.

    Args:
        connection: Open SQLite connection.

    Returns:
        Distinct searchable source domains.
    """
    rows = connection.execute(
        """
        SELECT DISTINCT source_domain
        FROM corpus_manifest
        WHERE lifecycle_status != 'deleted'
        ORDER BY source_domain
        """
    ).fetchall()
    return {str(row["source_domain"]) for row in rows if row["source_domain"] is not None}


def load_ingest_config_snapshot(connection: sqlite3.Connection) -> dict[str, str]:
    """Load persisted ingest config key-value rows.

    Args:
        connection: Open SQLite connection.

    Returns:
        Persisted ingest config key-value mapping.
    """
    rows = connection.execute("SELECT key, value FROM ingest_config ORDER BY key").fetchall()
    return {str(row["key"]): str(row["value"]) for row in rows}


def load_ontology_snapshot(connection: sqlite3.Connection) -> OntologySnapshotRecord | None:
    """Load the persisted ontology snapshot record.

    Args:
        connection: Open SQLite connection.

    Returns:
        Persisted ontology snapshot, if available.
    """
    row = connection.execute(
        """
        SELECT
            snapshot_id,
            ontology_root,
            blake3_hash,
            matcher_termset_hash,
            matcher_term_count,
            source_file_count,
            entity_file_count,
            entity_count,
            coverage_path_count,
            graph_relation_count,
            validation_issue_count,
            validation_issues_json,
            last_loaded_at
        FROM ontology_snapshot
        WHERE snapshot_id = 'current'
        """
    ).fetchone()
    if row is None:
        return None
    return OntologySnapshotRecord(
        snapshot_id=str(row["snapshot_id"]),
        ontology_root=str(row["ontology_root"]),
        snapshot_hash=str(row["blake3_hash"]),
        matcher_termset_hash=str(row["matcher_termset_hash"]),
        matcher_term_count=int(row["matcher_term_count"]),
        source_file_count=int(row["source_file_count"]),
        entity_file_count=int(row["entity_file_count"]),
        entity_count=int(row["entity_count"]),
        coverage_path_count=int(row["coverage_path_count"]),
        graph_relation_count=int(row["graph_relation_count"]),
        validation_issue_count=int(row["validation_issue_count"]),
        validation_issues_json=str(row["validation_issues_json"]),
        last_loaded_at=str(row["last_loaded_at"]),
    )


def store_has_committed_state(connection: sqlite3.Connection) -> bool:
    """Return whether committed corpus state exists.

    Args:
        connection: Open SQLite connection.

    Returns:
        `True` when committed searchable corpus state exists.
    """
    ontology_snapshot = load_ontology_snapshot(connection)
    if ontology_snapshot is not None:
        return True
    config_snapshot = load_ingest_config_snapshot(connection)
    if config_snapshot:
        return True
    manifest_row = connection.execute(
        "SELECT COUNT(*) AS count FROM corpus_manifest WHERE lifecycle_status != 'deleted'"
    ).fetchone()
    if int(_row_value(manifest_row, "count")) > 0:
        return True
    chunk_row = connection.execute("SELECT COUNT(*) AS count FROM chunk_rows").fetchone()
    if int(_row_value(chunk_row, "count")) > 0:
        return True
    mention_row = connection.execute("SELECT COUNT(*) AS count FROM mention_rows").fetchone()
    return int(_row_value(mention_row, "count")) > 0


def delete_source(connection: sqlite3.Connection, absolute_path: str) -> None:
    """Apply the requested persistence operation.

    Args:
        connection: Open SQLite connection.
        absolute_path: Absolute source file path.
    """
    with connection:
        connection.execute(
            """
            UPDATE corpus_manifest
            SET lifecycle_status = 'deleted',
                retrieval_status = 'not_searchable',
                chunk_count = 0
            WHERE file_path = ?
            """,
            (absolute_path,),
        )
        connection.execute("DELETE FROM chunk_rows WHERE source_path = ?", (absolute_path,))
        connection.execute("DELETE FROM asset_links WHERE asset_path = ?", (absolute_path,))


def replace_source_chunks(
    connection: sqlite3.Connection,
    *,
    absolute_source_path: str,
    chunk_records: list[ChunkRecord],
    mention_records: list[MentionRecord],
    relation_records: list[ExtractedRelationRecord] | None = None,
) -> None:
    """Replace all vector chunks for one source path.

    Args:
        connection: Open SQLite connection.
        absolute_source_path: Absolute source file path for chunk data.
        chunk_records: Chunk rows to persist for a source.
        mention_records: Mention rows to persist for the source.
        relation_records: Extracted relation rows to persist for the source.
    """
    with connection:
        connection.execute("DELETE FROM chunk_rows WHERE source_path = ?", (absolute_source_path,))
        connection.execute(
            """
            DELETE FROM mention_rows
            WHERE source_path = ?
            """,
            (absolute_source_path,),
        )
        if chunk_records:
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
                [
                    (
                        record.chunk_id,
                        record.document_id,
                        record.source_rel_path,
                        record.source_path,
                        record.source_filename,
                        record.source_type,
                        record.source_domain,
                        record.source_hash,
                        record.citation_label,
                        record.chunk_index,
                        record.chunk_occurrence,
                        record.token_count,
                        record.text,
                        record.chunk_hash,
                        record.score_hint,
                        record.metadata_json,
                        json.dumps(record.vector, separators=(",", ":")),
                        record.embedding_model,
                        record.embedding_dims,
                    )
                    for record in chunk_records
                ],
            )
        if mention_records:
            source_rel_path = chunk_records[0].source_rel_path if chunk_records else ""
            source_domain = chunk_records[0].source_domain if chunk_records else ""
            source_filename = (
                Path(source_rel_path).name if source_rel_path else Path(absolute_source_path).name
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
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        _mention_id(record),
                        record.entity_id,
                        record.term_source,
                        source_domain,
                        absolute_source_path,
                        source_filename,
                        record.chunk_id,
                        record.surface_form,
                        record.start_char,
                        record.end_char,
                    )
                    for record in mention_records
                ],
            )
        if relation_records:
            connection.executemany(
                """
                INSERT OR IGNORE INTO extracted_relations (
                    relation_id,
                    chunk_id,
                    document_id,
                    source_rel_path,
                    subject_entity_id,
                    predicate,
                    object_entity_id,
                    confidence,
                    extraction_model,
                    extracted_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        record.relation_id,
                        record.chunk_id,
                        record.document_id,
                        record.source_rel_path,
                        record.subject_entity_id,
                        record.predicate,
                        record.object_entity_id,
                        record.confidence,
                        record.extraction_model,
                        record.extracted_at,
                    )
                    for record in relation_records
                ],
            )


def load_chunk_records_for_source(
    connection: sqlite3.Connection, absolute_source_path: str
) -> list[ChunkRecord]:
    """Load persisted chunk records for a source path.

    Args:
        connection: Open SQLite connection.
        absolute_source_path: Absolute source file path for chunk data.

    Returns:
        Chunk records for the source path.
    """
    rows = connection.execute(
        """
        SELECT
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
        FROM chunk_rows
        WHERE source_path = ?
        ORDER BY chunk_index
        """,
        (absolute_source_path,),
    ).fetchall()
    return [_chunk_from_row(row) for row in rows]


def load_mentions_for_source(
    connection: sqlite3.Connection, absolute_source_path: str
) -> dict[str, list[MentionRecord]]:
    """Load persisted mentions grouped by chunk ID for a source.

    Args:
        connection: Open SQLite connection.
        absolute_source_path: Absolute source file path for chunk data.

    Returns:
        Mentions grouped by chunk ID for the source.
    """
    rows = connection.execute(
        """
        SELECT
            chunk_id,
            entity_id,
            term_source,
            surface_form,
            start_char,
            end_char
        FROM mention_rows
        WHERE source_path = ?
        ORDER BY chunk_id, start_char, end_char, entity_id
        """,
        (absolute_source_path,),
    ).fetchall()
    grouped: dict[str, list[MentionRecord]] = defaultdict(list)
    for row in rows:
        record = MentionRecord(
            chunk_id=str(row["chunk_id"]),
            entity_id=str(row["entity_id"]),
            term_source=str(row["term_source"]),
            surface_form=str(row["surface_form"]),
            start_char=int(row["start_char"]),
            end_char=int(row["end_char"]),
        )
        grouped[record.chunk_id].append(record)
    return dict(grouped)


def find_chunks_by_entity_mentions(
    connection: sqlite3.Connection,
    entity_ids: list[str],
    *,
    limit: int = 50,
) -> list[EntityMentionResult]:
    """Find chunks matching one or more entity mentions.

    Args:
        connection: Open SQLite connection.
        entity_ids: Entity identifiers used for relation-aware search.
        limit: Maximum number of records to return.

    Returns:
        Top chunk matches with entity mention counts.
    """
    if not entity_ids:
        return []
    placeholders = ",".join("?" * len(entity_ids))
    rows = connection.execute(
        f"""
        WITH matched AS (
            SELECT chunk_id, COUNT(DISTINCT entity_id) AS entity_match_count
            FROM mention_rows
            WHERE entity_id IN ({placeholders})
            GROUP BY chunk_id
        )
        SELECT
            c.chunk_id,
            c.document_id,
            c.source_rel_path,
            c.citation_label,
            c.chunk_index,
            c.text,
            c.score_hint,
            c.metadata_json,
            m.entity_match_count
        FROM chunk_rows c
        JOIN matched m ON c.chunk_id = m.chunk_id
        ORDER BY m.entity_match_count DESC, c.chunk_index ASC
        LIMIT ?
        """,
        (*entity_ids, limit * 4),
    ).fetchall()
    total = len(entity_ids)
    seen_sources: set[str] = set()
    results: list[EntityMentionResult] = []
    for row in rows:
        source_rel_path = str(row["source_rel_path"])
        if source_rel_path in seen_sources:
            continue
        seen_sources.add(source_rel_path)
        results.append(
            EntityMentionResult(
                chunk_id=str(row["chunk_id"]),
                document_id=str(row["document_id"]),
                source_rel_path=source_rel_path,
                citation_label=str(row["citation_label"]),
                chunk_index=int(row["chunk_index"]),
                text=str(row["text"]),
                score_hint=str(row["score_hint"]),
                metadata_json=str(row["metadata_json"]),
                entity_match_count=int(row["entity_match_count"]),
                total_entity_ids=total,
            )
        )
        if len(results) >= limit:
            break
    return results


def load_corpus_related_entity_ids(
    connection: sqlite3.Connection,
    entity_ids: list[str],
    *,
    min_confidence: float = 0.5,
    max_results: int = 20,
) -> list[str]:
    """Return entity IDs strongly related to `entity_ids` via extracted corpus relations.

    Returns the *other* end of any relation where one of `entity_ids` appears as subject
    or object, filtered by confidence and de-duplicated. Used to augment query expansion
    with corpus-derived rather than ontology-derived edges.
    """
    if not entity_ids:
        return []
    placeholders = ",".join("?" * len(entity_ids))
    rows = connection.execute(
        f"""
        SELECT subject_entity_id, object_entity_id, confidence
        FROM extracted_relations
        WHERE (subject_entity_id IN ({placeholders}) OR object_entity_id IN ({placeholders}))
          AND confidence >= ?
        ORDER BY confidence DESC
        LIMIT ?
        """,
        [*entity_ids, *entity_ids, min_confidence, max_results * 4],
    ).fetchall()
    seed_set = set(entity_ids)
    related: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for candidate in (str(row["subject_entity_id"]), str(row["object_entity_id"])):
            if candidate not in seed_set and candidate not in seen:
                seen.add(candidate)
                related.append(candidate)
                if len(related) >= max_results:
                    return related
    return related


def load_corpus_relations_for_entity(
    connection: sqlite3.Connection,
    entity_id: str,
    *,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Return extracted corpus relations where `entity_id` appears as subject or object."""
    rows = connection.execute(
        """
        SELECT subject_entity_id, predicate, object_entity_id, confidence,
               extraction_model, source_rel_path, chunk_id
        FROM extracted_relations
        WHERE subject_entity_id = ? OR object_entity_id = ?
        ORDER BY confidence DESC
        LIMIT ?
        """,
        (entity_id, entity_id, limit),
    ).fetchall()
    return [
        {
            "subject": str(row["subject_entity_id"]),
            "predicate": str(row["predicate"]),
            "object": str(row["object_entity_id"]),
            "confidence": float(row["confidence"]),
            "source_rel_path": str(row["source_rel_path"]),
            "chunk_id": str(row["chunk_id"]),
        }
        for row in rows
    ]


def load_relation_chunk_ids(
    connection: sqlite3.Connection,
    entity_ids: list[str],
) -> set[str]:
    """Return chunk IDs that contain an extracted relation involving any of `entity_ids`."""
    if not entity_ids:
        return set()
    placeholders = ",".join("?" * len(entity_ids))
    rows = connection.execute(
        f"""
        SELECT DISTINCT chunk_id
        FROM extracted_relations
        WHERE subject_entity_id IN ({placeholders})
           OR object_entity_id IN ({placeholders})
        """,
        [*entity_ids, *entity_ids],
    ).fetchall()
    return {str(row["chunk_id"]) for row in rows}


def summarize_store(
    connection: sqlite3.Connection,
    *,
    ontology_file_count: int,
    matcher_term_count: int,
    matcher_termset_hash: str | None,
    ontology_snapshot_hash: str | None,
    ontology_coverage_path_count: int = 0,
    ontology_graph_relation_count: int = 0,
    ontology_validation_issue_count: int = 0,
    ontology_validation_issue_samples: list[str] | None = None,
    config_drift_warnings: list[str] | None = None,
) -> CorpusStatusSummary:
    """Compute corpus, ontology, and retrieval status counters.

    Args:
        connection: Open SQLite connection.
        ontology_file_count: Number of ontology source files.
        matcher_term_count: Number of matcher terms loaded.
        matcher_termset_hash: Hash of the matcher term set.
        ontology_snapshot_hash: Hash of the ontology snapshot.
        ontology_coverage_path_count: Count of coverage paths discovered.
        ontology_graph_relation_count: Count of ontology graph relations.
        ontology_validation_issue_count: Count of ontology validation issues.
        ontology_validation_issue_samples: Sample ontology validation issue messages.
        config_drift_warnings: Configuration drift warnings to include.

    Returns:
        Current corpus and ontology summary counts.
    """
    manifest_row = connection.execute(
        """
        SELECT
            COUNT(*) AS corpus_file_count,
            SUM(CASE WHEN source_type = 'image_png' AND lifecycle_status != 'deleted' THEN 1 ELSE 0 END) AS asset_file_count,
            SUM(CASE WHEN source_type != 'image_png' AND lifecycle_status != 'deleted' THEN 1 ELSE 0 END) AS text_file_count,
            SUM(CASE WHEN retrieval_status = 'searchable' AND lifecycle_status != 'deleted' THEN 1 ELSE 0 END) AS searchable_count,
            SUM(CASE WHEN retrieval_status = 'asset_only' AND lifecycle_status != 'deleted' THEN 1 ELSE 0 END) AS asset_only_count,
            SUM(CASE WHEN retrieval_status = 'not_searchable' AND lifecycle_status != 'deleted' THEN 1 ELSE 0 END) AS not_searchable_count
        FROM corpus_manifest
        """
    ).fetchone()
    chunk_row = connection.execute("SELECT COUNT(*) AS chunk_count FROM chunk_rows").fetchone()
    mention_row = connection.execute(
        "SELECT COUNT(*) AS mention_count FROM mention_rows"
    ).fetchone()
    return CorpusStatusSummary(
        corpus_file_count=int(_row_value(manifest_row, "corpus_file_count")),
        text_file_count=int(_row_value(manifest_row, "text_file_count")),
        asset_file_count=int(_row_value(manifest_row, "asset_file_count")),
        retrieval_role_counts={
            "searchable": int(_row_value(manifest_row, "searchable_count")),
            "asset_only": int(_row_value(manifest_row, "asset_only_count")),
            "not_searchable": int(_row_value(manifest_row, "not_searchable_count")),
        },
        chunk_count=int(_row_value(chunk_row, "chunk_count")),
        mention_count=int(_row_value(mention_row, "mention_count")),
        ontology_file_count=ontology_file_count,
        matcher_term_count=matcher_term_count,
        matcher_termset_hash=matcher_termset_hash,
        ontology_snapshot_hash=ontology_snapshot_hash,
        ontology_coverage_path_count=ontology_coverage_path_count,
        ontology_graph_relation_count=ontology_graph_relation_count,
        ontology_validation_issue_count=ontology_validation_issue_count,
        ontology_validation_issue_samples=ontology_validation_issue_samples or [],
        config_drift_warnings=config_drift_warnings or [],
    )


# ---------------------------------------------------------------------------
# Knowledge Graph query functions (Phase 5)
# ---------------------------------------------------------------------------


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
    return [_claim_from_row(row) for row in rows]


def load_claims_for_chunk(connection: sqlite3.Connection, chunk_id: str) -> list[ClaimRecord]:
    """Load all claims extracted from a specific chunk."""
    rows = connection.execute(
        "SELECT * FROM claims WHERE chunk_id = ? ORDER BY confidence DESC",
        (chunk_id,),
    ).fetchall()
    return [_claim_from_row(row) for row in rows]


def count_claims(connection: sqlite3.Connection) -> int:
    """Return total claim count."""
    row = connection.execute("SELECT COUNT(*) AS cnt FROM claims").fetchone()
    return int(_row_value(row, "cnt"))


def load_chunk_ids_with_claims(connection: sqlite3.Connection) -> set[str]:
    """Return chunk IDs that already have claims extracted."""
    rows = connection.execute("SELECT DISTINCT chunk_id FROM claims").fetchall()
    return {str(row["chunk_id"]) for row in rows}


def upsert_entity_profile(connection: sqlite3.Connection, record: EntityProfileRecord) -> None:
    """Insert or update an entity profile."""
    with connection:
        connection.execute(
            """
            INSERT INTO entity_profiles (
                entity_id, label, entity_type, domain, aliases_json,
                deterministic_summary, llm_summary,
                chunk_count, doc_count, mention_count, claim_count,
                top_predicates_json, top_claims_json,
                pagerank, betweenness, closeness,
                in_degree, out_degree, eigenvector,
                community_id, source_hash, generated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(entity_id) DO UPDATE SET
                label = excluded.label,
                entity_type = excluded.entity_type,
                domain = excluded.domain,
                aliases_json = excluded.aliases_json,
                deterministic_summary = excluded.deterministic_summary,
                llm_summary = excluded.llm_summary,
                chunk_count = excluded.chunk_count,
                doc_count = excluded.doc_count,
                mention_count = excluded.mention_count,
                claim_count = excluded.claim_count,
                top_predicates_json = excluded.top_predicates_json,
                top_claims_json = excluded.top_claims_json,
                pagerank = excluded.pagerank,
                betweenness = excluded.betweenness,
                closeness = excluded.closeness,
                in_degree = excluded.in_degree,
                out_degree = excluded.out_degree,
                eigenvector = excluded.eigenvector,
                community_id = excluded.community_id,
                source_hash = excluded.source_hash,
                generated_at = excluded.generated_at
            """,
            (
                record.entity_id,
                record.label,
                record.entity_type,
                record.domain,
                record.aliases_json,
                record.deterministic_summary,
                record.llm_summary,
                record.chunk_count,
                record.doc_count,
                record.mention_count,
                record.claim_count,
                record.top_predicates_json,
                record.top_claims_json,
                record.pagerank,
                record.betweenness,
                record.closeness,
                record.in_degree,
                record.out_degree,
                record.eigenvector,
                record.community_id,
                record.source_hash,
                record.generated_at,
            ),
        )


def load_entity_profile(
    connection: sqlite3.Connection, entity_id: str
) -> EntityProfileRecord | None:
    """Load a single entity profile by ID."""
    row = connection.execute(
        "SELECT * FROM entity_profiles WHERE entity_id = ?", (entity_id,)
    ).fetchone()
    if row is None:
        return None
    return _entity_profile_from_row(row)


def load_all_entity_profiles(connection: sqlite3.Connection) -> list[EntityProfileRecord]:
    """Load all entity profiles, ordered by PageRank descending."""
    rows = connection.execute("SELECT * FROM entity_profiles ORDER BY pagerank DESC").fetchall()
    return [_entity_profile_from_row(row) for row in rows]


def search_entity_profiles(
    connection: sqlite3.Connection,
    query: str,
    *,
    limit: int = 20,
) -> list[EntityProfileRecord]:
    """Search entity profiles by label or alias substring, ranked by PageRank."""
    pattern = f"%{query}%"
    rows = connection.execute(
        """
        SELECT * FROM entity_profiles
        WHERE label LIKE ? OR aliases_json LIKE ?
        ORDER BY pagerank DESC
        LIMIT ?
        """,
        (pattern, pattern, limit),
    ).fetchall()
    return [_entity_profile_from_row(row) for row in rows]


def load_top_entities_by_pagerank(
    connection: sqlite3.Connection, *, limit: int = 20
) -> list[EntityProfileRecord]:
    """Load top entities ranked by PageRank."""
    rows = connection.execute(
        "SELECT * FROM entity_profiles ORDER BY pagerank DESC LIMIT ?", (limit,)
    ).fetchall()
    return [_entity_profile_from_row(row) for row in rows]


def load_top_entities_by_betweenness(
    connection: sqlite3.Connection, *, limit: int = 20
) -> list[EntityProfileRecord]:
    """Load top entities ranked by betweenness centrality."""
    rows = connection.execute(
        "SELECT * FROM entity_profiles ORDER BY betweenness DESC LIMIT ?", (limit,)
    ).fetchall()
    return [_entity_profile_from_row(row) for row in rows]


def load_top_entities_by_closeness(
    connection: sqlite3.Connection, *, limit: int = 20
) -> list[EntityProfileRecord]:
    """Load top entities ranked by closeness centrality."""
    rows = connection.execute(
        "SELECT * FROM entity_profiles ORDER BY closeness DESC LIMIT ?", (limit,)
    ).fetchall()
    return [_entity_profile_from_row(row) for row in rows]


def load_entity_profile_source_hashes(
    connection: sqlite3.Connection,
) -> dict[str, str]:
    """Load entity_id → source_hash mapping for incremental rebuild."""
    rows = connection.execute("SELECT entity_id, source_hash FROM entity_profiles").fetchall()
    return {str(row["entity_id"]): str(row["source_hash"]) for row in rows}


def replace_entity_communities(
    connection: sqlite3.Connection, records: list[EntityCommunityRecord]
) -> None:
    """Replace all community assignments (truncate and rebuild)."""
    with connection:
        connection.execute("DELETE FROM entity_communities")
        if records:
            connection.executemany(
                """
                INSERT INTO entity_communities (
                    entity_id, community_id, community_level, modularity_class, assigned_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        r.entity_id,
                        r.community_id,
                        r.community_level,
                        r.modularity_class,
                        r.assigned_at,
                    )
                    for r in records
                ],
            )


def load_entity_community(
    connection: sqlite3.Connection, entity_id: str
) -> EntityCommunityRecord | None:
    """Load community assignment for one entity."""
    row = connection.execute(
        "SELECT * FROM entity_communities WHERE entity_id = ?", (entity_id,)
    ).fetchone()
    if row is None:
        return None
    return EntityCommunityRecord(
        entity_id=str(row["entity_id"]),
        community_id=int(row["community_id"]),
        community_level=int(row["community_level"]),
        modularity_class=_optional_str(row["modularity_class"]),
        assigned_at=str(row["assigned_at"]),
    )


def load_community_members(connection: sqlite3.Connection, community_id: int) -> list[str]:
    """Return entity IDs belonging to a community."""
    rows = connection.execute(
        "SELECT entity_id FROM entity_communities WHERE community_id = ?",
        (community_id,),
    ).fetchall()
    return [str(row["entity_id"]) for row in rows]


def upsert_community_report(connection: sqlite3.Connection, record: CommunityReportRecord) -> None:
    """Insert or update a community report."""
    with connection:
        connection.execute(
            """
            INSERT INTO community_reports (
                community_id, community_level, member_count, member_entity_ids_json,
                deterministic_summary, llm_summary,
                top_entities_json, top_claims_json,
                intra_community_edge_count, source_hash, generated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(community_id) DO UPDATE SET
                community_level = excluded.community_level,
                member_count = excluded.member_count,
                member_entity_ids_json = excluded.member_entity_ids_json,
                deterministic_summary = excluded.deterministic_summary,
                llm_summary = excluded.llm_summary,
                top_entities_json = excluded.top_entities_json,
                top_claims_json = excluded.top_claims_json,
                intra_community_edge_count = excluded.intra_community_edge_count,
                source_hash = excluded.source_hash,
                generated_at = excluded.generated_at
            """,
            (
                record.community_id,
                record.community_level,
                record.member_count,
                record.member_entity_ids_json,
                record.deterministic_summary,
                record.llm_summary,
                record.top_entities_json,
                record.top_claims_json,
                record.intra_community_edge_count,
                record.source_hash,
                record.generated_at,
            ),
        )


def load_community_report(
    connection: sqlite3.Connection, community_id: int
) -> CommunityReportRecord | None:
    """Load a single community report."""
    row = connection.execute(
        "SELECT * FROM community_reports WHERE community_id = ?", (community_id,)
    ).fetchone()
    if row is None:
        return None
    return _community_report_from_row(row)


def load_all_community_reports(
    connection: sqlite3.Connection,
) -> list[CommunityReportRecord]:
    """Load all community reports."""
    rows = connection.execute(
        "SELECT * FROM community_reports ORDER BY member_count DESC"
    ).fetchall()
    return [_community_report_from_row(row) for row in rows]


def delete_stale_community_reports(connection: sqlite3.Connection) -> int:
    """Remove community reports whose community_id no longer exists in entity_communities."""
    with connection:
        cursor = connection.execute(
            """
            DELETE FROM community_reports
            WHERE community_id NOT IN (
                SELECT DISTINCT community_id FROM entity_communities
            )
            """
        )
    return cursor.rowcount


def replace_canonical_relations(
    connection: sqlite3.Connection, records: list[CanonicalRelationRecord]
) -> None:
    """Truncate and rebuild the canonical relations table."""
    with connection:
        connection.execute("DELETE FROM relations")
        if records:
            connection.executemany(
                """
                INSERT INTO relations (
                    relation_id, subject_entity_id, predicate, object_entity_id,
                    support_count, avg_confidence, min_confidence, max_confidence,
                    first_seen_at, last_seen_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        r.relation_id,
                        r.subject_entity_id,
                        r.predicate,
                        r.object_entity_id,
                        r.support_count,
                        r.avg_confidence,
                        r.min_confidence,
                        r.max_confidence,
                        r.first_seen_at,
                        r.last_seen_at,
                    )
                    for r in records
                ],
            )


def load_canonical_relation(
    connection: sqlite3.Connection, relation_id: str
) -> CanonicalRelationRecord | None:
    """Load a single canonical relation by ID."""
    row = connection.execute(
        "SELECT * FROM relations WHERE relation_id = ?", (relation_id,)
    ).fetchone()
    if row is None:
        return None
    return _canonical_relation_from_row(row)


def load_relations_for_entity(
    connection: sqlite3.Connection,
    entity_id: str,
    *,
    limit: int = 50,
) -> list[CanonicalRelationRecord]:
    """Load canonical relations where entity appears as subject or object."""
    rows = connection.execute(
        """
        SELECT * FROM relations
        WHERE subject_entity_id = ? OR object_entity_id = ?
        ORDER BY support_count DESC
        LIMIT ?
        """,
        (entity_id, entity_id, limit),
    ).fetchall()
    return [_canonical_relation_from_row(row) for row in rows]


def load_top_predicates_for_entity(
    connection: sqlite3.Connection,
    entity_id: str,
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Return top predicates for an entity by frequency."""
    rows = connection.execute(
        """
        SELECT predicate, COUNT(*) AS cnt
        FROM relations
        WHERE subject_entity_id = ? OR object_entity_id = ?
        GROUP BY predicate
        ORDER BY cnt DESC
        LIMIT ?
        """,
        (entity_id, entity_id, limit),
    ).fetchall()
    return [{"predicate": str(row["predicate"]), "count": int(row["cnt"])} for row in rows]


def replace_relation_evidence(
    connection: sqlite3.Connection, records: list[RelationEvidenceRecord]
) -> None:
    """Truncate and rebuild the relation evidence table."""
    with connection:
        connection.execute("DELETE FROM relation_evidence")
        if records:
            connection.executemany(
                """
                INSERT INTO relation_evidence (
                    evidence_id, relation_id, chunk_id,
                    surface_subject, surface_object, evidence_text,
                    confidence, extraction_model, extracted_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        r.evidence_id,
                        r.relation_id,
                        r.chunk_id,
                        r.surface_subject,
                        r.surface_object,
                        r.evidence_text,
                        r.confidence,
                        r.extraction_model,
                        r.extracted_at,
                    )
                    for r in records
                ],
            )


def load_evidence_for_relation(
    connection: sqlite3.Connection, relation_id: str
) -> list[RelationEvidenceRecord]:
    """Load all evidence records for a canonical relation."""
    rows = connection.execute(
        """
        SELECT * FROM relation_evidence
        WHERE relation_id = ?
        ORDER BY confidence DESC
        """,
        (relation_id,),
    ).fetchall()
    return [
        RelationEvidenceRecord(
            evidence_id=str(row["evidence_id"]),
            relation_id=str(row["relation_id"]),
            chunk_id=str(row["chunk_id"]),
            surface_subject=str(row["surface_subject"]),
            surface_object=str(row["surface_object"]),
            evidence_text=str(row["evidence_text"]),
            confidence=float(row["confidence"]),
            extraction_model=str(row["extraction_model"]),
            extracted_at=str(row["extracted_at"]),
        )
        for row in rows
    ]


def count_canonical_relations(connection: sqlite3.Connection) -> int:
    """Return total canonical relation count."""
    row = connection.execute("SELECT COUNT(*) AS cnt FROM relations").fetchone()
    return int(_row_value(row, "cnt"))


def count_relation_evidence(connection: sqlite3.Connection) -> int:
    """Return total relation evidence count."""
    row = connection.execute("SELECT COUNT(*) AS cnt FROM relation_evidence").fetchone()
    return int(_row_value(row, "cnt"))


def begin_graph_build(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    started_at: str,
    graph_version: int,
) -> None:
    """Insert initial graph build state row."""
    with connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO graph_build_state (
                run_id, started_at, finished_at, status, current_phase, graph_version,
                relations_consolidated, evidence_rows_built, claims_extracted,
                entity_profiles_built, communities_detected, community_reports_built,
                centrality_computed, entity_embeddings_computed, llm_enrichment_count,
                notes_json
            )
            VALUES (?, ?, NULL, 'running', 'pending', ?, 0, 0, 0, 0, 0, 0, 0, 0, 0, '[]')
            """,
            (run_id, started_at, graph_version),
        )


def update_graph_build_phase(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    current_phase: str,
    **counters: int,
) -> None:
    """Update the current phase and counter columns on a graph build."""
    set_clauses = ["current_phase = ?"]
    params: list[Any] = [current_phase]
    valid_columns = {
        "relations_consolidated",
        "evidence_rows_built",
        "claims_extracted",
        "entity_profiles_built",
        "communities_detected",
        "community_reports_built",
        "centrality_computed",
        "entity_embeddings_computed",
        "llm_enrichment_count",
    }
    for key, value in counters.items():
        if key in valid_columns:
            set_clauses.append(f"{key} = ?")
            params.append(value)
    params.append(run_id)
    with connection:
        connection.execute(
            f"UPDATE graph_build_state SET {', '.join(set_clauses)} WHERE run_id = ?",
            params,
        )


def finish_graph_build(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    finished_at: str,
    status: str,
    notes: list[str],
) -> None:
    """Finalise a graph build run."""
    with connection:
        connection.execute(
            """
            UPDATE graph_build_state
            SET finished_at = ?, status = ?, notes_json = ?
            WHERE run_id = ?
            """,
            (finished_at, status, json.dumps(notes, separators=(",", ":")), run_id),
        )


def load_latest_graph_build_state(
    connection: sqlite3.Connection,
) -> GraphBuildStateRecord | None:
    """Load the most recent graph build state row."""
    row = connection.execute(
        "SELECT * FROM graph_build_state ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return None
    return GraphBuildStateRecord(
        run_id=str(row["run_id"]),
        started_at=str(row["started_at"]),
        finished_at=_optional_str(row["finished_at"]),
        status=str(row["status"]),
        current_phase=str(row["current_phase"]),
        graph_version=int(row["graph_version"]),
        relations_consolidated=int(row["relations_consolidated"]),
        evidence_rows_built=int(row["evidence_rows_built"]),
        claims_extracted=int(row["claims_extracted"]),
        entity_profiles_built=int(row["entity_profiles_built"]),
        communities_detected=int(row["communities_detected"]),
        community_reports_built=int(row["community_reports_built"]),
        centrality_computed=int(row["centrality_computed"]),
        entity_embeddings_computed=int(row["entity_embeddings_computed"]),
        llm_enrichment_count=int(row["llm_enrichment_count"]),
        notes_json=str(row["notes_json"]),
    )


def upsert_graph_metadata(
    connection: sqlite3.Connection,
    key: str,
    value: str,
    updated_at: str,
) -> None:
    """Insert or update a graph metadata key-value entry."""
    with connection:
        connection.execute(
            """
            INSERT INTO graph_metadata (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
            """,
            (key, value, updated_at),
        )


def load_graph_metadata(connection: sqlite3.Connection) -> dict[str, str]:
    """Load all graph metadata key-value pairs."""
    rows = connection.execute("SELECT key, value FROM graph_metadata").fetchall()
    return {str(row["key"]): str(row["value"]) for row in rows}


def load_graph_version(connection: sqlite3.Connection) -> int:
    """Load the current graph version number, defaulting to 0."""
    row = connection.execute(
        "SELECT value FROM graph_metadata WHERE key = 'graph_version'"
    ).fetchone()
    if row is None:
        return 0
    return int(row["value"])


def load_all_extracted_relations(
    connection: sqlite3.Connection,
) -> list[ExtractedRelationRecord]:
    """Load all rows from extracted_relations."""
    rows = connection.execute(
        """
        SELECT relation_id, chunk_id, document_id, source_rel_path,
               subject_entity_id, predicate, object_entity_id,
               confidence, extraction_model, extracted_at
        FROM extracted_relations
        ORDER BY subject_entity_id, predicate, object_entity_id
        """
    ).fetchall()
    return [
        ExtractedRelationRecord(
            relation_id=str(row["relation_id"]),
            chunk_id=str(row["chunk_id"]),
            document_id=str(row["document_id"]),
            source_rel_path=str(row["source_rel_path"]),
            subject_entity_id=str(row["subject_entity_id"]),
            predicate=str(row["predicate"]),
            object_entity_id=str(row["object_entity_id"]),
            confidence=float(row["confidence"]),
            extraction_model=str(row["extraction_model"]),
            extracted_at=str(row["extracted_at"]),
        )
        for row in rows
    ]


def load_entity_mention_stats(
    connection: sqlite3.Connection,
) -> dict[str, dict[str, int]]:
    """Load per-entity mention statistics (chunk_count, doc_count, mention_count)."""
    rows = connection.execute(
        """
        SELECT
            m.entity_id,
            COUNT(DISTINCT m.chunk_id) AS chunk_count,
            COUNT(DISTINCT c.source_rel_path) AS doc_count,
            COUNT(*) AS mention_count
        FROM mention_rows m
        JOIN chunk_rows c ON m.chunk_id = c.chunk_id
        GROUP BY m.entity_id
        """
    ).fetchall()
    return {
        str(row["entity_id"]): {
            "chunk_count": int(row["chunk_count"]),
            "doc_count": int(row["doc_count"]),
            "mention_count": int(row["mention_count"]),
        }
        for row in rows
    }


def load_chunk_ids_for_entity(
    connection: sqlite3.Connection, entity_id: str, *, limit: int = 100
) -> list[str]:
    """Return chunk IDs mentioning an entity, ordered by mention frequency."""
    rows = connection.execute(
        """
        SELECT chunk_id, COUNT(*) AS cnt
        FROM mention_rows
        WHERE entity_id = ?
        GROUP BY chunk_id
        ORDER BY cnt DESC
        LIMIT ?
        """,
        (entity_id, limit),
    ).fetchall()
    return [str(row["chunk_id"]) for row in rows]


def count_entity_profiles(connection: sqlite3.Connection) -> int:
    """Return total entity profile count."""
    row = connection.execute("SELECT COUNT(*) AS cnt FROM entity_profiles").fetchone()
    return int(_row_value(row, "cnt"))


def count_communities(connection: sqlite3.Connection) -> int:
    """Return number of distinct communities."""
    row = connection.execute(
        "SELECT COUNT(DISTINCT community_id) AS cnt FROM entity_communities"
    ).fetchone()
    return int(_row_value(row, "cnt"))


def count_community_reports(connection: sqlite3.Connection) -> int:
    """Return total community report count."""
    row = connection.execute("SELECT COUNT(*) AS cnt FROM community_reports").fetchone()
    return int(_row_value(row, "cnt"))


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _claim_from_row(row: sqlite3.Row) -> ClaimRecord:
    return ClaimRecord(
        claim_id=str(row["claim_id"]),
        chunk_id=str(row["chunk_id"]),
        document_id=str(row["document_id"]),
        source_rel_path=str(row["source_rel_path"]),
        claim_text=str(row["claim_text"]),
        subject_entity_id=_optional_str(row["subject_entity_id"]),
        object_entity_id=_optional_str(row["object_entity_id"]),
        claim_type=str(row["claim_type"]),
        confidence=float(row["confidence"]),
        extraction_model=str(row["extraction_model"]),
        extracted_at=str(row["extracted_at"]),
    )


def _entity_profile_from_row(row: sqlite3.Row) -> EntityProfileRecord:
    return EntityProfileRecord(
        entity_id=str(row["entity_id"]),
        label=str(row["label"]),
        entity_type=str(row["entity_type"]),
        domain=str(row["domain"]),
        aliases_json=str(row["aliases_json"]),
        deterministic_summary=str(row["deterministic_summary"]),
        llm_summary=_optional_str(row["llm_summary"]),
        chunk_count=int(row["chunk_count"]),
        doc_count=int(row["doc_count"]),
        mention_count=int(row["mention_count"]),
        claim_count=int(row["claim_count"]),
        top_predicates_json=str(row["top_predicates_json"]),
        top_claims_json=str(row["top_claims_json"]),
        pagerank=float(row["pagerank"]),
        betweenness=float(row["betweenness"]),
        closeness=float(row["closeness"]),
        in_degree=int(row["in_degree"]),
        out_degree=int(row["out_degree"]),
        eigenvector=float(row["eigenvector"]),
        community_id=int(row["community_id"]) if row["community_id"] is not None else None,
        source_hash=str(row["source_hash"]),
        generated_at=str(row["generated_at"]),
    )


def _community_report_from_row(row: sqlite3.Row) -> CommunityReportRecord:
    return CommunityReportRecord(
        community_id=int(row["community_id"]),
        community_level=int(row["community_level"]),
        member_count=int(row["member_count"]),
        member_entity_ids_json=str(row["member_entity_ids_json"]),
        deterministic_summary=str(row["deterministic_summary"]),
        llm_summary=_optional_str(row["llm_summary"]),
        top_entities_json=str(row["top_entities_json"]),
        top_claims_json=str(row["top_claims_json"]),
        intra_community_edge_count=int(row["intra_community_edge_count"]),
        source_hash=str(row["source_hash"]),
        generated_at=str(row["generated_at"]),
    )


def _canonical_relation_from_row(row: sqlite3.Row) -> CanonicalRelationRecord:
    return CanonicalRelationRecord(
        relation_id=str(row["relation_id"]),
        subject_entity_id=str(row["subject_entity_id"]),
        predicate=str(row["predicate"]),
        object_entity_id=str(row["object_entity_id"]),
        support_count=int(row["support_count"]),
        avg_confidence=float(row["avg_confidence"]),
        min_confidence=float(row["min_confidence"]),
        max_confidence=float(row["max_confidence"]),
        first_seen_at=str(row["first_seen_at"]),
        last_seen_at=str(row["last_seen_at"]),
    )


def _manifest_from_row(row: sqlite3.Row) -> ManifestRecord:
    return ManifestRecord(
        source_rel_path=str(row["file_rel_path"]),
        absolute_path=str(row["file_path"]),
        source_type=str(row["source_type"]),
        source_domain=str(row["source_domain"]),
        document_id=_optional_str(row["document_id"]),
        file_size_bytes=int(row["file_size_bytes"]),
        content_hash=str(row["blake3_hash"]),
        parent_source_path=_optional_str(row["parent_source_path"]),
        chunk_count=int(row["chunk_count"]),
        last_seen_at=str(row["last_seen_at"]),
        last_processed_at=_optional_str(row["last_processed_at"]),
        last_committed_at=_optional_str(row["last_committed_at"]),
        error_message=_optional_str(row["error_message"]),
        lifecycle_status=str(row["lifecycle_status"]),
        retrieval_status=str(row["retrieval_status"]),
    )


def _chunk_from_row(row: sqlite3.Row) -> ChunkRecord:
    vector_payload = json.loads(str(row["vector_json"]))
    if not isinstance(vector_payload, list):
        raise ValueError("Expected vector_json to decode to a list of floats")
    return ChunkRecord(
        chunk_id=str(row["chunk_id"]),
        document_id=str(row["document_id"]),
        source_rel_path=str(row["source_rel_path"]),
        source_path=str(row["source_path"]),
        source_filename=str(row["source_filename"]),
        source_type=str(row["source_type"]),
        source_domain=str(row["source_domain"]),
        source_hash=str(row["source_hash"]),
        citation_label=str(row["citation_label"]),
        chunk_index=int(row["chunk_index"]),
        chunk_occurrence=int(row["chunk_occurrence"]),
        token_count=int(row["token_count"]),
        text=str(row["text"]),
        chunk_hash=str(row["chunk_hash"]),
        score_hint=str(row["score_hint"]),
        metadata_json=str(row["metadata_json"]),
        vector=[float(item) for item in vector_payload],
        embedding_model=str(row["embedding_model"]),
        embedding_dims=int(row["embedding_dims"]),
    )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _mention_id(record: MentionRecord) -> str:
    return blake3_hex(record.entity_id, record.chunk_id, str(record.start_char))


def _row_value(row: sqlite3.Row | None, key: str) -> int:
    if row is None:
        return 0
    value: Any = row[key]
    if value is None:
        return 0
    return int(value)


def _migrate_legacy_schema(connection: sqlite3.Connection) -> None:
    _migrate_legacy_corpus_manifest(connection)
    _migrate_legacy_chunk_rows(connection)
    _migrate_legacy_mention_rows(connection)
    _migrate_legacy_ontology_snapshot(connection)
    _migrate_legacy_ingest_runs(connection)


def _migrate_legacy_corpus_manifest(connection: sqlite3.Connection) -> None:
    if not _table_exists(connection, "corpus_manifest"):
        return
    columns = _table_columns(connection, "corpus_manifest")
    if "blake3_hash" in columns and "file_path" in columns and "file_rel_path" in columns:
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
    if "document_id" in columns and "source_path" in columns and "metadata_json" in columns:
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
    required_columns = {"mention_id", "term_source", "source_path", "source_domain"}
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


def _table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    row = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _table_columns(connection: sqlite3.Connection, table_name: str) -> set[str]:
    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {str(row["name"]) for row in rows}
