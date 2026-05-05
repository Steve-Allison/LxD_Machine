"""Persist ingest state, manifests, and ontology snapshots in SQLite."""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

from lxd.stores._sqlite_rows import (
    canonical_relation_from_row,
    chunk_from_row,
    claim_from_row,
    community_report_from_row,
    entity_profile_from_row,
    manifest_from_row,
    mention_id,
    optional_str,
    row_value,
)
from lxd.stores.models import (
    AssetLinkRecord,
    CanonicalRelationRecord,
    ChunkCentralitySignals,
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
    """Resolve SQLite and LanceDB paths under the data directory.

    Args:
        data_path: Root data directory for local stores.

    Returns:
        Resolved SQLite and LanceDB store paths.
    """
    return StorePaths(sqlite_path=data_path / _SQLITE_FILENAME, lancedb_path=data_path / "lancedb")


def assert_no_v2_legacy_tables(connection: sqlite3.Connection) -> None:
    """Refuse to proceed if any ``*_v2_legacy`` table is present.

    These tables only exist as the smoking gun of a half-finished pre-v0
    migration. The numbered migration system in :mod:`lxd.stores.schema`
    cannot reason about them and downstream writes will fail mid-batch if
    we silently continue. Raising here surfaces the corruption loudly.

    Args:
        connection: Open SQLite connection.

    Raises:
        sqlite3.DatabaseError: If any ``*_v2_legacy`` table is present.
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

    Args:
        connection: Open SQLite connection.

    Side Effects:
        Executes DDL and bumps ``PRAGMA user_version`` to the current schema
        version.
    """
    with connection:
        assert_no_v2_legacy_tables(connection)
    ensure_schema(connection)
    with connection:
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
    embedding_tokens: int | None = None,
    llm_tokens: int | None = None,
    estimated_cost_usd: float | None = None,
    embedding_cache_hits: int | None = None,
    embedding_cache_misses: int | None = None,
) -> None:
    """Finalize ingest run status and counters.

    Telemetry columns are nullable so callers that don't know the values
    can omit them without breaking schema constraints.
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
                notes = ?,
                embedding_tokens = ?,
                llm_tokens = ?,
                estimated_cost_usd = ?,
                embedding_cache_hits = ?,
                embedding_cache_misses = ?
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
                embedding_tokens,
                llm_tokens,
                estimated_cost_usd,
                embedding_cache_hits,
                embedding_cache_misses,
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
            source_rel_path,
            absolute_path,
            source_type,
            source_domain,
            document_id,
            blake3_hash,
            file_size_bytes,
            parent_source_rel_path,
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
    return {record.source_rel_path: record for record in (manifest_from_row(row) for row in rows)}


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
            source_rel_path,
            absolute_path,
            source_type,
            source_domain,
            document_id,
            blake3_hash,
            file_size_bytes,
            parent_source_rel_path,
            lifecycle_status,
            retrieval_status,
            chunk_count,
            last_seen_at,
            last_processed_at,
            last_committed_at,
            error_message
        FROM corpus_manifest
        ORDER BY source_rel_path
        """
    ).fetchall()
    grouped: dict[str, list[ManifestRecord]] = defaultdict(list)
    for row in rows:
        record = manifest_from_row(row)
        grouped[record.content_hash].append(record)
    return dict(grouped)


def load_manifest_by_rel_path(
    connection: sqlite3.Connection, rel_path: str
) -> ManifestRecord | None:
    """Load one manifest record by relative path.

    Args:
        connection: Open SQLite connection.
        rel_path: Corpus-relative source file path.

    Returns:
        Matching manifest record, if present.
    """
    row = connection.execute(
        """
        SELECT
            source_rel_path,
            absolute_path,
            source_type,
            source_domain,
            document_id,
            blake3_hash,
            file_size_bytes,
            parent_source_rel_path,
            lifecycle_status,
            retrieval_status,
            chunk_count,
            last_seen_at,
            last_processed_at,
            last_committed_at,
            error_message
        FROM corpus_manifest
        WHERE source_rel_path = ?
        """,
        (rel_path,),
    ).fetchone()
    if row is None:
        return None
    return manifest_from_row(row)


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
                source_rel_path,
                absolute_path,
                source_type,
                source_domain,
                document_id,
                blake3_hash,
                file_size_bytes,
                parent_source_rel_path,
                lifecycle_status,
                retrieval_status,
                chunk_count,
                last_seen_at,
                last_processed_at,
                last_committed_at,
                error_message
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_rel_path) DO UPDATE SET
                absolute_path = excluded.absolute_path,
                source_type = excluded.source_type,
                source_domain = excluded.source_domain,
                document_id = excluded.document_id,
                blake3_hash = excluded.blake3_hash,
                file_size_bytes = excluded.file_size_bytes,
                parent_source_rel_path = excluded.parent_source_rel_path,
                lifecycle_status = excluded.lifecycle_status,
                retrieval_status = excluded.retrieval_status,
                chunk_count = excluded.chunk_count,
                last_seen_at = excluded.last_seen_at,
                last_processed_at = excluded.last_processed_at,
                last_committed_at = excluded.last_committed_at,
                error_message = excluded.error_message
            """,
            (
                record.source_rel_path,
                record.absolute_path,
                record.source_type,
                record.source_domain,
                record.document_id,
                record.content_hash,
                record.file_size_bytes,
                record.parent_source_rel_path,
                record.lifecycle_status,
                record.retrieval_status,
                record.chunk_count,
                record.last_seen_at,
                record.last_processed_at,
                record.last_committed_at,
                record.error_message,
            ),
        )


def upsert_asset_link(connection: sqlite3.Connection, record: AssetLinkRecord) -> None:
    """Insert or update an asset-to-parent linkage record.

    Args:
        connection: Open SQLite connection.
        record: Record instance to persist.
    """
    with connection:
        connection.execute(
            """
            INSERT INTO asset_links (
                asset_rel_path,
                asset_filename,
                source_domain,
                parent_source_rel_path,
                parent_document_id,
                page_no,
                asset_index,
                link_method,
                blake3_hash,
                last_committed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(asset_rel_path) DO UPDATE SET
                asset_filename = excluded.asset_filename,
                source_domain = excluded.source_domain,
                parent_source_rel_path = excluded.parent_source_rel_path,
                parent_document_id = excluded.parent_document_id,
                page_no = excluded.page_no,
                asset_index = excluded.asset_index,
                link_method = excluded.link_method,
                blake3_hash = excluded.blake3_hash,
                last_committed_at = excluded.last_committed_at
            """,
            (
                record.asset_rel_path,
                record.asset_filename,
                record.source_domain,
                record.parent_source_rel_path,
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
                INSERT INTO ontology_sources (file_rel_path, blake3_hash, last_seen_at)
                VALUES (?, ?, ?)
                """,
                [
                    (
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
    if int(row_value(manifest_row, "count")) > 0:
        return True
    chunk_row = connection.execute("SELECT COUNT(*) AS count FROM chunk_rows").fetchone()
    if int(row_value(chunk_row, "count")) > 0:
        return True
    mention_row = connection.execute("SELECT COUNT(*) AS count FROM mention_rows").fetchone()
    return int(row_value(mention_row, "count")) > 0


def delete_source(connection: sqlite3.Connection, source_rel_path: str) -> None:
    """Delete a source and its dependent rows.

    Args:
        connection: Open SQLite connection.
        source_rel_path: Corpus-relative source file path.
    """
    with connection:
        connection.execute(
            """
            UPDATE corpus_manifest
            SET lifecycle_status = 'deleted',
                retrieval_status = 'not_searchable',
                chunk_count = 0
            WHERE source_rel_path = ?
            """,
            (source_rel_path,),
        )
        connection.execute("DELETE FROM chunk_rows WHERE source_rel_path = ?", (source_rel_path,))
        connection.execute("DELETE FROM asset_links WHERE asset_rel_path = ?", (source_rel_path,))


def replace_source_chunks(
    connection: sqlite3.Connection,
    *,
    source_rel_path: str,
    chunk_records: list[ChunkRecord],
    mention_records: list[MentionRecord],
    relation_records: list[ExtractedRelationRecord] | None = None,
) -> None:
    """Replace all vector chunks for one source path.

    Args:
        connection: Open SQLite connection.
        source_rel_path: Corpus-relative source file path.
        chunk_records: Chunk rows to persist for a source.
        mention_records: Mention rows to persist for the source.
        relation_records: Extracted relation rows to persist for the source.
    """
    with connection:
        connection.execute("DELETE FROM chunk_rows WHERE source_rel_path = ?", (source_rel_path,))
        connection.execute("DELETE FROM mention_rows WHERE source_rel_path = ?", (source_rel_path,))
        if chunk_records:
            connection.executemany(
                """
                INSERT INTO chunk_rows (
                    chunk_id,
                    document_id,
                    source_rel_path,
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
                    embedding_model,
                    embedding_dims,
                    cited_sources_json,
                    wiki_links_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        record.chunk_id,
                        record.document_id,
                        record.source_rel_path,
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
                        record.embedding_model,
                        record.embedding_dims,
                        json.dumps(list(record.cited_sources)),
                        json.dumps(list(record.wiki_links)),
                    )
                    for record in chunk_records
                ],
            )
        if mention_records:
            rel_path = chunk_records[0].source_rel_path if chunk_records else source_rel_path
            source_domain = chunk_records[0].source_domain if chunk_records else ""
            source_filename = Path(rel_path).name if rel_path else ""
            connection.executemany(
                """
                INSERT INTO mention_rows (
                    mention_id,
                    entity_id,
                    term_source,
                    source_domain,
                    source_rel_path,
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
                        mention_id(record),
                        record.entity_id,
                        record.term_source,
                        source_domain,
                        rel_path,
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
    connection: sqlite3.Connection, source_rel_path: str
) -> list[ChunkRecord]:
    """Load persisted chunk records for a source path.

    Args:
        connection: Open SQLite connection.
        source_rel_path: Corpus-relative source file path.

    Returns:
        Chunk records for the source path.
    """
    rows = connection.execute(
        """
        SELECT
            chunk_id,
            document_id,
            source_rel_path,
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
            embedding_model,
            embedding_dims,
            cited_sources_json,
            wiki_links_json
        FROM chunk_rows
        WHERE source_rel_path = ?
        ORDER BY chunk_index
        """,
        (source_rel_path,),
    ).fetchall()
    return [chunk_from_row(row) for row in rows]


def load_mentions_for_source(
    connection: sqlite3.Connection, source_rel_path: str
) -> dict[str, list[MentionRecord]]:
    """Load persisted mentions grouped by chunk ID for a source.

    Args:
        connection: Open SQLite connection.
        source_rel_path: Corpus-relative source file path.

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
        WHERE source_rel_path = ?
        ORDER BY chunk_id, start_char, end_char, entity_id
        """,
        (source_rel_path,),
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


def load_chunk_centrality_signals(
    connection: sqlite3.Connection, chunk_ids: list[str]
) -> dict[str, ChunkCentralitySignals]:
    """Return ``chunk_id -> ChunkCentralitySignals`` for the given chunks.

    Resolves each chunk's mentioned entities, joins to ``entity_profiles``
    (built by ``pixi run build-graph``), and returns:

    * ``max_pagerank`` — the highest PageRank across the chunk's mentioned
      entities, treated as the chunk's "graph importance" signal.
    * ``community_ids`` — distinct community ids the chunk's entities
      belong to, sorted ascending. Used for community-aware
      diversification at retrieval time.

    Returns an empty dict if ``chunk_ids`` is empty. Chunks with no rows
    in ``entity_profiles`` (graph not yet built) are silently absent from
    the result; callers default-fill with :class:`ChunkCentralitySignals`.
    """
    if not chunk_ids:
        return {}
    placeholders = ",".join("?" * len(chunk_ids))
    rows = connection.execute(
        f"""
        SELECT
            m.chunk_id AS chunk_id,
            MAX(p.pagerank) AS max_pagerank,
            GROUP_CONCAT(DISTINCT p.community_id) AS community_ids
        FROM mention_rows m
        JOIN entity_profiles p ON m.entity_id = p.entity_id
        WHERE m.chunk_id IN ({placeholders})
        GROUP BY m.chunk_id
        """,
        list(chunk_ids),
    ).fetchall()
    result: dict[str, ChunkCentralitySignals] = {}
    for row in rows:
        chunk_id = str(row["chunk_id"])
        max_pr_value = row["max_pagerank"]
        max_pr = float(max_pr_value) if max_pr_value is not None else 0.0
        raw_comm = row["community_ids"]
        community_ids: tuple[int, ...] = ()
        if isinstance(raw_comm, str) and raw_comm:
            community_ids = tuple(
                sorted(
                    {
                        int(token)
                        for token in raw_comm.split(",")
                        if token.strip().lstrip("-").isdigit()
                    }
                )
            )
        result[chunk_id] = ChunkCentralitySignals(
            max_pagerank=max_pr,
            community_ids=community_ids,
        )
    return result


def _summarize_manifest(connection: sqlite3.Connection) -> dict[str, int]:
    """Return manifest-level counters grouped by source type and retrieval role.

    Args:
        connection: Open SQLite connection.

    Returns:
        Mapping with keys ``corpus_file_count``, ``text_file_count``,
        ``asset_file_count``, ``searchable_count``, ``asset_only_count``,
        ``not_searchable_count``. All deleted manifests are excluded from the
        per-role tallies so status never double-counts tombstones.
    """
    row = connection.execute(
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
    return {
        "corpus_file_count": int(row_value(row, "corpus_file_count")),
        "text_file_count": int(row_value(row, "text_file_count")),
        "asset_file_count": int(row_value(row, "asset_file_count")),
        "searchable_count": int(row_value(row, "searchable_count")),
        "asset_only_count": int(row_value(row, "asset_only_count")),
        "not_searchable_count": int(row_value(row, "not_searchable_count")),
    }


def _summarize_chunk_counts(connection: sqlite3.Connection) -> tuple[int, int]:
    """Return ``(chunk_count, mention_count)`` across the whole store.

    Args:
        connection: Open SQLite connection.

    Returns:
        Two-tuple of total chunk rows and total mention rows.
    """
    chunk_row = connection.execute("SELECT COUNT(*) AS chunk_count FROM chunk_rows").fetchone()
    mention_row = connection.execute(
        "SELECT COUNT(*) AS mention_count FROM mention_rows"
    ).fetchone()
    return (
        int(row_value(chunk_row, "chunk_count")),
        int(row_value(mention_row, "mention_count")),
    )


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
    manifest = _summarize_manifest(connection)
    chunk_count, mention_count = _summarize_chunk_counts(connection)
    return CorpusStatusSummary(
        corpus_file_count=manifest["corpus_file_count"],
        text_file_count=manifest["text_file_count"],
        asset_file_count=manifest["asset_file_count"],
        retrieval_role_counts={
            "searchable": manifest["searchable_count"],
            "asset_only": manifest["asset_only_count"],
            "not_searchable": manifest["not_searchable_count"],
        },
        chunk_count=chunk_count,
        mention_count=mention_count,
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
# Knowledge Graph query functions
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
    return entity_profile_from_row(row)


def load_all_entity_profiles(connection: sqlite3.Connection) -> list[EntityProfileRecord]:
    """Load all entity profiles, ordered by PageRank descending."""
    rows = connection.execute("SELECT * FROM entity_profiles ORDER BY pagerank DESC").fetchall()
    return [entity_profile_from_row(row) for row in rows]


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
    return [entity_profile_from_row(row) for row in rows]


def load_top_entities_by_pagerank(
    connection: sqlite3.Connection, *, limit: int = 20
) -> list[EntityProfileRecord]:
    """Load top entities ranked by PageRank."""
    rows = connection.execute(
        "SELECT * FROM entity_profiles ORDER BY pagerank DESC LIMIT ?", (limit,)
    ).fetchall()
    return [entity_profile_from_row(row) for row in rows]


def load_top_entities_by_betweenness(
    connection: sqlite3.Connection, *, limit: int = 20
) -> list[EntityProfileRecord]:
    """Load top entities ranked by betweenness centrality."""
    rows = connection.execute(
        "SELECT * FROM entity_profiles ORDER BY betweenness DESC LIMIT ?", (limit,)
    ).fetchall()
    return [entity_profile_from_row(row) for row in rows]


def load_top_entities_by_closeness(
    connection: sqlite3.Connection, *, limit: int = 20
) -> list[EntityProfileRecord]:
    """Load top entities ranked by closeness centrality."""
    rows = connection.execute(
        "SELECT * FROM entity_profiles ORDER BY closeness DESC LIMIT ?", (limit,)
    ).fetchall()
    return [entity_profile_from_row(row) for row in rows]


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
        modularity_class=optional_str(row["modularity_class"]),
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
    return community_report_from_row(row)


def load_all_community_reports(
    connection: sqlite3.Connection,
) -> list[CommunityReportRecord]:
    """Load all community reports."""
    rows = connection.execute(
        "SELECT * FROM community_reports ORDER BY member_count DESC"
    ).fetchall()
    return [community_report_from_row(row) for row in rows]


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
    return canonical_relation_from_row(row)


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
    return [canonical_relation_from_row(row) for row in rows]


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
    return int(row_value(row, "cnt"))


def count_relation_evidence(connection: sqlite3.Connection) -> int:
    """Return total relation evidence count."""
    row = connection.execute("SELECT COUNT(*) AS cnt FROM relation_evidence").fetchone()
    return int(row_value(row, "cnt"))


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
        finished_at=optional_str(row["finished_at"]),
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
    return int(row_value(row, "cnt"))


def count_communities(connection: sqlite3.Connection) -> int:
    """Return number of distinct communities."""
    row = connection.execute(
        "SELECT COUNT(DISTINCT community_id) AS cnt FROM entity_communities"
    ).fetchone()
    return int(row_value(row, "cnt"))


def count_community_reports(connection: sqlite3.Connection) -> int:
    """Return total community report count."""
    row = connection.execute("SELECT COUNT(*) AS cnt FROM community_reports").fetchone()
    return int(row_value(row, "cnt"))


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
