"""Ontology state, ingest config snapshot, and committed-state probes."""

import sqlite3

from lxd.stores._sqlite_rows import row_value
from lxd.stores.models import (
    IngestConfigSnapshotRecord,
    OntologySnapshotRecord,
    OntologySourceRecord,
)


def replace_ontology_sources(
    connection: sqlite3.Connection, records: list[OntologySourceRecord]
) -> None:
    """Replace persisted ontology source records."""
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
    """Replace the persisted ontology snapshot row."""
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
    """Replace persisted ingest config key-value rows."""
    with connection:
        connection.execute("DELETE FROM ingest_config")
        if records:
            connection.executemany(
                "INSERT INTO ingest_config (key, value) VALUES (?, ?)",
                [(record.key, record.value) for record in records],
            )


def list_allowed_domains(connection: sqlite3.Connection) -> set[str]:
    """List available source domains from committed manifest rows."""
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
    """Load persisted ingest config key-value rows."""
    rows = connection.execute("SELECT key, value FROM ingest_config ORDER BY key").fetchall()
    return {str(row["key"]): str(row["value"]) for row in rows}


def load_ontology_snapshot(connection: sqlite3.Connection) -> OntologySnapshotRecord | None:
    """Load the persisted ontology snapshot record."""
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
    """Return whether committed corpus state exists."""
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
