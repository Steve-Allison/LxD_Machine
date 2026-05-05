"""Corpus manifest + asset_links + per-source delete."""

from __future__ import annotations

import sqlite3
from collections import defaultdict

from lxd.stores._sqlite_rows import manifest_from_row
from lxd.stores.models import AssetLinkRecord, ManifestRecord


def load_manifest_index(connection: sqlite3.Connection) -> dict[str, ManifestRecord]:
    """Load manifest records keyed by relative path."""
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
    """Load manifest records grouped by content hash."""
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
    """Load one manifest record by relative path."""
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
    """Insert or update a corpus manifest record."""
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
    """Insert or update an asset-to-parent linkage record."""
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


def delete_source(connection: sqlite3.Connection, source_rel_path: str) -> None:
    """Delete a source and its dependent rows."""
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
