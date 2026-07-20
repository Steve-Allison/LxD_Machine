"""Aggregate corpus / chunk / mention counters and the public summary builder."""

import sqlite3

from lxd.stores._sqlite_rows import row_value
from lxd.stores.models import CorpusStatusSummary


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
    """Compute corpus, ontology, and retrieval status counters."""
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


def _summarize_manifest(connection: sqlite3.Connection) -> dict[str, int]:
    """Manifest-level counters grouped by source type and retrieval role.

    All deleted manifests are excluded from the per-role tallies so status
    never double-counts tombstones.
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


def count_image_asset_retrieval_status(connection: sqlite3.Connection) -> dict[str, int]:
    """Return ``{"captioned": N, "asset_only": N}`` counts for PNG assets.

    Distinguishes PNG assets that became searchable via a generated caption
    (Phase 4 multimodal captioning) from those still ``asset_only``
    (captioning disabled, caption generation failed, or not yet
    backfilled). Excludes tombstoned (``deleted``) manifest rows.
    """
    row = connection.execute(
        """
        SELECT
            SUM(CASE WHEN retrieval_status = 'searchable' THEN 1 ELSE 0 END) AS captioned,
            SUM(CASE WHEN retrieval_status = 'asset_only' THEN 1 ELSE 0 END) AS asset_only
        FROM corpus_manifest
        WHERE source_type = 'image_png' AND lifecycle_status != 'deleted'
        """
    ).fetchone()
    return {
        "captioned": int(row_value(row, "captioned")),
        "asset_only": int(row_value(row, "asset_only")),
    }


def _summarize_chunk_counts(connection: sqlite3.Connection) -> tuple[int, int]:
    """Return ``(chunk_count, mention_count)`` across the whole store."""
    chunk_row = connection.execute("SELECT COUNT(*) AS chunk_count FROM chunk_rows").fetchone()
    mention_row = connection.execute(
        "SELECT COUNT(*) AS mention_count FROM mention_rows"
    ).fetchone()
    return (
        int(row_value(chunk_row, "chunk_count")),
        int(row_value(mention_row, "mention_count")),
    )
