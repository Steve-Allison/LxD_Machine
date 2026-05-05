"""Ingest run lifecycle: begin / progress-update / finish."""

from __future__ import annotations

import json
import sqlite3


def begin_ingest_run(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    started_at: str,
    mode: str,
    files_total: int,
) -> None:
    """Insert the initial ingest run row."""
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
    """Persist incremental ingest progress counters."""
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
