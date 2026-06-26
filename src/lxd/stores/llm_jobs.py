"""Persistent LLM job queue backed by the SQLite ``llm_jobs`` table.

Responsibility:
    Provide a minimal, idempotent queue for long-running LLM workloads —
    OpenAI Batch jobs, Ollama background calls, or any structured-output
    task that outlives a single process. Callers pick their own ``job_id``
    (usually a blake3 hash of the input + tenancy), so re-enqueueing the
    same work is safe.

Design boundary:
    This module owns *only* the queue surface. Executors (claim extraction,
    relation extraction, batch runners) are free to interpret ``payload_json``
    however they like. The queue knows nothing about OpenAI or Ollama.

Key constraints:
    * All timestamps are ISO-8601 UTC strings for audit consistency with the
      rest of the SQLite store.
    * Status transitions are validated by a ``CHECK`` constraint in SQLite;
      callers must pass one of the five allowed states.
    * Queue operations commit via a single ``with connection`` block so they
      are atomic and safe under WAL concurrency.
"""

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

type JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]

_ALLOWED_STATUSES: frozenset[JobStatus] = frozenset(
    {"queued", "running", "succeeded", "failed", "cancelled"}
)


@dataclass(frozen=True, slots=True)
class LLMJobRecord:
    """Immutable snapshot of a row in ``llm_jobs``.

    Attributes:
        job_id: Caller-supplied stable identifier.
        kind: Logical job category (e.g. ``claims.openai_batch``).
        corpus_id: Tenancy marker (``"default"`` for single-tenant setups).
        status: One of the five allowed lifecycle states.
        payload: Parsed payload dictionary; empty when the job carried no
            structured input.
        result: Parsed result dictionary or ``None`` until completion.
        error: Short human-readable error string, or ``None``.
        attempts: Retry counter (never decreases).
        created_at: ISO-8601 UTC creation timestamp.
        updated_at: ISO-8601 UTC timestamp of the latest status transition.
    """

    job_id: str
    kind: str
    corpus_id: str
    status: JobStatus
    payload: dict[str, Any]
    result: dict[str, Any] | None
    error: str | None
    attempts: int
    created_at: str
    updated_at: str


def enqueue_job(
    connection: sqlite3.Connection,
    *,
    job_id: str,
    kind: str,
    payload: dict[str, Any],
    corpus_id: str = "default",
) -> LLMJobRecord:
    """Insert a ``queued`` job, or return the existing record idempotently.

    Args:
        connection: Open SQLite connection.
        job_id: Caller-chosen stable identifier.
        kind: Logical job category used for observability + filtering.
        payload: JSON-serialisable payload describing the work to perform.
        corpus_id: Tenancy marker; defaults to the single-tenant value.

    Returns:
        The newly inserted record, or the pre-existing record if a job
        with the same ``job_id`` was already queued.

    Side Effects:
        Writes a row to ``llm_jobs`` under an ``INSERT ... ON CONFLICT DO
        NOTHING`` strategy, then reads it back for a consistent snapshot.
    """
    now = _now_iso()
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    with connection:
        connection.execute(
            """
            INSERT INTO llm_jobs
                (job_id, kind, corpus_id, status, payload_json,
                 result_json, error, attempts, created_at, updated_at)
            VALUES (?, ?, ?, 'queued', ?, NULL, NULL, 0, ?, ?)
            ON CONFLICT(job_id) DO NOTHING
            """,
            (job_id, kind, corpus_id, payload_json, now, now),
        )
    record = get_job(connection, job_id=job_id)
    if record is None:
        raise RuntimeError(f"Failed to enqueue llm_jobs row job_id={job_id!r}")
    return record


def get_job(connection: sqlite3.Connection, *, job_id: str) -> LLMJobRecord | None:
    """Fetch the ``llm_jobs`` row for ``job_id`` if it exists.

    Args:
        connection: Open SQLite connection.
        job_id: Identifier to look up.

    Returns:
        :class:`LLMJobRecord` snapshot, or ``None`` when no row exists.
    """
    row = connection.execute(
        """
        SELECT job_id, kind, corpus_id, status, payload_json,
               result_json, error, attempts, created_at, updated_at
        FROM llm_jobs WHERE job_id = ?
        """,
        (job_id,),
    ).fetchone()
    if row is None:
        return None
    return _record_from_row(row)


def list_jobs(
    connection: sqlite3.Connection,
    *,
    status: JobStatus | None = None,
    corpus_id: str | None = None,
    limit: int = 100,
) -> list[LLMJobRecord]:
    """Return recent jobs filtered by optional status/corpus.

    Args:
        connection: Open SQLite connection.
        status: When provided, only jobs in this state are returned.
        corpus_id: When provided, only jobs under this tenancy are returned.
        limit: Upper bound on returned rows (``1 <= limit <= 1000``).

    Returns:
        Jobs ordered by ``updated_at DESC``; most recently changed first.
    """
    if not 1 <= limit <= 1000:
        raise ValueError("limit must be between 1 and 1000")
    clauses: list[str] = []
    params: list[object] = []
    if status is not None:
        _require_valid_status(status)
        clauses.append("status = ?")
        params.append(status)
    if corpus_id is not None:
        clauses.append("corpus_id = ?")
        params.append(corpus_id)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    query = (
        "SELECT job_id, kind, corpus_id, status, payload_json, "
        "       result_json, error, attempts, created_at, updated_at "
        f"FROM llm_jobs {where} "
        "ORDER BY updated_at DESC LIMIT ?"
    )
    params.append(limit)
    rows = connection.execute(query, params).fetchall()
    return [_record_from_row(row) for row in rows]


def mark_running(connection: sqlite3.Connection, *, job_id: str, attempt: int) -> None:
    """Transition a job to ``running`` and record the current attempt count."""
    _update_status(
        connection,
        job_id=job_id,
        status="running",
        attempts=attempt,
        result=None,
        error=None,
    )


def mark_succeeded(
    connection: sqlite3.Connection,
    *,
    job_id: str,
    result: dict[str, Any] | None = None,
) -> None:
    """Transition a job to ``succeeded`` with an optional result payload."""
    _update_status(
        connection,
        job_id=job_id,
        status="succeeded",
        attempts=None,
        result=result,
        error=None,
    )


def mark_failed(connection: sqlite3.Connection, *, job_id: str, error: str) -> None:
    """Transition a job to ``failed`` and record a short error string."""
    _update_status(
        connection,
        job_id=job_id,
        status="failed",
        attempts=None,
        result=None,
        error=error,
    )


def mark_cancelled(connection: sqlite3.Connection, *, job_id: str) -> None:
    """Transition a job to ``cancelled``."""
    _update_status(
        connection,
        job_id=job_id,
        status="cancelled",
        attempts=None,
        result=None,
        error=None,
    )


def _update_status(
    connection: sqlite3.Connection,
    *,
    job_id: str,
    status: JobStatus,
    attempts: int | None,
    result: dict[str, Any] | None,
    error: str | None,
) -> None:
    _require_valid_status(status)
    now = _now_iso()
    result_json = (
        json.dumps(result, sort_keys=True, separators=(",", ":")) if result is not None else None
    )
    with connection:
        if attempts is None:
            connection.execute(
                """
                UPDATE llm_jobs
                   SET status = ?, result_json = ?, error = ?, updated_at = ?
                 WHERE job_id = ?
                """,
                (status, result_json, error, now, job_id),
            )
        else:
            connection.execute(
                """
                UPDATE llm_jobs
                   SET status = ?, attempts = ?, result_json = ?, error = ?, updated_at = ?
                 WHERE job_id = ?
                """,
                (status, attempts, result_json, error, now, job_id),
            )


def _record_from_row(row: sqlite3.Row | tuple[Any, ...]) -> LLMJobRecord:
    data = _row_mapping(row)
    payload_raw = data["payload_json"]
    result_raw = data["result_json"]
    payload = json.loads(payload_raw) if payload_raw else {}
    result = json.loads(result_raw) if result_raw else None
    return LLMJobRecord(
        job_id=str(data["job_id"]),
        kind=str(data["kind"]),
        corpus_id=str(data["corpus_id"]),
        status=_cast_status(str(data["status"])),
        payload=payload,
        result=result,
        error=str(data["error"]) if data["error"] is not None else None,
        attempts=int(data["attempts"]),
        created_at=str(data["created_at"]),
        updated_at=str(data["updated_at"]),
    )


def _row_mapping(row: sqlite3.Row | tuple[Any, ...]) -> dict[str, Any]:
    if isinstance(row, sqlite3.Row):
        return dict(row)
    keys = (
        "job_id",
        "kind",
        "corpus_id",
        "status",
        "payload_json",
        "result_json",
        "error",
        "attempts",
        "created_at",
        "updated_at",
    )
    return dict(zip(keys, row, strict=True))


def _cast_status(raw: str) -> JobStatus:
    _require_valid_status(raw)
    return raw  # type: ignore[return-value]


def _require_valid_status(status: str) -> None:
    if status not in _ALLOWED_STATUSES:
        raise ValueError(f"invalid llm_jobs status: {status!r}")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()
