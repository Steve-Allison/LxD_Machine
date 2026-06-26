"""Regression tests for the Wave 11 persistent LLM job queue."""

import sqlite3
from collections.abc import Iterator
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from lxd.stores.llm_jobs import (
    LLMJobRecord,
    enqueue_job,
    get_job,
    list_jobs,
    mark_failed,
    mark_running,
    mark_succeeded,
)
from lxd.stores.schema import ensure_schema


@pytest.fixture
def db(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    """Open an isolated, migrated SQLite database for each test."""
    conn = sqlite3.connect(tmp_path / "lxd.sqlite3")
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)
    try:
        yield conn
    finally:
        conn.close()


def test_enqueue_is_idempotent(db: sqlite3.Connection) -> None:
    """Re-enqueueing the same ``job_id`` returns the original record."""
    first = enqueue_job(db, job_id="j1", kind="claims", payload={"x": 1})
    second = enqueue_job(db, job_id="j1", kind="claims", payload={"x": 2})
    assert first.job_id == second.job_id == "j1"
    assert first.payload == {"x": 1}
    assert second.payload == {"x": 1}, "existing payload must not be overwritten"


def test_job_status_lifecycle(db: sqlite3.Connection) -> None:
    """Running -> succeeded transitions update attempts and result payload."""
    enqueue_job(db, job_id="j2", kind="relations", payload={"n": 3})
    mark_running(db, job_id="j2", attempt=1)
    running = get_job(db, job_id="j2")
    assert running is not None
    assert running.status == "running"
    assert running.attempts == 1

    mark_succeeded(db, job_id="j2", result={"ok": True})
    done = get_job(db, job_id="j2")
    assert done is not None
    assert done.status == "succeeded"
    assert done.result == {"ok": True}
    assert done.error is None


def test_mark_failed_records_error_message(db: sqlite3.Connection) -> None:
    """Failure transitions preserve a short error string."""
    enqueue_job(db, job_id="j3", kind="relations", payload={})
    mark_failed(db, job_id="j3", error="boom")
    failed = get_job(db, job_id="j3")
    assert failed is not None
    assert failed.status == "failed"
    assert failed.error == "boom"


def test_list_jobs_filters_by_status_and_corpus(db: sqlite3.Connection) -> None:
    """``list_jobs`` respects optional status and corpus_id filters."""
    enqueue_job(db, job_id="a", kind="k", payload={}, corpus_id="tenant_a")
    enqueue_job(db, job_id="b", kind="k", payload={}, corpus_id="tenant_b")
    enqueue_job(db, job_id="c", kind="k", payload={}, corpus_id="tenant_a")
    mark_succeeded(db, job_id="a")

    queued_a = list_jobs(db, status="queued", corpus_id="tenant_a")
    assert [j.job_id for j in queued_a] == ["c"]

    all_a = list_jobs(db, corpus_id="tenant_a")
    assert {j.job_id for j in all_a} == {"a", "c"}


def test_get_job_returns_none_for_missing_ids(db: sqlite3.Connection) -> None:
    """Unknown job_ids resolve to ``None`` rather than raising."""
    assert get_job(db, job_id="nope") is None


def test_record_is_frozen_dataclass(db: sqlite3.Connection) -> None:
    """Records are frozen so callers cannot drift from the DB row."""
    enqueue_job(db, job_id="j", kind="k", payload={})
    record = get_job(db, job_id="j")
    assert isinstance(record, LLMJobRecord)
    with pytest.raises(FrozenInstanceError):
        record.job_id = "other"  # type: ignore[misc]
