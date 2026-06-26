"""Tests for the error-classification + persistent circuit-breaker module."""

import sqlite3
from pathlib import Path

import pytest

from lxd.ingest.error_classification import (
    CircuitBreakerTripped,
    ErrorClass,
    PersistentCircuitBreaker,
    classify,
    reset_circuit_breaker,
)
from lxd.stores.schema import ensure_schema
from lxd.stores.sqlite.connection import connect_sqlite


def _open_with_schema(path: Path) -> sqlite3.Connection:
    connection = connect_sqlite(path)
    ensure_schema(connection)
    return connection


def test_classify_no_such_table_is_systemic() -> None:
    err = sqlite3.OperationalError("no such table: main.chunk_rows_v2_legacy")
    assert classify(err) is ErrorClass.SYSTEMIC


def test_classify_integrity_error_is_data_not_systemic() -> None:
    """UNIQUE / FK / CHECK constraint violations are per-row failures.

    Regression test: previously classify() returned SYSTEMIC for every
    sqlite3.Error subclass, which meant 3 duplicate-INSERT errors in a row
    (entirely plausible during normal corpus operation) would trip the
    circuit-breaker and abort the run.
    """
    assert classify(sqlite3.IntegrityError("UNIQUE constraint failed: x.y")) is ErrorClass.DATA
    assert classify(sqlite3.IntegrityError("FOREIGN KEY constraint failed")) is ErrorClass.DATA
    assert classify(sqlite3.IntegrityError("CHECK constraint failed")) is ErrorClass.DATA


def test_classify_other_sqlite_errors_remain_systemic() -> None:
    """Non-Integrity sqlite3 errors should still classify as SYSTEMIC."""
    assert classify(sqlite3.OperationalError("disk i/o error")) is ErrorClass.SYSTEMIC
    assert (
        classify(sqlite3.DatabaseError("database disk image is malformed")) is ErrorClass.SYSTEMIC
    )
    assert classify(sqlite3.ProgrammingError("incorrect bindings")) is ErrorClass.SYSTEMIC


def test_classify_value_error_is_data() -> None:
    assert classify(ValueError("malformed")) is ErrorClass.DATA


def test_classify_file_not_found_is_data() -> None:
    assert classify(FileNotFoundError("missing")) is ErrorClass.DATA


def test_classify_runtime_rate_limit_is_transient() -> None:
    assert classify(RuntimeError("rate limit hit")) is ErrorClass.TRANSIENT


def test_persistent_breaker_trips_on_3rd_consecutive_systemic(tmp_path: Path) -> None:
    connection = _open_with_schema(tmp_path / "lxd.sqlite3")
    try:
        breaker = PersistentCircuitBreaker(connection, threshold=3)
        breaker.record_failure(sqlite3.OperationalError("no such table: x"))
        breaker.record_failure(sqlite3.OperationalError("no such table: x"))
        with pytest.raises(CircuitBreakerTripped) as exc_info:
            breaker.record_failure(sqlite3.OperationalError("no such table: x"))
        assert exc_info.value.count == 3
    finally:
        connection.close()


def test_persistent_breaker_resets_on_success(tmp_path: Path) -> None:
    connection = _open_with_schema(tmp_path / "lxd.sqlite3")
    try:
        breaker = PersistentCircuitBreaker(connection, threshold=3)
        breaker.record_failure(sqlite3.OperationalError("no such table: x"))
        breaker.record_failure(sqlite3.OperationalError("no such table: x"))
        breaker.record_success()
        breaker.record_failure(sqlite3.OperationalError("no such table: x"))
        breaker.record_failure(sqlite3.OperationalError("no such table: x"))
        # 2 consecutive after success — must not have tripped.
        assert breaker.consecutive_failures == 2
    finally:
        connection.close()


def test_persistent_breaker_does_not_advance_on_data_errors(tmp_path: Path) -> None:
    connection = _open_with_schema(tmp_path / "lxd.sqlite3")
    try:
        breaker = PersistentCircuitBreaker(connection, threshold=3)
        breaker.record_failure(ValueError("bad data"))
        breaker.record_failure(FileNotFoundError("gone"))
        breaker.record_failure(ValueError("bad data again"))
        assert breaker.consecutive_failures == 0
    finally:
        connection.close()


def test_persistent_breaker_state_survives_process_restart(tmp_path: Path) -> None:
    """A crashed pid mid-trip resumes at the persisted count on the
    next start — the whole point of persistence."""
    db_path = tmp_path / "lxd.sqlite3"
    first = _open_with_schema(db_path)
    try:
        breaker = PersistentCircuitBreaker(first, threshold=3)
        breaker.record_failure(sqlite3.OperationalError("no such table: x"))
        breaker.record_failure(sqlite3.OperationalError("no such table: x"))
        # Process "crashes" — close the connection without recording success.
    finally:
        first.close()

    second = connect_sqlite(db_path)
    try:
        ensure_schema(second)
        # New process. The breaker reads its state on construct.
        resumed = PersistentCircuitBreaker(second, threshold=3)
        assert resumed.consecutive_failures == 2
        with pytest.raises(CircuitBreakerTripped):
            resumed.record_failure(sqlite3.OperationalError("no such table: x"))
    finally:
        second.close()


def test_persistent_breaker_scopes_are_independent(tmp_path: Path) -> None:
    """Different scopes share the table but track independently."""
    connection = _open_with_schema(tmp_path / "lxd.sqlite3")
    try:
        ingest_breaker = PersistentCircuitBreaker(connection, scope="ingest", threshold=3)
        graph_breaker = PersistentCircuitBreaker(connection, scope="graph", threshold=3)
        ingest_breaker.record_failure(sqlite3.OperationalError("no such table: x"))
        ingest_breaker.record_failure(sqlite3.OperationalError("no such table: x"))
        assert graph_breaker.consecutive_failures == 0
        assert ingest_breaker.consecutive_failures == 2
    finally:
        connection.close()


def test_reset_circuit_breaker_clears_state(tmp_path: Path) -> None:
    """`reset_circuit_breaker(connection, scope=...)` removes the row so
    the next breaker construct starts fresh from zero."""
    db_path = tmp_path / "lxd.sqlite3"
    connection = _open_with_schema(db_path)
    try:
        breaker = PersistentCircuitBreaker(connection, threshold=3)
        breaker.record_failure(sqlite3.OperationalError("no such table: x"))
        breaker.record_failure(sqlite3.OperationalError("no such table: x"))
        reset_circuit_breaker(connection)
        fresh = PersistentCircuitBreaker(connection, threshold=3)
        assert fresh.consecutive_failures == 0
    finally:
        connection.close()


def test_persistent_breaker_rejects_zero_threshold(tmp_path: Path) -> None:
    connection = _open_with_schema(tmp_path / "lxd.sqlite3")
    try:
        with pytest.raises(ValueError, match="threshold must be >= 1"):
            PersistentCircuitBreaker(connection, threshold=0)
    finally:
        connection.close()
