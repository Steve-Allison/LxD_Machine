"""Tests for the error-classification + circuit-breaker module."""

from __future__ import annotations

import sqlite3

import pytest

from lxd.ingest.error_classification import (
    CircuitBreakerTripped,
    ErrorClass,
    SystemicErrorCircuitBreaker,
    classify,
)


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


def test_circuit_breaker_trips_on_3rd_consecutive_systemic() -> None:
    breaker = SystemicErrorCircuitBreaker(threshold=3)
    breaker.record_failure(sqlite3.OperationalError("no such table: x"))
    breaker.record_failure(sqlite3.OperationalError("no such table: x"))
    with pytest.raises(CircuitBreakerTripped) as exc_info:
        breaker.record_failure(sqlite3.OperationalError("no such table: x"))
    assert exc_info.value.count == 3


def test_circuit_breaker_resets_on_success() -> None:
    breaker = SystemicErrorCircuitBreaker(threshold=3)
    breaker.record_failure(sqlite3.OperationalError("no such table: x"))
    breaker.record_failure(sqlite3.OperationalError("no such table: x"))
    breaker.record_success()
    breaker.record_failure(sqlite3.OperationalError("no such table: x"))
    breaker.record_failure(sqlite3.OperationalError("no such table: x"))
    # 2 consecutive after success — must not have tripped.
    assert breaker.consecutive_failures == 2


def test_circuit_breaker_does_not_advance_on_data_errors() -> None:
    breaker = SystemicErrorCircuitBreaker(threshold=3)
    breaker.record_failure(ValueError("bad data"))
    breaker.record_failure(FileNotFoundError("gone"))
    breaker.record_failure(ValueError("bad data again"))
    # Three data errors in a row — counter should not have advanced.
    assert breaker.consecutive_failures == 0
