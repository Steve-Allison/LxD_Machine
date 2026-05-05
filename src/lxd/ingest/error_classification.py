"""Classify ingest-time exceptions so the pipeline can react proportionately.

Three classes:

- ``transient`` — network blips, rate limits, "service unavailable" errors.
  Worth retrying or treating as one-off file-level failures.
- ``data`` — bad source content: a malformed PDF, an empty file, an unparsable
  Markdown frontmatter. Per-file failure; continue to the next file.
- ``systemic`` — store-level failure (broken schema, missing table, disk
  full). Will fail identically for every subsequent file. Trip the
  circuit-breaker and abort.

The pipeline used to lump every recoverable error into a single bucket and
keep going. That is what burned API spend on every changed Research file when
the ghost FK was tripped: the first failure was systemic, but the pipeline
treated it as transient and re-paid 17 more times for the same failure.

Persistence: the breaker stores its consecutive-failure count in a
``circuit_breaker_state`` row keyed by ``scope``. A crashed process
mid-trip resumes at the last persisted count on the next start, so a
flaky run that already saw 2 systemic failures can trip on the very
next failure rather than starting fresh from zero.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from enum import Enum


class ErrorClass(Enum):
    """Severity / treatment class for an ingest-time exception."""

    TRANSIENT = "transient"
    DATA = "data"
    SYSTEMIC = "systemic"


_SYSTEMIC_SQLITE_PATTERNS = (
    "no such table",
    "no such column",
    "schema integrity check failed",
    "database is locked",
    "database disk image is malformed",
    "disk i/o error",
    "out of memory",
    "no space left on device",
)

_TRANSIENT_OS_ERRNOS = frozenset(
    {
        # Connection issues that look like network blips
        110,  # ETIMEDOUT
        111,  # ECONNREFUSED
        104,  # ECONNRESET
        # macOS equivalents
        60,  # ETIMEDOUT
        61,  # ECONNREFUSED
        54,  # ECONNRESET
    }
)


def classify(exc: BaseException) -> ErrorClass:
    """Return the class for ``exc``.

    Default is ``DATA`` — per-file failure, log and continue. Only SQL errors
    that look like they will repeat for every file get bumped to ``SYSTEMIC``.
    """
    if isinstance(exc, sqlite3.Error):
        message = str(exc).lower()
        if any(pat in message for pat in _SYSTEMIC_SQLITE_PATTERNS):
            return ErrorClass.SYSTEMIC
        if isinstance(exc, sqlite3.IntegrityError):
            # UNIQUE / CHECK / FK violations on insert are per-row data
            # problems. They do not repeat for every file in the corpus, so
            # they should not trip the systemic circuit-breaker.
            return ErrorClass.DATA
        # Anything else from sqlite3 (Operational/Database/Programming/
        # Interface/NotSupported) is store-level and will repeat for every
        # subsequent file: classify SYSTEMIC so the circuit-breaker can stop
        # the run before more API budget is spent.
        return ErrorClass.SYSTEMIC

    if isinstance(exc, OSError):
        if exc.errno in _TRANSIENT_OS_ERRNOS:
            return ErrorClass.TRANSIENT
        if isinstance(exc, FileNotFoundError):
            return ErrorClass.DATA
        return ErrorClass.SYSTEMIC

    if isinstance(exc, RuntimeError):
        message = str(exc).lower()
        if "rate limit" in message or "timed out" in message or "timeout" in message:
            return ErrorClass.TRANSIENT
        if "schema integrity" in message:
            return ErrorClass.SYSTEMIC
        return ErrorClass.DATA

    if isinstance(exc, ValueError):
        return ErrorClass.DATA

    return ErrorClass.SYSTEMIC


class CircuitBreakerTripped(RuntimeError):
    """Raised by :class:`PersistentCircuitBreaker` after threshold is hit.

    Contains the last exception so the operator sees what went wrong instead
    of a generic "aborted" message.
    """

    def __init__(self, count: int, last_error: BaseException) -> None:
        super().__init__(
            f"Aborting ingest: {count} consecutive systemic errors. "
            f"Last error: {type(last_error).__name__}: {last_error}"
        )
        self.count = count
        self.last_error = last_error


_DEFAULT_SCOPE = "ingest_default"


class PersistentCircuitBreaker:
    """Counts consecutive systemic errors with state persisted to SQLite.

    State lives in the ``circuit_breaker_state`` table keyed by
    ``scope`` so a crashed process resumes at the last-known count on
    the next start (rather than starting fresh from zero and re-paying
    on the same systemic failure pattern). Successes reset the counter.
    Transient and data errors do not advance it — they are normal
    per-file failures that should not abort a long run.

    Thread safety: writes happen inside the connection's autocommit-on-
    ``execute`` model with explicit ``commit()``; the breaker is intended
    for single-source-loop ingest. Concurrent ingest must guard externally.
    """

    __slots__ = ("_connection", "_consecutive", "_scope", "_threshold")

    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        scope: str = _DEFAULT_SCOPE,
        threshold: int = 3,
    ) -> None:
        if threshold < 1:
            raise ValueError("threshold must be >= 1")
        self._connection = connection
        self._scope = scope
        self._threshold = threshold
        self._consecutive = self._load_consecutive()

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive

    @property
    def threshold(self) -> int:
        return self._threshold

    def record_success(self) -> None:
        """Reset the counter and persist the success timestamp."""
        self._consecutive = 0
        now = datetime.now(UTC).isoformat()
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO circuit_breaker_state (
                    scope, consecutive_failures, last_success_at, tripped_at
                )
                VALUES (?, 0, ?, NULL)
                ON CONFLICT(scope) DO UPDATE SET
                    consecutive_failures = 0,
                    last_success_at = excluded.last_success_at,
                    tripped_at = NULL
                """,
                (self._scope, now),
            )

    def record_failure(self, exc: BaseException) -> None:
        """Record an exception. Persists the new state and raises
        :class:`CircuitBreakerTripped` once the threshold of consecutive
        systemic failures is reached.
        """
        cls = classify(exc)
        if cls is not ErrorClass.SYSTEMIC:
            # Non-systemic errors don't advance the counter and don't reset
            # it either — a transient blip in the middle of a systemic run
            # shouldn't mask the pattern. No persistence needed.
            return
        self._consecutive += 1
        now = datetime.now(UTC).isoformat()
        tripped_at = now if self._consecutive >= self._threshold else None
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO circuit_breaker_state (
                    scope, consecutive_failures, last_error_class,
                    last_error_message, last_error_type, last_failure_at,
                    tripped_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(scope) DO UPDATE SET
                    consecutive_failures = excluded.consecutive_failures,
                    last_error_class = excluded.last_error_class,
                    last_error_message = excluded.last_error_message,
                    last_error_type = excluded.last_error_type,
                    last_failure_at = excluded.last_failure_at,
                    tripped_at = COALESCE(
                        excluded.tripped_at, circuit_breaker_state.tripped_at
                    )
                """,
                (
                    self._scope,
                    self._consecutive,
                    cls.value,
                    str(exc)[:1000],
                    type(exc).__name__,
                    now,
                    tripped_at,
                ),
            )
        if self._consecutive >= self._threshold:
            raise CircuitBreakerTripped(self._consecutive, exc)

    def _load_consecutive(self) -> int:
        row = self._connection.execute(
            "SELECT consecutive_failures FROM circuit_breaker_state WHERE scope = ?",
            (self._scope,),
        ).fetchone()
        if row is None:
            return 0
        return int(row[0])


def reset_circuit_breaker(connection: sqlite3.Connection, *, scope: str = _DEFAULT_SCOPE) -> None:
    """Clear the persisted breaker row for ``scope``.

    Use after manually resolving a tripped breaker (the underlying
    systemic failure has been fixed) so the next ingest run starts fresh.
    """
    with connection:
        connection.execute("DELETE FROM circuit_breaker_state WHERE scope = ?", (scope,))
