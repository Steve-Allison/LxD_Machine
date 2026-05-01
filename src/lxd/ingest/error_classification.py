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
"""

from __future__ import annotations

import sqlite3
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
    """Raised by :class:`SystemicErrorCircuitBreaker` after threshold is hit.

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


class SystemicErrorCircuitBreaker:
    """Counts consecutive systemic errors and trips after a threshold.

    Successes reset the counter. Transient and data errors do not advance
    the counter — they are normal per-file failures that should not abort a
    long run.
    """

    def __init__(self, threshold: int = 3) -> None:
        if threshold < 1:
            raise ValueError("threshold must be >= 1")
        self._threshold = threshold
        self._consecutive = 0

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive

    def record_success(self) -> None:
        self._consecutive = 0

    def record_failure(self, exc: BaseException) -> None:
        """Record an exception. Raises :class:`CircuitBreakerTripped` if the
        threshold of consecutive systemic failures is reached.
        """
        cls = classify(exc)
        if cls is ErrorClass.SYSTEMIC:
            self._consecutive += 1
            if self._consecutive >= self._threshold:
                raise CircuitBreakerTripped(self._consecutive, exc)
        else:
            # Non-systemic errors don't advance the counter, but they don't
            # reset it either — a transient blip in the middle of a systemic
            # run shouldn't mask the pattern.
            return
