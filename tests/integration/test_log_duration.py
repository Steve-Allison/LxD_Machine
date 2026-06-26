"""Regression tests for the ``log_duration`` context manager (Wave 8)."""

from typing import Any, cast

import pytest
import structlog
from structlog.stdlib import BoundLogger

from lxd.observability.logging import log_duration


class _SpyLogger:
    """Capture structlog-style calls for assertions."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict[str, Any]]] = []

    def info(self, event: str, **fields: Any) -> None:
        self.events.append(("info", event, fields))

    def error(self, event: str, **fields: Any) -> None:
        self.events.append(("error", event, fields))

    def debug(self, event: str, **fields: Any) -> None:
        self.events.append(("debug", event, fields))


def test_log_duration_emits_started_and_completed() -> None:
    """A clean block produces a matched ``started``/``completed`` pair."""
    spy = _SpyLogger()

    with log_duration("ingest.run", logger=cast("BoundLogger", spy), tenant="acme"):
        pass

    assert [event for _, event, _ in spy.events] == [
        "ingest.run.started",
        "ingest.run.completed",
    ]
    assert spy.events[0][2] == {"tenant": "acme"}
    assert spy.events[1][2]["tenant"] == "acme"
    assert "duration_ms" in spy.events[1][2]
    assert spy.events[1][2]["duration_ms"] >= 0


def test_log_duration_reports_failure_and_reraises() -> None:
    """Exceptions produce a ``failed`` record and propagate unchanged."""
    spy = _SpyLogger()

    with (
        pytest.raises(ValueError, match="boom"),
        log_duration("ingest.run", logger=cast("BoundLogger", spy)),
    ):
        raise ValueError("boom")

    events = [event for _, event, _ in spy.events]
    assert events == ["ingest.run.started", "ingest.run.failed"]
    failed = spy.events[1]
    assert failed[0] == "error"
    assert failed[2]["error_class"] == "ValueError"
    assert "duration_ms" in failed[2]


def test_log_duration_mutable_fields_survive_to_completion() -> None:
    """Fields added mid-block are visible on the completion event."""
    spy = _SpyLogger()

    with log_duration("ingest.run", logger=cast("BoundLogger", spy)) as extras:
        extras["files_processed"] = 42

    assert spy.events[1][2]["files_processed"] == 42


def test_log_duration_accepts_real_structlog_logger() -> None:
    """Contract check: the helper accepts a real structlog BoundLogger too."""
    real_logger = structlog.get_logger(__name__)
    with log_duration("smoke.test", logger=real_logger):
        pass
