"""Tests for `tool=<name>` contextvar binding in `run_tool` (B-STACK-8)."""

from __future__ import annotations

import io
import json

import anyio
import pytest
import structlog

from lxd.mcp.async_runtime import run_tool


@pytest.fixture
def captured_log_events() -> tuple[io.StringIO, structlog.types.WrappedLogger]:
    """Configure structlog to render JSON into a StringIO for assertion-friendly capture."""
    buffer = io.StringIO()
    structlog.contextvars.clear_contextvars()
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(0),
        logger_factory=structlog.PrintLoggerFactory(file=buffer),
        cache_logger_on_first_use=False,
    )
    log = structlog.get_logger("test_runtime_ctx")
    return buffer, log


def _events(buffer: io.StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in buffer.getvalue().splitlines() if line.strip()]


def test_log_inside_tool_carries_tool_contextvar(
    captured_log_events: tuple[io.StringIO, structlog.types.WrappedLogger],
) -> None:
    buffer, log = captured_log_events

    def _body() -> str:
        log.info("inside_tool")
        return "ok"

    async def _runner() -> str:
        return await run_tool("search_corpus", _body, timeout_secs=10)

    result = anyio.run(_runner)
    assert result == "ok"

    events = _events(buffer)
    assert any(
        e.get("event") == "inside_tool" and e.get("tool") == "search_corpus" for e in events
    ), f"Expected an `inside_tool` event tagged tool=search_corpus; saw {events}."


def test_tool_contextvar_cleared_after_run(
    captured_log_events: tuple[io.StringIO, structlog.types.WrappedLogger],
) -> None:
    buffer, log = captured_log_events

    def _body() -> None:
        return None

    async def _runner() -> None:
        await run_tool("search_corpus", _body, timeout_secs=10)

    anyio.run(_runner)
    log.info("after_tool")

    events = _events(buffer)
    after = [e for e in events if e.get("event") == "after_tool"]
    assert after, "Post-run log line should have been captured."
    for e in after:
        assert "tool" not in e, f"`tool` contextvar must not leak past run_tool exit; saw {e}."


def test_tool_contextvar_cleared_after_exception(
    captured_log_events: tuple[io.StringIO, structlog.types.WrappedLogger],
) -> None:
    buffer, log = captured_log_events

    def _body() -> None:
        raise RuntimeError("boom")

    async def _runner() -> None:
        await run_tool("failing_tool", _body, timeout_secs=10)

    with pytest.raises(RuntimeError, match="boom"):
        anyio.run(_runner)

    log.info("after_exception")

    events = _events(buffer)
    after = [e for e in events if e.get("event") == "after_exception"]
    assert after, "Post-exception log line should have been captured."
    for e in after:
        assert "tool" not in e, f"`tool` contextvar must be cleared even on exception; saw {e}."
