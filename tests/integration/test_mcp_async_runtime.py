"""Regression tests for ``lxd.mcp.async_runtime.run_tool``.

These tests lock in the Wave 4 contract:

* blocking callables are offloaded to a worker thread so they do not stall
  the event loop;
* a hard timeout raises :class:`TimeoutError` rather than hanging the
  server process;
* a callable that raises has its exception propagated unchanged.
"""

from __future__ import annotations

import time

import anyio
import pytest

from lxd.mcp.async_runtime import run_tool


def test_run_tool_returns_callable_result() -> None:
    """A trivial callable returns its value via the async wrapper."""

    async def _run() -> int:
        return await run_tool("noop", lambda: 42, timeout_secs=5.0)

    assert anyio.run(_run) == 42


def test_run_tool_enforces_timeout() -> None:
    """A callable that exceeds the budget raises ``TimeoutError``."""

    def _slow() -> None:
        time.sleep(1.0)

    async def _run() -> None:
        await run_tool("slow", _slow, timeout_secs=0.05)

    with pytest.raises(TimeoutError):
        anyio.run(_run)


def test_run_tool_propagates_exceptions() -> None:
    """Exceptions raised inside the callable surface to the awaiter."""

    class _Boom(RuntimeError):
        pass

    def _raiser() -> None:
        raise _Boom("nope")

    async def _run() -> None:
        await run_tool("boom", _raiser, timeout_secs=1.0)

    with pytest.raises(_Boom):
        anyio.run(_run)


def test_run_tool_disabled_timeout_still_runs() -> None:
    """Passing ``timeout_secs<=0`` disables the deadline."""

    async def _run() -> str:
        return await run_tool("no-budget", lambda: "ok", timeout_secs=0.0)

    assert anyio.run(_run) == "ok"
