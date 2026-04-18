"""Run synchronous MCP tool bodies inside the async server loop safely.

Responsibility:
    Centralise the bridge between FastMCP's async tool API and the still
    largely synchronous ``lxd.stores.sqlite`` / ``lxd.stores.lancedb`` stack.
    Callers wrap a blocking callable in :func:`run_tool` and get timeout,
    thread-pool offload, and uniform structured logging for free.

Design boundary:
    Only MCP server bindings should import this module. Stores, retrieval,
    and ingest code must remain synchronous until Wave 5+ migrate them to
    truly-async equivalents.

Key constraints:
    * Each call runs in a worker thread, so the MCP event loop stays
      responsive even when LanceDB or SQLite do disk I/O.
    * A tool that does not finish within ``tool_timeout_secs`` raises
      :class:`TimeoutError`, which FastMCP surfaces as an MCP error to the
      client rather than hanging the session. The worker thread is detached
      (``abandon_on_cancel=True``) so the event loop can return immediately;
      callers must accept that an in-flight SQLite/LanceDB call may continue
      briefly in the background before noticing it has no readers.
    * Exceptions are logged with ``exc_info=True`` so stack traces are never
      silently swallowed.
"""

from __future__ import annotations

from collections.abc import Callable

import anyio
import anyio.to_thread
import structlog

_log = structlog.get_logger(__name__)


async def run_tool[T](
    name: str,
    func: Callable[[], T],
    *,
    timeout_secs: float,
) -> T:
    """Execute ``func`` in a worker thread with a hard timeout.

    Args:
        name: Human-readable tool identifier; used for logging.
        func: Zero-argument callable that performs the synchronous work.
        timeout_secs: Hard upper bound on wall-clock duration. Use ``0`` or
            a negative value to disable the timeout.

    Returns:
        Whatever ``func`` returns.

    Raises:
        TimeoutError: If ``func`` does not complete within ``timeout_secs``.
        Exception: Any exception raised by ``func`` is propagated unchanged
            (after being logged with ``exc_info=True``).
    """
    if timeout_secs <= 0:
        return await anyio.to_thread.run_sync(func)

    try:
        with anyio.fail_after(timeout_secs):
            return await anyio.to_thread.run_sync(func, abandon_on_cancel=True)
    except TimeoutError:
        _log.warning("mcp.tool.timeout", tool=name, timeout_secs=timeout_secs)
        raise
    except Exception:
        _log.error("mcp.tool.error", tool=name, exc_info=True)
        raise
