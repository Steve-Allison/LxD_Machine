"""Per-thread SQLite connection pool for the long-lived MCP request path.

The MCP server is long-lived: every tool call previously called
:func:`connect_sqlite` and re-applied the WAL/timeout/cache pragmas. For
one-shot CLI commands that's fine, but at request rate the pragma overhead
adds up. This pool amortises the connect-and-pragma cost by retaining one
schema-initialised :class:`sqlite3.Connection` per (thread, path) pair.

SQLite connections default to ``check_same_thread=True`` — keying on
thread identity matches that constraint. FastMCP dispatches tool bodies
through ``asyncio.to_thread`` (see :mod:`lxd.mcp.async_runtime`), whose
underlying executor reuses worker threads, so the pool stays warm.
"""

import contextlib
import sqlite3
import threading
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from lxd.stores.sqlite.connection import connect_sqlite, initialize_schema

_pool: dict[tuple[int, str], sqlite3.Connection] = {}
_lock = threading.Lock()


@contextmanager
def pooled_connection(path: Path) -> Generator[sqlite3.Connection]:
    """Yield a per-thread schema-initialised SQLite connection for ``path``.

    On first use per ``(thread, resolved-path)`` key the connection is
    created via :func:`connect_sqlite` and the schema is migrated via
    :func:`initialize_schema`. Subsequent calls in the same thread reuse
    the same connection without re-applying pragmas or running migrations.

    The connection is **not** closed on context exit — the pool keeps it
    alive for the thread's lifetime so the next request amortises both
    the connect-and-pragma cost and the user_version probe in
    ``ensure_schema``. On an uncaught exception inside the ``with`` block
    any in-flight transaction is rolled back so the next caller sees a
    clean session; the connection itself stays pooled.

    Args:
        path: Absolute or relative path to the SQLite database file.

    Yields:
        Pooled :class:`sqlite3.Connection`, owned by the current thread.
    """
    key = (threading.get_ident(), str(path.resolve()))
    with _lock:
        connection = _pool.get(key)
        if connection is None:
            connection = connect_sqlite(path)
            initialize_schema(connection)
            _pool[key] = connection
    try:
        yield connection
    except Exception:
        with contextlib.suppress(sqlite3.Error):
            connection.rollback()
        raise


def reset_pool() -> None:
    """Close every pooled connection and clear the pool.

    Intended for tests; production code relies on per-process lifetime.
    """
    with _lock:
        for connection in _pool.values():
            with contextlib.suppress(sqlite3.Error):
                connection.close()
        _pool.clear()
