"""Provide managed SQLite connections for the LxD store.

Responsibility:
    Own the open/close lifecycle for ``sqlite3`` connections so callers never
    leak file descriptors on errors. Combines :func:`connect_sqlite` with
    :func:`ensure_schema` so every managed connection is schema-correct on
    entry.

Design boundary:
    Use :func:`open_store_connection` anywhere a caller would otherwise write
    ``conn = connect_sqlite(...); initialize_schema(conn); try: ... finally:
    conn.close()``. One-shot ad-hoc callers may keep the explicit form.

Key constraints:
    * The yielded connection is configured with the tuned pragmas from
      :func:`connect_sqlite` and is at schema version
      :data:`CURRENT_SCHEMA_VERSION`.
    * The context manager always closes the connection, even on exceptions.
    * Not thread-safe: each thread must acquire its own connection.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from lxd.stores.schema import ensure_schema


@contextmanager
def open_store_connection(
    sqlite_path: Path, *, ensure: bool = True
) -> Iterator[sqlite3.Connection]:
    """Yield an open, schema-correct SQLite connection and close it on exit.

    Args:
        sqlite_path: Filesystem path to the SQLite database file.
        ensure: When ``True`` (default), runs :func:`ensure_schema` before
            yielding so tables and indexes are guaranteed to exist. Set to
            ``False`` for read-only tools that must not mutate on-disk shape.

    Yields:
        A connection configured with the store's standard pragmas.

    Side Effects:
        Opens a SQLite connection; closes it on context exit, even if the
        caller raises.
    """
    from lxd.stores.sqlite import connect_sqlite

    connection = connect_sqlite(sqlite_path)
    try:
        if ensure:
            ensure_schema(connection)
        yield connection
    finally:
        connection.close()
