"""Regression tests for Wave 2 schema versioning and lifespan-owned connections.

Covers:

1. ``ensure_schema`` stamps ``PRAGMA user_version`` to the current version
   when run against a fresh database.
2. ``ensure_schema`` is a no-op (still at ``CURRENT_SCHEMA_VERSION``) when run
   twice in succession.
3. ``initialize_schema`` (the compatibility wrapper) still advances the
   version, so pre-existing callers keep working.
4. ``open_store_connection`` yields a connection, runs ``ensure_schema``, and
   closes the connection on context exit even if the caller raises.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from lxd.stores.connection import open_store_connection
from lxd.stores.schema import CURRENT_SCHEMA_VERSION, ensure_schema, get_schema_version
from lxd.stores.sqlite import build_store_paths, connect_sqlite, initialize_schema


def test_ensure_schema_stamps_user_version(tmp_path: Path) -> None:
    """Fresh databases advance to ``CURRENT_SCHEMA_VERSION`` on first contact."""
    store_paths = build_store_paths(tmp_path)
    connection = connect_sqlite(store_paths.sqlite_path)
    try:
        assert get_schema_version(connection) == 0
        ensure_schema(connection)
        assert get_schema_version(connection) == CURRENT_SCHEMA_VERSION
    finally:
        connection.close()


def test_ensure_schema_is_idempotent(tmp_path: Path) -> None:
    """Running ``ensure_schema`` twice leaves the version unchanged."""
    store_paths = build_store_paths(tmp_path)
    connection = connect_sqlite(store_paths.sqlite_path)
    try:
        ensure_schema(connection)
        first = get_schema_version(connection)
        ensure_schema(connection)
        second = get_schema_version(connection)
    finally:
        connection.close()

    assert first == second == CURRENT_SCHEMA_VERSION


def test_initialize_schema_still_advances_user_version(tmp_path: Path) -> None:
    """The legacy ``initialize_schema`` entrypoint must stamp the version too."""
    store_paths = build_store_paths(tmp_path)
    connection = connect_sqlite(store_paths.sqlite_path)
    try:
        initialize_schema(connection)
        assert get_schema_version(connection) == CURRENT_SCHEMA_VERSION
    finally:
        connection.close()


def test_open_store_connection_closes_on_exception(tmp_path: Path) -> None:
    """Context manager must close the connection even when the caller raises."""
    store_paths = build_store_paths(tmp_path)

    captured: list[object] = []
    with (
        pytest.raises(RuntimeError),
        open_store_connection(store_paths.sqlite_path) as connection,
    ):
        captured.append(connection)
        raise RuntimeError("caller error")

    connection = captured[0]
    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1")  # type: ignore[attr-defined]
