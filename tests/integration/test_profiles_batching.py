"""Regression tests for the Wave 5 N+1 fixes in ``ontology.profiles``."""

from __future__ import annotations

import sqlite3

import pytest

from lxd.ontology.profiles import _load_chunk_ids_by_entity


@pytest.fixture()
def mention_db(tmp_path) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE mention_rows (
            mention_id TEXT PRIMARY KEY,
            entity_id TEXT NOT NULL,
            chunk_id TEXT NOT NULL
        );
        INSERT INTO mention_rows VALUES
            ('m1', 'bloom_apply', 'chunk_a'),
            ('m2', 'bloom_apply', 'chunk_b'),
            ('m3', 'bloom_apply', 'chunk_a'),
            ('m4', 'bloom_remember', 'chunk_c'),
            ('m5', 'cognitive_load', 'chunk_b');
        """
    )
    try:
        yield conn
    finally:
        conn.close()


def test_load_chunk_ids_groups_by_entity(mention_db: sqlite3.Connection) -> None:
    """Each entity returns a sorted, de-duplicated list of chunk IDs."""

    grouped = _load_chunk_ids_by_entity(mention_db)

    assert grouped["bloom_apply"] == ["chunk_a", "chunk_b"]
    assert grouped["bloom_remember"] == ["chunk_c"]
    assert grouped["cognitive_load"] == ["chunk_b"]


def test_load_chunk_ids_empty_db() -> None:
    """An empty mention_rows table yields an empty mapping (not an error)."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE mention_rows (mention_id TEXT, entity_id TEXT, chunk_id TEXT);")

    try:
        assert _load_chunk_ids_by_entity(conn) == {}
    finally:
        conn.close()
