"""Round-trip persistence test for the new wiki-metadata columns on chunks.

Verifies that ``cited_sources`` and ``wiki_links`` survive a write-read cycle
through both SQLite (``chunk_rows``) and LanceDB (``chunk_vectors``), and
that the canonical-row view (:func:`chunk_from_row`) reconstructs the lists
into typed tuples.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from lxd.stores.lancedb import (
    connect_lancedb,
    open_chunk_table,
    replace_source_chunks,
    search_chunks,
)
from lxd.stores.models import ChunkRecord, ManifestRecord
from lxd.stores.schema import ensure_schema
from lxd.stores.sqlite import (
    connect_sqlite,
    load_chunk_records_for_source,
    upsert_manifest_record,
)
from lxd.stores.sqlite import (
    replace_source_chunks as sqlite_replace_source_chunks,
)


def _make_chunk(
    *,
    chunk_id: str,
    cited_sources: tuple[str, ...] = (),
    wiki_links: tuple[str, ...] = (),
) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        document_id="doc-a",
        source_rel_path="addie-model.md",
        source_filename="addie-model.md",
        source_type="markdown",
        source_domain="wiki",
        source_hash="h1",
        citation_label="addie-model.md#0",
        chunk_index=0,
        chunk_occurrence=0,
        token_count=10,
        text="ADDIE is a five-phase model.",
        chunk_hash="ch1",
        score_hint="normal",
        metadata_json="{}",
        vector=[0.1, 0.2, 0.3],
        embedding_model="m",
        embedding_dims=3,
        cited_sources=cited_sources,
        wiki_links=wiki_links,
    )


def _seed_manifest(connection: sqlite3.Connection, source_rel_path: str) -> None:
    upsert_manifest_record(
        connection,
        ManifestRecord(
            source_rel_path=source_rel_path,
            absolute_path=f"/abs/{source_rel_path}",
            source_type="markdown",
            source_domain="wiki",
            document_id="doc-a",
            file_size_bytes=1,
            content_hash="h1",
            parent_source_rel_path=None,
            chunk_count=1,
            last_seen_at="2026-05-01",
            last_processed_at="2026-05-01",
            last_committed_at="2026-05-01",
            error_message=None,
            lifecycle_status="complete",
            retrieval_status="searchable",
        ),
    )


def test_sqlite_round_trip_preserves_cited_sources_and_wiki_links(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "store.sqlite3"
    connection = connect_sqlite(sqlite_path)
    try:
        ensure_schema(connection)
        _seed_manifest(connection, "addie-model.md")
        chunk = _make_chunk(
            chunk_id="c1",
            cited_sources=("theory_addie_model.md", "2025 Learning Experience (LX) Design.pdf"),
            wiki_links=("backward-design", "kirkpatricks-evaluation-model"),
        )
        sqlite_replace_source_chunks(
            connection,
            source_rel_path="addie-model.md",
            chunk_records=[chunk],
            mention_records=[],
        )
        loaded = load_chunk_records_for_source(connection, "addie-model.md")
        assert len(loaded) == 1
        assert loaded[0].cited_sources == (
            "theory_addie_model.md",
            "2025 Learning Experience (LX) Design.pdf",
        )
        assert loaded[0].wiki_links == (
            "backward-design",
            "kirkpatricks-evaluation-model",
        )
    finally:
        connection.close()


def test_sqlite_default_empty_for_chunks_without_wiki_metadata(tmp_path: Path) -> None:
    """Chunks from non-wiki sources persist with empty tuples — the
    LLM-extraction layer must continue to work as before."""
    sqlite_path = tmp_path / "store.sqlite3"
    connection = connect_sqlite(sqlite_path)
    try:
        ensure_schema(connection)
        _seed_manifest(connection, "addie-model.md")
        chunk = _make_chunk(chunk_id="c2")  # no wiki metadata
        sqlite_replace_source_chunks(
            connection,
            source_rel_path="addie-model.md",
            chunk_records=[chunk],
            mention_records=[],
        )
        loaded = load_chunk_records_for_source(connection, "addie-model.md")
        assert loaded[0].cited_sources == ()
        assert loaded[0].wiki_links == ()
    finally:
        connection.close()


def test_lancedb_round_trip_preserves_wiki_metadata(tmp_path: Path) -> None:
    """LanceDB chunk_vectors carries the same fields as SQLite so retrieval
    can surface them without a SQLite hop."""
    db = connect_lancedb(tmp_path / "lancedb")
    table = open_chunk_table(db, vector_size=3)
    chunk = _make_chunk(
        chunk_id="c-lance",
        cited_sources=("theory_x.md",),
        wiki_links=("alpha", "beta"),
    )
    replace_source_chunks(table, "addie-model.md", [chunk])
    hits = search_chunks(table, query_vector=[0.1, 0.2, 0.3], domain=None, limit=1)
    assert len(hits) == 1
    assert hits[0].cited_sources == ("theory_x.md",)
    assert hits[0].wiki_links == ("alpha", "beta")


def test_lancedb_round_trip_handles_empty_wiki_metadata(tmp_path: Path) -> None:
    db = connect_lancedb(tmp_path / "lancedb")
    table = open_chunk_table(db, vector_size=3)
    chunk = _make_chunk(chunk_id="c-empty")
    replace_source_chunks(table, "addie-model.md", [chunk])
    hits = search_chunks(table, query_vector=[0.1, 0.2, 0.3], domain=None, limit=1)
    assert hits[0].cited_sources == ()
    assert hits[0].wiki_links == ()
