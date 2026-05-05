from __future__ import annotations

from lxd.stores.lancedb import (
    connect_lancedb,
    replace_source_chunks,
    reset_chunk_table,
    search_chunks,
)
from lxd.stores.models import ChunkRecord


def test_lancedb_search_and_domain_filter(tmp_path) -> None:
    database = connect_lancedb(tmp_path / "lancedb")
    table = reset_chunk_table(database, vector_size=3)
    replace_source_chunks(
        table,
        "Guides/example.md",
        [
            ChunkRecord(
                chunk_id="chunk-guides",
                document_id="doc-guides",
                source_rel_path="Guides/example.md",
                source_filename="example.md",
                source_type="markdown",
                source_domain="guides",
                source_hash="hash-guides-source",
                citation_label="Guides/example.md",
                chunk_index=0,
                chunk_occurrence=0,
                token_count=2,
                text="Guide text",
                chunk_hash="hash-guides",
                score_hint="Guide text",
                metadata_json="{}",
                vector=[1.0, 0.0, 0.0],
                embedding_model="test-embed",
                embedding_dims=3,
            ),
            ChunkRecord(
                chunk_id="chunk-theories",
                document_id="doc-theories",
                source_rel_path="Theories/example.md",
                source_filename="example.md",
                source_type="markdown",
                source_domain="theories",
                source_hash="hash-theories-source",
                citation_label="Theories/example.md",
                chunk_index=0,
                chunk_occurrence=0,
                token_count=2,
                text="Theory text",
                chunk_hash="hash-theories",
                score_hint="Theory text",
                metadata_json="{}",
                vector=[0.0, 1.0, 0.0],
                embedding_model="test-embed",
                embedding_dims=3,
            ),
        ],
    )

    guides_hits = search_chunks(table, query_vector=[1.0, 0.0, 0.0], domain="guides", limit=5)
    theory_hits = search_chunks(table, query_vector=[1.0, 0.0, 0.0], domain="theories", limit=5)

    assert [item.chunk_id for item in guides_hits] == ["chunk-guides"]
    assert [item.chunk_id for item in theory_hits] == ["chunk-theories"]


def test_reset_chunk_table_creates_when_no_prior_table_exists(tmp_path) -> None:
    """``reset_chunk_table`` must succeed even when the table is missing —
    the underlying ``drop_table`` raises and that error is swallowed."""
    database = connect_lancedb(tmp_path / "lancedb")
    table = reset_chunk_table(database, vector_size=3)
    assert "chunk_vectors" in database.list_tables().tables
    # Newly created table is empty and queryable.
    assert table.count_rows() == 0


def test_search_chunks_fts_returns_bm25_ordering(tmp_path) -> None:
    """The FTS lane is BM25 over the ``text`` column, ordered by score."""
    from lxd.stores.lancedb import search_chunks_fts

    database = connect_lancedb(tmp_path / "lancedb")
    table = reset_chunk_table(database, vector_size=3)
    replace_source_chunks(
        table,
        "Theories/addie.md",
        [
            ChunkRecord(
                chunk_id="addie-1",
                document_id="d1",
                source_rel_path="Theories/addie.md",
                source_filename="addie.md",
                source_type="markdown",
                source_domain="theories",
                source_hash="h1",
                citation_label="Theories/addie.md#0",
                chunk_index=0,
                chunk_occurrence=0,
                token_count=8,
                text="ADDIE is a five-phase instructional design model.",
                chunk_hash="ch-addie",
                score_hint="ADDIE",
                metadata_json="{}",
                vector=[1.0, 0.0, 0.0],
                embedding_model="test-embed",
                embedding_dims=3,
            ),
            ChunkRecord(
                chunk_id="kirkpatrick-1",
                document_id="d2",
                source_rel_path="Theories/kirkpatrick.md",
                source_filename="kirkpatrick.md",
                source_type="markdown",
                source_domain="theories",
                source_hash="h2",
                citation_label="Theories/kirkpatrick.md#0",
                chunk_index=0,
                chunk_occurrence=0,
                token_count=8,
                text="Kirkpatrick describes four levels of training evaluation.",
                chunk_hash="ch-kirk",
                score_hint="Kirkpatrick",
                metadata_json="{}",
                vector=[0.0, 1.0, 0.0],
                embedding_model="test-embed",
                embedding_dims=3,
            ),
        ],
    )
    # The FTS index is auto-rebuilt on table open; rebuild explicitly so
    # the just-added rows are visible.
    from lxd.stores.lancedb import refresh_fts_index

    refresh_fts_index(table)
    hits = search_chunks_fts(table, query="ADDIE phase", domain=None, limit=5)
    assert hits, "BM25 should match at least the ADDIE chunk."
    assert hits[0].chunk_id == "addie-1"
    # Empty query returns nothing rather than raising.
    assert search_chunks_fts(table, query="   ", domain=None, limit=5) == []
