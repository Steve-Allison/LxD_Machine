from pathlib import Path

import pyarrow as pa
from lancedb.index import FTS
from lancedb.query import PhraseQuery

from lxd.stores.lancedb import (
    WeightedRRFReranker,
    connect_lancedb,
    ensure_fts_index,
    refresh_fts_index,
    replace_source_chunks,
    reset_chunk_table,
    search_chunks,
    search_chunks_fts,
    search_chunks_hybrid,
)
from lxd.stores.models import ChunkRecord


def test_lancedb_search_and_domain_filter(tmp_path: Path) -> None:
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


def test_reset_chunk_table_creates_when_no_prior_table_exists(tmp_path: Path) -> None:
    """``reset_chunk_table`` must succeed even when the table is missing —
    the underlying ``drop_table`` raises and that error is swallowed."""
    database = connect_lancedb(tmp_path / "lancedb")
    table = reset_chunk_table(database, vector_size=3)
    assert "chunk_vectors" in database.list_tables().tables
    # Newly created table is empty and queryable.
    assert table.count_rows() == 0


def test_search_chunks_fts_returns_bm25_ordering(tmp_path: Path) -> None:
    """The FTS lane is BM25 over the ``text`` column, ordered by score."""
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
    refresh_fts_index(table)
    hits = search_chunks_fts(table, query="ADDIE phase", domain=None, limit=5)
    assert hits, "BM25 should match at least the ADDIE chunk."
    assert hits[0].chunk_id == "addie-1"
    # Empty query returns nothing rather than raising.
    assert search_chunks_fts(table, query="   ", domain=None, limit=5) == []


def _chunk(chunk_id: str, source: str, text: str, vector: list[float]) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        source_rel_path=source,
        source_filename=Path(source).name,
        source_type="markdown",
        source_domain="guides",
        source_hash=f"hash-{chunk_id}",
        citation_label=source,
        chunk_index=0,
        chunk_occurrence=0,
        token_count=8,
        text=text,
        chunk_hash=f"chunk-{chunk_id}",
        score_hint=text,
        metadata_json="{}",
        vector=vector,
        embedding_model="test-embed",
        embedding_dims=3,
    )


def test_phrase_query_requires_token_positions(tmp_path: Path) -> None:
    """Adjacent framework names match; the same words in reverse order do not."""
    database = connect_lancedb(tmp_path / "lancedb")
    table = reset_chunk_table(database, vector_size=3)
    replace_source_chunks(
        table,
        "backward-design.md",
        [
            _chunk(
                "phrase", "backward-design.md", "Backward design is a framework.", [1.0, 0.0, 0.0]
            )
        ],
    )
    replace_source_chunks(
        table,
        "reversed.md",
        [
            _chunk(
                "reversed", "reversed.md", "Design comes before backward thinking.", [0.0, 1.0, 0.0]
            )
        ],
    )
    refresh_fts_index(table)

    hits = search_chunks_fts(
        table,
        query=PhraseQuery(query="backward design", column="text"),
        domain=None,
        limit=5,
    )

    assert [hit.chunk_id for hit in hits] == ["phrase"]


def test_lexical_weight_promotes_the_bm25_hit(tmp_path: Path) -> None:
    """A keyword hit outranks a nearer vector when the lexical weight is 2."""
    database = connect_lancedb(tmp_path / "lancedb")
    table = reset_chunk_table(database, vector_size=3)
    replace_source_chunks(
        table,
        "named.md",
        [
            _chunk(
                "named",
                "named.md",
                "Backward design aligns outcomes and assessment.",
                [1.0, 0.0, 0.0],
            )
        ],
    )
    replace_source_chunks(
        table,
        "near.md",
        [_chunk("near", "near.md", "Unrelated gardening notes about soil.", [0.0, 1.0, 0.0])],
    )
    refresh_fts_index(table)
    dense_prefers_near = [0.0, 1.0, 0.0]

    weighted = search_chunks_hybrid(
        table,
        query="backward design",
        query_vector=dense_prefers_near,
        domain=None,
        limit=2,
        lexical_weight=2.0,
        rrf_k=20,
    )
    unweighted = search_chunks_hybrid(
        table,
        query="backward design",
        query_vector=dense_prefers_near,
        domain=None,
        limit=2,
        lexical_weight=0.0,
        rrf_k=20,
    )

    assert weighted[0].chunk_id == "named"
    assert unweighted[0].chunk_id == "near"


def test_ensure_fts_index_replaces_positionless_legacy_index(tmp_path: Path) -> None:
    database = connect_lancedb(tmp_path / "lancedb")
    table = reset_chunk_table(database, vector_size=3)
    table.drop_index("text_fts_pos_idx")
    table.create_index("text", config=FTS(with_position=False), name="text_fts_idx", replace=False)

    ensure_fts_index(table)

    names = {getattr(index, "name", None) for index in table.list_indices()}
    assert "text_fts_pos_idx" in names
    assert "text_fts_idx" not in names


def test_weighted_rrf_multiplies_only_the_fts_lane() -> None:
    vector_results = pa.table({"_rowid": [1, 2], "chunk_id": ["near", "named"]})
    fts_results = pa.table({"_rowid": [2], "chunk_id": ["named"]})

    weighted = WeightedRRFReranker(k=20, fts_weight=2.0).rerank_hybrid(
        "backward design", vector_results, fts_results
    )
    plain = WeightedRRFReranker(k=20, fts_weight=0.0).rerank_hybrid(
        "backward design", vector_results, fts_results
    )

    assert weighted.column("chunk_id").to_pylist()[0] == "named"
    assert plain.column("chunk_id").to_pylist()[0] == "near"
