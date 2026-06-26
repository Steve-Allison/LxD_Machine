"""Integration test for `load_chunk_centrality_signals`.

Uses real SQLite — verifies the join across ``chunk_rows -> mention_rows
-> entity_profiles`` returns max-PageRank and de-duplicated community
ids per chunk, and that chunks without graph data degrade gracefully
(absent from the result, callers default-fill).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from lxd.stores.models import (
    ChunkRecord,
    EntityProfileRecord,
    ManifestRecord,
    MentionRecord,
)
from lxd.stores.schema import ensure_schema
from lxd.stores.sqlite.chunks import load_chunk_centrality_signals, replace_source_chunks
from lxd.stores.sqlite.connection import connect_sqlite
from lxd.stores.sqlite.kg_profiles import upsert_entity_profile
from lxd.stores.sqlite.manifest import upsert_manifest_record


def _seed_manifest(connection: sqlite3.Connection, source_rel_path: str) -> None:
    upsert_manifest_record(
        connection,
        ManifestRecord(
            source_rel_path=source_rel_path,
            absolute_path=f"/abs/{source_rel_path}",
            source_type="markdown",
            source_domain="wiki",
            document_id="doc-1",
            file_size_bytes=10,
            content_hash=f"hash-{source_rel_path}",
            parent_source_rel_path=None,
            chunk_count=1,
            last_seen_at="2026-05-05",
            last_processed_at="2026-05-05",
            last_committed_at="2026-05-05",
            error_message=None,
            lifecycle_status="complete",
            retrieval_status="searchable",
        ),
    )


def _chunk(chunk_id: str, source_rel_path: str) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        document_id="doc-1",
        source_rel_path=source_rel_path,
        source_filename=Path(source_rel_path).name,
        source_type="markdown",
        source_domain="wiki",
        source_hash=f"hash-{source_rel_path}",
        citation_label=f"{source_rel_path}#0",
        chunk_index=0,
        chunk_occurrence=0,
        token_count=10,
        text=f"text-{chunk_id}",
        chunk_hash=f"ch-{chunk_id}",
        score_hint="hint",
        metadata_json="{}",
        vector=[0.1, 0.2, 0.3],
        embedding_model="m",
        embedding_dims=3,
    )


def _mention(*, chunk_id: str, entity_id: str) -> MentionRecord:
    return MentionRecord(
        chunk_id=chunk_id,
        entity_id=entity_id,
        term_source="canonical",
        surface_form=entity_id,
        start_char=0,
        end_char=len(entity_id),
    )


def _profile(*, entity_id: str, pagerank: float, community_id: int | None) -> EntityProfileRecord:
    return EntityProfileRecord(
        entity_id=entity_id,
        label=entity_id,
        entity_type="concept",
        domain="wiki",
        aliases_json="[]",
        deterministic_summary="",
        llm_summary=None,
        chunk_count=0,
        doc_count=0,
        mention_count=0,
        claim_count=0,
        top_predicates_json="[]",
        top_claims_json="[]",
        pagerank=pagerank,
        betweenness=0.0,
        closeness=0.0,
        in_degree=0,
        out_degree=0,
        eigenvector=0.0,
        community_id=community_id,
        source_hash="src-hash",
        generated_at="2026-05-05",
    )


def test_load_chunk_centrality_signals_aggregates_max_pagerank_and_communities(
    tmp_path: Path,
) -> None:
    connection = connect_sqlite(tmp_path / "lxd.sqlite3")
    try:
        ensure_schema(connection)
        _seed_manifest(connection, "addie-model.md")

        replace_source_chunks(
            connection,
            source_rel_path="addie-model.md",
            chunk_records=[_chunk("c1", "addie-model.md")],
            mention_records=[
                _mention(chunk_id="c1", entity_id="addie_model"),
                _mention(chunk_id="c1", entity_id="backward_design"),
            ],
        )
        upsert_entity_profile(
            connection, _profile(entity_id="addie_model", pagerank=0.45, community_id=2)
        )
        upsert_entity_profile(
            connection, _profile(entity_id="backward_design", pagerank=0.85, community_id=5)
        )

        signals = load_chunk_centrality_signals(connection, ["c1"])
    finally:
        connection.close()

    assert "c1" in signals
    assert signals["c1"].max_pagerank == 0.85
    assert signals["c1"].community_ids == (2, 5)


def test_load_chunk_centrality_signals_returns_empty_for_unmentioned_chunks(
    tmp_path: Path,
) -> None:
    """Chunks with no profiled entities are absent from the result."""
    connection = connect_sqlite(tmp_path / "lxd.sqlite3")
    try:
        ensure_schema(connection)
        _seed_manifest(connection, "lonely.md")
        replace_source_chunks(
            connection,
            source_rel_path="lonely.md",
            chunk_records=[_chunk("c-lonely", "lonely.md")],
            mention_records=[],
        )
        signals = load_chunk_centrality_signals(connection, ["c-lonely"])
    finally:
        connection.close()
    assert signals == {}


def test_load_chunk_centrality_signals_handles_empty_chunk_list(tmp_path: Path) -> None:
    connection = connect_sqlite(tmp_path / "lxd.sqlite3")
    try:
        ensure_schema(connection)
        assert load_chunk_centrality_signals(connection, []) == {}
    finally:
        connection.close()


def test_load_chunk_centrality_signals_dedupes_communities_per_chunk(tmp_path: Path) -> None:
    """If two mentioned entities share a community, the chunk's
    community_ids tuple includes that community only once."""
    connection = connect_sqlite(tmp_path / "lxd.sqlite3")
    try:
        ensure_schema(connection)
        _seed_manifest(connection, "page.md")
        replace_source_chunks(
            connection,
            source_rel_path="page.md",
            chunk_records=[_chunk("c-shared", "page.md")],
            mention_records=[
                _mention(chunk_id="c-shared", entity_id="entity_a"),
                _mention(chunk_id="c-shared", entity_id="entity_b"),
            ],
        )
        upsert_entity_profile(
            connection, _profile(entity_id="entity_a", pagerank=0.2, community_id=7)
        )
        upsert_entity_profile(
            connection, _profile(entity_id="entity_b", pagerank=0.3, community_id=7)
        )
        signals = load_chunk_centrality_signals(connection, ["c-shared"])
    finally:
        connection.close()
    assert signals["c-shared"].community_ids == (7,)


def test_load_chunk_centrality_signals_omits_null_community_ids(tmp_path: Path) -> None:
    """Profiles with community_id=NULL contribute their pagerank but
    no community membership."""
    connection = connect_sqlite(tmp_path / "lxd.sqlite3")
    try:
        ensure_schema(connection)
        _seed_manifest(connection, "p.md")
        replace_source_chunks(
            connection,
            source_rel_path="p.md",
            chunk_records=[_chunk("c", "p.md")],
            mention_records=[_mention(chunk_id="c", entity_id="loner")],
        )
        upsert_entity_profile(
            connection, _profile(entity_id="loner", pagerank=0.55, community_id=None)
        )
        signals = load_chunk_centrality_signals(connection, ["c"])
    finally:
        connection.close()
    assert signals["c"].max_pagerank == 0.55
    assert signals["c"].community_ids == ()
