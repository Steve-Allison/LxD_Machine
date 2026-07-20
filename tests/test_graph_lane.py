"""Tests for the graph-as-retrieval-lane path (Phase 1b).

Uses real SQLite (schema + claims + chunk_rows + entity_profiles +
community_reports) so the claim-to-chunk join and the community-report
fallback are exercised end to end; ``RuntimeConfig`` is a lightweight
``SimpleNamespace`` stand-in, matching the existing convention in
``tests/test_query_pipeline.py``.
"""

import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import cast

from lxd.retrieval.graph_lane import graph_lane_chunk_ids, load_graph_lane_hits
from lxd.settings.models import RuntimeConfig
from lxd.stores.models import (
    ChunkRecord,
    ClaimRecord,
    CommunityReportRecord,
    EntityProfileRecord,
    ManifestRecord,
)
from lxd.stores.schema import ensure_schema
from lxd.stores.sqlite.chunks import replace_source_chunks
from lxd.stores.sqlite.claims import insert_claims
from lxd.stores.sqlite.connection import connect_sqlite
from lxd.stores.sqlite.kg_profiles import upsert_community_report, upsert_entity_profile
from lxd.stores.sqlite.manifest import upsert_manifest_record


def _config(*, graph_lane_enabled: bool = True, max_graph_lane_hits: int = 10) -> RuntimeConfig:
    return cast(
        "RuntimeConfig",
        SimpleNamespace(
            retrieval=SimpleNamespace(
                graph_lane_enabled=graph_lane_enabled,
                max_graph_lane_hits=max_graph_lane_hits,
            )
        ),
    )


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


def _claim(
    claim_id: str,
    *,
    chunk_id: str,
    subject: str | None,
    object_: str | None = None,
    confidence: float,
) -> ClaimRecord:
    return ClaimRecord(
        claim_id=claim_id,
        chunk_id=chunk_id,
        document_id="doc-1",
        source_rel_path="page.md",
        claim_text=f"claim text {claim_id}",
        subject_entity_id=subject,
        object_entity_id=object_,
        claim_type="assertion",
        confidence=confidence,
        extraction_model="test",
        extracted_at="2026-05-05T00:00:00Z",
    )


def _report(community_id: int) -> CommunityReportRecord:
    return CommunityReportRecord(
        community_id=community_id,
        community_level=0,
        member_count=2,
        member_entity_ids_json="[]",
        deterministic_summary=f"Deterministic summary for community {community_id}.",
        llm_summary=None,
        top_entities_json="[]",
        top_claims_json="[]",
        intra_community_edge_count=0,
        source_hash="hash",
        generated_at="2026-05-05T00:00:00Z",
    )


def test_returns_empty_when_graph_lane_disabled(tmp_path: Path) -> None:
    connection = connect_sqlite(tmp_path / "lxd.sqlite3")
    try:
        ensure_schema(connection)
        hits = load_graph_lane_hits(
            connection, ["entity-a"], _config(graph_lane_enabled=False)
        )
    finally:
        connection.close()
    assert hits == []


def test_returns_empty_when_no_matched_entities(tmp_path: Path) -> None:
    connection = connect_sqlite(tmp_path / "lxd.sqlite3")
    try:
        ensure_schema(connection)
        hits = load_graph_lane_hits(connection, [], _config())
    finally:
        connection.close()
    assert hits == []


def test_loads_claim_hits_ordered_by_confidence(tmp_path: Path) -> None:
    connection = connect_sqlite(tmp_path / "lxd.sqlite3")
    try:
        ensure_schema(connection)
        _seed_manifest(connection, "page.md")
        replace_source_chunks(
            connection,
            source_rel_path="page.md",
            chunk_records=[_chunk("c1", "page.md"), _chunk("c2", "page.md")],
            mention_records=[],
        )
        insert_claims(
            connection,
            [
                _claim("claim-low", chunk_id="c1", subject="entity-a", confidence=0.4),
                _claim("claim-high", chunk_id="c2", subject="entity-a", confidence=0.9),
            ],
        )

        hits = load_graph_lane_hits(connection, ["entity-a"], _config())
    finally:
        connection.close()

    assert [hit.chunk_id for hit in hits] == ["c2", "c1"]
    assert [hit.lane_kind for hit in hits] == ["claim", "claim"]
    assert hits[0].score == 0.9
    assert hits[0].claim_id == "claim-high"
    assert hits[0].text == "text-c2"
    assert hits[0].citation_label == "page.md#0"


def test_skips_claims_whose_chunk_row_is_missing(tmp_path: Path) -> None:
    """A claim whose backing chunk row is absent is skipped, not surfaced broken.

    ``claims.chunk_id`` has an ``ON DELETE CASCADE`` foreign key onto
    ``chunk_rows``, so this state cannot arise through normal deletes —
    the FK constraint is bypassed here purely to exercise the defensive
    skip path in :func:`load_graph_lane_hits`.
    """
    connection = connect_sqlite(tmp_path / "lxd.sqlite3")
    try:
        ensure_schema(connection)
        _seed_manifest(connection, "page.md")
        replace_source_chunks(
            connection,
            source_rel_path="page.md",
            chunk_records=[_chunk("c1", "page.md")],
            mention_records=[],
        )
        insert_claims(
            connection,
            [_claim("claim-real", chunk_id="c1", subject="entity-a", confidence=0.7)],
        )
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute(
            """
            INSERT INTO claims (
                claim_id, chunk_id, document_id, source_rel_path,
                claim_text, subject_entity_id, object_entity_id,
                claim_type, confidence, extraction_model, extracted_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "claim-orphan",
                "missing-chunk",
                "doc-1",
                "page.md",
                "claim text claim-orphan",
                "entity-a",
                None,
                "assertion",
                0.95,
                "test",
                "2026-05-05T00:00:00Z",
            ),
        )
        connection.commit()
        connection.execute("PRAGMA foreign_keys = ON")

        hits = load_graph_lane_hits(connection, ["entity-a"], _config())
    finally:
        connection.close()

    assert [hit.claim_id for hit in hits] == ["claim-real"]


def test_falls_back_to_community_report_when_no_claims(tmp_path: Path) -> None:
    connection = connect_sqlite(tmp_path / "lxd.sqlite3")
    try:
        ensure_schema(connection)
        upsert_entity_profile(
            connection,
            _profile_with_community("entity-a", community_id=3),
        )
        upsert_community_report(connection, _report(3))

        hits = load_graph_lane_hits(connection, ["entity-a"], _config())
    finally:
        connection.close()

    assert len(hits) == 1
    assert hits[0].lane_kind == "community_report"
    assert hits[0].community_id == 3
    assert hits[0].chunk_id == "community:3"
    assert "Deterministic summary for community 3" in hits[0].text


def test_claims_are_preferred_over_community_reports_under_cap(tmp_path: Path) -> None:
    connection = connect_sqlite(tmp_path / "lxd.sqlite3")
    try:
        ensure_schema(connection)
        _seed_manifest(connection, "page.md")
        replace_source_chunks(
            connection,
            source_rel_path="page.md",
            chunk_records=[_chunk("c1", "page.md")],
            mention_records=[],
        )
        insert_claims(
            connection,
            [_claim("claim-1", chunk_id="c1", subject="entity-a", confidence=0.6)],
        )
        upsert_entity_profile(
            connection,
            _profile_with_community("entity-a", community_id=3),
        )
        upsert_community_report(connection, _report(3))

        hits = load_graph_lane_hits(connection, ["entity-a"], _config(max_graph_lane_hits=1))
    finally:
        connection.close()

    assert len(hits) == 1
    assert hits[0].lane_kind == "claim"


def test_graph_lane_chunk_ids_excludes_community_hits_and_preserves_order(tmp_path: Path) -> None:
    connection = connect_sqlite(tmp_path / "lxd.sqlite3")
    try:
        ensure_schema(connection)
        _seed_manifest(connection, "page.md")
        replace_source_chunks(
            connection,
            source_rel_path="page.md",
            chunk_records=[_chunk("c1", "page.md"), _chunk("c2", "page.md")],
            mention_records=[],
        )
        insert_claims(
            connection,
            [
                _claim("claim-high", chunk_id="c2", subject="entity-a", confidence=0.9),
                _claim("claim-low", chunk_id="c1", subject="entity-a", confidence=0.4),
            ],
        )
        upsert_entity_profile(
            connection,
            _profile_with_community("entity-a", community_id=3),
        )
        upsert_community_report(connection, _report(3))

        hits = load_graph_lane_hits(connection, ["entity-a"], _config())
        chunk_ids = graph_lane_chunk_ids(hits)
    finally:
        connection.close()

    assert chunk_ids == ["c2", "c1"]


def _profile_with_community(entity_id: str, *, community_id: int) -> EntityProfileRecord:
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
        pagerank=0.5,
        betweenness=0.0,
        closeness=0.0,
        in_degree=0,
        out_degree=0,
        eigenvector=0.0,
        community_id=community_id,
        source_hash="hash",
        generated_at="2026-05-05",
    )
