"""Tests for graph context token-budget truncation (B-KG-4)."""

from __future__ import annotations

import tiktoken

from lxd.retrieval import graph_routing as _graph_routing
from lxd.retrieval.graph_routing import GraphContext, format_graph_context_prompt
from lxd.stores.models import ClaimRecord, CommunityReportRecord, EntityProfileRecord

_trim_to_token_budget = _graph_routing._trim_to_token_budget  # pyright: ignore[reportPrivateUsage]


def _profile(entity_id: str, *, pagerank: float, summary: str) -> EntityProfileRecord:
    return EntityProfileRecord(
        entity_id=entity_id,
        label=f"Label {entity_id}",
        entity_type="concept",
        domain="learning",
        aliases_json="[]",
        deterministic_summary=summary,
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
        community_id=1,
        source_hash="hash",
        generated_at="2026-05-05T00:00:00Z",
    )


def _claim(claim_id: str, *, confidence: float, text: str) -> ClaimRecord:
    return ClaimRecord(
        claim_id=claim_id,
        chunk_id="c1",
        document_id="d1",
        source_rel_path="x.md",
        claim_text=text,
        subject_entity_id=None,
        object_entity_id=None,
        claim_type="assertion",
        confidence=confidence,
        extraction_model="test",
        extracted_at="2026-05-05T00:00:00Z",
    )


def _report(community_id: int, *, summary: str) -> CommunityReportRecord:
    return CommunityReportRecord(
        community_id=community_id,
        community_level=0,
        member_count=2,
        member_entity_ids_json="[]",
        deterministic_summary=summary,
        llm_summary=None,
        top_entities_json="[]",
        top_claims_json="[]",
        intra_community_edge_count=0,
        source_hash="hash",
        generated_at="2026-05-05T00:00:00Z",
    )


def _token_count(graph: GraphContext) -> int:
    encoder = tiktoken.get_encoding("cl100k_base")
    return len(encoder.encode(format_graph_context_prompt(graph)))


def test_trim_does_not_truncate_when_already_under_budget() -> None:
    profiles = [_profile("a", pagerank=0.9, summary="short summary a")]
    reports: list[CommunityReportRecord] = []
    claims = [_claim("c1", confidence=0.9, text="claim one")]

    out_p, out_r, out_c = _trim_to_token_budget(
        profiles=profiles,
        reports=reports,
        claims=claims,
        max_tokens=1500,
    )

    assert out_p == profiles
    assert out_r == reports
    assert out_c == claims


def test_trim_drops_lowest_confidence_claims_first() -> None:
    profiles = [_profile("a", pagerank=0.9, summary="short summary a")]
    reports: list[CommunityReportRecord] = []
    claims = [
        _claim(
            f"c{i}",
            confidence=0.9 - i * 0.05,
            text=("padding text " * 50) + f"claim {i}",
        )
        for i in range(20)
    ]

    out_p, out_r, out_c = _trim_to_token_budget(
        profiles=profiles,
        reports=reports,
        claims=list(claims),
        max_tokens=200,
    )

    rendered_tokens = _token_count(
        GraphContext(
            level="entity",
            entity_profiles=out_p,
            community_reports=out_r,
            claims=out_c,
            expansion_hops=0,
        )
    )
    assert rendered_tokens <= 200, f"Rendered context {rendered_tokens} tokens exceeds cap of 200."
    assert len(out_c) < len(claims), "Trim should have dropped at least one claim."
    if out_c:
        surviving_min_confidence = min(c.confidence for c in out_c)
        dropped_max_confidence = max(
            (c.confidence for c in claims if c.claim_id not in {x.claim_id for x in out_c}),
            default=0.0,
        )
        assert surviving_min_confidence >= dropped_max_confidence, (
            "Higher-confidence claims should survive; saw "
            f"surviving_min={surviving_min_confidence}, dropped_max={dropped_max_confidence}."
        )


def test_trim_preserves_at_least_one_entity_profile_under_severe_budget() -> None:
    profiles = [
        _profile("alpha", pagerank=0.9, summary="alpha " * 200),
        _profile("beta", pagerank=0.5, summary="beta " * 200),
    ]
    reports = [_report(1, summary="report " * 200)]
    claims = [_claim("c1", confidence=0.9, text="claim " * 200)]

    out_p, _out_r, _out_c = _trim_to_token_budget(
        profiles=list(profiles),
        reports=list(reports),
        claims=list(claims),
        max_tokens=10,
    )

    assert len(out_p) >= 1, "Even under a severe budget, at least one entity profile must survive."
    assert out_p[0].entity_id == "alpha", "The retained profile should be the highest-PageRank one."
