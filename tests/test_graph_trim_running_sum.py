"""O(n) graph-context token-budget trim — pin behaviour against the prompt formatter."""

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


def test_trim_running_sum_matches_full_render_cap() -> None:
    """Running-sum trim must keep the rendered prompt under ``max_tokens``."""
    profiles = [_profile("a", pagerank=0.9, summary="short summary a")]
    reports = [_report(1, summary="community " * 40)]
    claims = [
        _claim(f"c{i}", confidence=0.9 - i * 0.02, text=("padding text " * 30) + f"claim {i}")
        for i in range(15)
    ]

    out_p, out_r, out_c = _trim_to_token_budget(
        profiles=list(profiles),
        reports=list(reports),
        claims=list(claims),
        max_tokens=250,
    )
    rendered = _token_count(
        GraphContext(
            level="community" if out_r else "entity",
            entity_profiles=out_p,
            community_reports=out_r,
            claims=out_c,
            expansion_hops=0,
        )
    )
    assert rendered <= 250
    assert len(out_p) >= 1
