"""Graph-aware query routing — augment synthesis context with entity and community data."""

import sqlite3
from dataclasses import dataclass
from operator import attrgetter
from typing import Final

import structlog
import tiktoken

from lxd.settings.models import RuntimeConfig
from lxd.stores.models import ClaimRecord, CommunityReportRecord, EntityProfileRecord
from lxd.stores.sqlite.claims import load_claims_for_entities
from lxd.stores.sqlite.kg_profiles import load_community_report, load_entity_profile

_log = structlog.get_logger(__name__)

_TOKEN_ENCODING_NAME: Final = "cl100k_base"


@dataclass(frozen=True, slots=True)
class GraphContext:
    """Graph context layers to prepend to synthesis prompt."""

    level: str  # "none", "entity", "community"
    entity_profiles: list[EntityProfileRecord]
    community_reports: list[CommunityReportRecord]
    claims: list[ClaimRecord]
    expansion_hops: int


def build_graph_context(
    connection: sqlite3.Connection,
    matched_entity_ids: list[str],
    config: RuntimeConfig,
) -> GraphContext:
    """Build graph context layers from matched entity IDs.

    Graph context is additive — it frames chunk evidence, it does not replace it.

    Truncation under ``knowledge_graph.max_graph_context_tokens`` (B-KG-4):

    1. Always preserve at least one entity profile per matched entity that
       resolves (so synthesis sees evidence framing for every matched
       concept, never silent drop).
    2. Add lower-PageRank profiles, community reports (sorted by community
       id), then claims (sorted by confidence) until the prompt-token
       count reaches the cap.
    3. Token counting uses ``tiktoken`` (``cl100k_base``); for non-OpenAI
       synthesis the count is a high-fidelity proxy.
    """
    kg_cfg = config.knowledge_graph
    if not matched_entity_ids:
        return GraphContext(
            level="none",
            entity_profiles=[],
            community_reports=[],
            claims=[],
            expansion_hops=0,
        )

    profiles: list[EntityProfileRecord] = []
    for entity_id in matched_entity_ids:
        profile = load_entity_profile(connection, entity_id)
        if profile:
            profiles.append(profile)

    profiles.sort(key=attrgetter("pagerank"), reverse=True)
    profiles = profiles[: kg_cfg.max_entity_context]

    if not profiles:
        return GraphContext(
            level="none",
            entity_profiles=[],
            community_reports=[],
            claims=[],
            expansion_hops=0,
        )

    community_ids = {p.community_id for p in profiles if p.community_id is not None}
    reports: list[CommunityReportRecord] = []
    if len(community_ids) >= 2:
        for cid in sorted(community_ids):
            report = load_community_report(connection, cid)
            if report:
                reports.append(report)
        reports = reports[: kg_cfg.max_community_context]

    claims = load_claims_for_entities(
        connection,
        matched_entity_ids,
        limit=kg_cfg.max_claim_context,
    )

    profiles, reports, claims = _trim_to_token_budget(
        profiles=profiles,
        reports=reports,
        claims=claims,
        max_tokens=kg_cfg.max_graph_context_tokens,
    )

    level = "community" if reports else "entity"

    _log.info(
        "graph context built",
        level=level,
        entity_profiles=len(profiles),
        community_reports=len(reports),
        claims=len(claims),
    )

    return GraphContext(
        level=level,
        entity_profiles=profiles,
        community_reports=reports,
        claims=claims,
        expansion_hops=0,
    )


def format_graph_context_prompt(context: GraphContext) -> str:
    """Format graph context as a text block to prepend to the synthesis prompt.

    Returns empty string if no graph context is available.
    """
    if context.level == "none":
        return ""

    sections: list[str] = ["## Graph Context\n"]

    if context.entity_profiles:
        sections.append("### Entity Profiles\n")
        for profile in context.entity_profiles:
            sections.append(f"**{profile.label}** ({profile.entity_type})")
            sections.append(profile.deterministic_summary)
            if profile.llm_summary:
                sections.append(profile.llm_summary)
            sections.append("")

    if context.community_reports:
        sections.append("### Community Context\n")
        for report in context.community_reports:
            sections.append(f"**Community {report.community_id}** ({report.member_count} members)")
            sections.append(report.deterministic_summary)
            if report.llm_summary:
                sections.append(report.llm_summary)
            sections.append("")

    if context.claims:
        sections.append("### Related Claims\n")
        for claim in context.claims:
            sections.append(
                f"- [{claim.claim_type}] {claim.claim_text} (confidence: {claim.confidence:.2f})"
            )
        sections.append("")

    return "\n".join(sections)


def _trim_to_token_budget(
    *,
    profiles: list[EntityProfileRecord],
    reports: list[CommunityReportRecord],
    claims: list[ClaimRecord],
    max_tokens: int,
) -> tuple[list[EntityProfileRecord], list[CommunityReportRecord], list[ClaimRecord]]:
    """Drop low-priority context items until the rendered prompt fits ``max_tokens``.

    Order of preservation (highest first):

    1. The single highest-PageRank entity profile (always kept; even when
       the budget is below this single item the synthesis path still
       receives a non-empty graph context rather than silent fallback).
    2. Remaining entity profiles (already PageRank-sorted).
    3. Community reports (already community-id-sorted).
    4. Claims (sorted by descending confidence).

    Truncation runs from low priority upward: claims first, then reports,
    then trailing profiles.
    """
    encoder = tiktoken.get_encoding(_TOKEN_ENCODING_NAME)

    def fits(
        p: list[EntityProfileRecord], r: list[CommunityReportRecord], c: list[ClaimRecord]
    ) -> bool:
        rendered = format_graph_context_prompt(
            GraphContext(
                level="community" if r else "entity",
                entity_profiles=p,
                community_reports=r,
                claims=c,
                expansion_hops=0,
            )
        )
        return len(encoder.encode(rendered)) <= max_tokens

    sorted_claims = sorted(claims, key=lambda c: c.confidence, reverse=True)

    while sorted_claims and not fits(profiles, reports, sorted_claims):
        sorted_claims.pop()
    if fits(profiles, reports, sorted_claims):
        return profiles, reports, sorted_claims

    while reports and not fits(profiles, reports, sorted_claims):
        reports.pop()
    if fits(profiles, reports, sorted_claims):
        return profiles, reports, sorted_claims

    while len(profiles) > 1 and not fits(profiles, reports, sorted_claims):
        profiles.pop()
    return profiles, reports, sorted_claims
