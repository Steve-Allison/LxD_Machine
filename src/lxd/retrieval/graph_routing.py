"""Graph-aware query routing — augment synthesis context with entity and community data."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import structlog

from lxd.settings.models import RuntimeConfig
from lxd.stores.models import ClaimRecord, CommunityReportRecord, EntityProfileRecord
from lxd.stores.sqlite import (
    load_claims_for_entities,
    load_community_report,
    load_entity_profile,
)

_log = structlog.get_logger(__name__)


@dataclass(frozen=True)
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

    # Entity profiles — top N by PageRank
    profiles: list[EntityProfileRecord] = []
    for entity_id in matched_entity_ids:
        profile = load_entity_profile(connection, entity_id)
        if profile:
            profiles.append(profile)

    # Sort by PageRank and limit
    profiles.sort(key=lambda p: p.pagerank, reverse=True)
    profiles = profiles[: kg_cfg.max_entity_context]

    if not profiles:
        return GraphContext(
            level="none",
            entity_profiles=[],
            community_reports=[],
            claims=[],
            expansion_hops=0,
        )

    # Community reports — if matched entities span 2+ communities
    community_ids = {p.community_id for p in profiles if p.community_id is not None}
    reports: list[CommunityReportRecord] = []
    if len(community_ids) >= 2:
        for cid in sorted(community_ids):
            report = load_community_report(connection, cid)
            if report:
                reports.append(report)
        reports = reports[: kg_cfg.max_community_context]

    # Claims — top N for matched entities, ranked by confidence
    claims = load_claims_for_entities(
        connection,
        matched_entity_ids,
        limit=kg_cfg.max_claim_context,
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
