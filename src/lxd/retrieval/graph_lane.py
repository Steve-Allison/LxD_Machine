"""Graph-as-retrieval-lane — surface claims and community reports as boostable hits.

The knowledge graph already frames synthesis prompts additively (see
:mod:`lxd.retrieval.graph_routing`). This module goes one step further:
claims linked to the query's matched entities point back at real chunk
rows, so they can also participate in the dense/BM25/relation RRF fusion
in :mod:`lxd.retrieval.query_pipeline` — a claim-backed chunk should rank
higher even if its raw vector similarity was middling. Community-report
summaries have no backing chunk row; they are included here only as
synthetic hits for prompt framing, never as fusion input.
"""

import sqlite3
from dataclasses import dataclass
from typing import Literal

import structlog

from lxd.settings.models import RuntimeConfig
from lxd.stores.sqlite.chunks import load_chunk_record_by_id
from lxd.stores.sqlite.claims import load_claims_for_entities
from lxd.stores.sqlite.kg_profiles import load_community_report, load_entity_profile

_log = structlog.get_logger(__name__)

_DEFAULT_COMMUNITY_HIT_SCORE = 0.5


@dataclass(frozen=True, slots=True)
class GraphLaneHit:
    """One graph-derived retrieval hit — a claim or a community-report summary.

    ``lane_kind`` distinguishes the two: ``"claim"`` hits carry a real
    ``chunk_id`` that exists in ``chunk_rows`` / the vector store, so
    :func:`graph_lane_chunk_ids` can feed them into RRF fusion.
    ``"community_report"`` hits are synthetic — their ``chunk_id`` is a
    ``community:{id}`` placeholder label, not a real chunk, and they are
    never used for fusion boosting.
    """

    chunk_id: str
    document_id: str
    citation_label: str
    source_rel_path: str
    source_filename: str
    source_type: str
    source_domain: str
    source_hash: str
    chunk_index: int
    text: str
    score: float
    lane_kind: Literal["claim", "community_report"]
    claim_id: str | None = None
    community_id: int | None = None


def load_graph_lane_hits(
    connection: sqlite3.Connection,
    matched_entity_ids: list[str],
    config: RuntimeConfig,
) -> list[GraphLaneHit]:
    """Load graph-derived retrieval hits for the given matched entities.

    Claims are loaded first (ranked by confidence, the same order
    :func:`lxd.stores.sqlite.claims.load_claims_for_entities` already
    returns) and are preferred over community-report hits when both are
    present and ``max_graph_lane_hits`` would otherwise be exceeded.
    Claims whose backing chunk row is missing (e.g. the source was
    deleted after claim extraction) are silently skipped rather than
    surfaced as broken hits.

    Returns an empty list when the graph lane is disabled, no entities
    matched, or the caller is on a store without knowledge-graph tables
    (graceful degradation — the query pipeline treats an empty list as
    "no graph lane signal").
    """
    retrieval_cfg = config.retrieval
    if not matched_entity_ids or not retrieval_cfg.graph_lane_enabled:
        return []
    max_hits = retrieval_cfg.max_graph_lane_hits

    try:
        claims = load_claims_for_entities(connection, matched_entity_ids, limit=max_hits)
    except sqlite3.DatabaseError, OSError:
        _log.warning("graph_lane_claims_load_failed", exc_info=True)
        return []

    hits: list[GraphLaneHit] = []
    for claim in claims:
        chunk = load_chunk_record_by_id(connection, claim.chunk_id)
        if chunk is None:
            continue
        hits.append(
            GraphLaneHit(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                citation_label=chunk.citation_label,
                source_rel_path=chunk.source_rel_path,
                source_filename=chunk.source_filename,
                source_type=chunk.source_type,
                source_domain=chunk.source_domain,
                source_hash=chunk.source_hash,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                score=claim.confidence,
                lane_kind="claim",
                claim_id=claim.claim_id,
            )
        )
        if len(hits) >= max_hits:
            return hits

    remaining = max_hits - len(hits)
    if remaining <= 0:
        return hits

    for community_id in _matched_community_ids(connection, matched_entity_ids):
        report = load_community_report(connection, community_id)
        if report is None:
            continue
        text = report.deterministic_summary
        if report.llm_summary:
            text = f"{text}\n{report.llm_summary}"
        hits.append(
            GraphLaneHit(
                chunk_id=f"community:{community_id}",
                document_id=f"community:{community_id}",
                citation_label=f"community:{community_id}",
                source_rel_path="",
                source_filename="",
                source_type="community_report",
                source_domain="",
                source_hash="",
                chunk_index=0,
                text=text,
                score=_DEFAULT_COMMUNITY_HIT_SCORE,
                lane_kind="community_report",
                community_id=community_id,
            )
        )
        if len(hits) >= max_hits:
            break
    return hits


def graph_lane_chunk_ids(hits: list[GraphLaneHit]) -> list[str]:
    """Return claim-linked chunk IDs in hit order, preserving order for RRF.

    Community-report synthetic hits are excluded on purpose — they have
    no backing row in the vector store, so mixing them into a chunk-id
    boost set would silently no-op (or worse, collide with a real
    ``chunk_id`` string) inside the fusion lane.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for hit in hits:
        if hit.lane_kind != "claim":
            continue
        if hit.chunk_id in seen:
            continue
        seen.add(hit.chunk_id)
        ordered.append(hit.chunk_id)
    return ordered


def _matched_community_ids(
    connection: sqlite3.Connection, entity_ids: list[str]
) -> list[int]:
    """Return distinct, ascending community IDs for the given matched entities.

    Reuses :func:`lxd.stores.sqlite.kg_profiles.load_entity_profile` (the
    same accessor :mod:`lxd.retrieval.graph_routing` uses) rather than a
    bespoke query, so community membership can never drift between the
    prompt-framing path and the retrieval-lane path.
    """
    community_ids: set[int] = set()
    for entity_id in entity_ids:
        try:
            profile = load_entity_profile(connection, entity_id)
        except sqlite3.DatabaseError, OSError:
            _log.warning("graph_lane_entity_profile_load_failed", exc_info=True)
            continue
        if profile is not None and profile.community_id is not None:
            community_ids.add(profile.community_id)
    return sorted(community_ids)
