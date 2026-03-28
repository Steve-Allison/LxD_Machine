"""Build deterministic entity profiles and optional LLM enrichment."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import UTC, datetime
from typing import Any

import structlog

from lxd.domain.ids import blake3_hex
from lxd.ontology.entity_graph import CentralityScores
from lxd.settings.models import RuntimeConfig
from lxd.stores.models import CommunityReportRecord, EntityProfileRecord
from lxd.stores.sqlite import (
    load_claims_for_entities,
    load_community_members,
    load_entity_mention_stats,
    load_entity_profile_source_hashes,
    load_top_predicates_for_entity,
    upsert_community_report,
    upsert_entity_profile,
)

_log = structlog.get_logger(__name__)


def build_entity_profiles(
    connection: sqlite3.Connection,
    entity_definitions: list[dict[str, Any]],
    centrality: dict[str, CentralityScores],
    community_assignments: dict[str, int],
    config: RuntimeConfig,
    *,
    force: bool = False,
) -> int:
    """Build deterministic profiles for all entities.

    Returns:
        Number of profiles built or updated.
    """
    mention_stats = load_entity_mention_stats(connection)
    existing_hashes = load_entity_profile_source_hashes(connection) if not force else {}

    timestamp = datetime.now(UTC).isoformat()
    profiles_built = 0

    for entity_def in entity_definitions:
        entity_id = str(entity_def.get("canonical_id", ""))
        if not entity_id:
            continue

        label = entity_id.replace("_", " ").title()
        entity_type = str(entity_def.get("entity_type", ""))
        domain = str(entity_def.get("domain", ""))
        aliases = entity_def.get("aliases", [])
        if not isinstance(aliases, list):
            aliases = []

        # Mention stats
        stats = mention_stats.get(entity_id, {"chunk_count": 0, "doc_count": 0, "mention_count": 0})
        chunk_count = stats["chunk_count"]
        doc_count = stats["doc_count"]
        mention_count = stats["mention_count"]

        # Centrality
        scores = centrality.get(entity_id)
        pagerank = scores.pagerank if scores else 0.0
        betweenness = scores.betweenness if scores else 0.0
        closeness = scores.closeness if scores else 0.0
        in_degree = scores.in_degree if scores else 0
        out_degree = scores.out_degree if scores else 0
        eigenvector = scores.eigenvector if scores else 0.0

        # Community
        community_id = community_assignments.get(entity_id)

        # Claims
        claims = load_claims_for_entities(connection, [entity_id], limit=10)
        claim_count = len(claims)
        top_claims = [
            {"claim_text": c.claim_text, "confidence": c.confidence, "claim_type": c.claim_type}
            for c in claims[:5]
        ]

        # Top predicates from canonical relations
        top_preds = load_top_predicates_for_entity(connection, entity_id, limit=10)

        # Compute chunk IDs for source hash
        chunk_ids_rows = connection.execute(
            "SELECT DISTINCT chunk_id FROM mention_rows WHERE entity_id = ? ORDER BY chunk_id",
            (entity_id,),
        ).fetchall()
        chunk_ids = sorted(str(r["chunk_id"]) for r in chunk_ids_rows)
        claim_ids = sorted(c.claim_id for c in claims)

        # Source hash per Section 3.2 definition
        source_hash = blake3_hex(
            *chunk_ids,
            str(pagerank),
            str(betweenness),
            str(closeness),
            str(in_degree),
            str(out_degree),
            str(eigenvector),
            str(community_id),
            *claim_ids,
        )

        # Incremental: skip if source hash unchanged
        if entity_id in existing_hashes and existing_hashes[entity_id] == source_hash:
            continue

        # Deterministic summary
        total_entities = len(entity_definitions)
        pr_rank = _rank_position(entity_id, centrality, "pagerank")
        bt_rank = _rank_position(entity_id, centrality, "betweenness")
        cl_rank = _rank_position(entity_id, centrality, "closeness")

        # Community member count
        community_member_count = 0
        if community_id is not None:
            members = load_community_members(connection, community_id)
            community_member_count = len(members)

        # Top relations as text
        top_rels_text = (
            "; ".join(f"{p['predicate']} ({p['count']})" for p in top_preds[:5]) or "none"
        )

        # Top claims as text
        top_claims_text = "; ".join(str(c["claim_text"])[:80] for c in top_claims[:3]) or "none"

        community_text = (
            f"{community_id} ({community_member_count} members)"
            if community_id is not None
            else "unassigned"
        )

        deterministic_summary = (
            f"{label} is a {entity_type} entity in the {domain} domain. "
            f"It has {mention_count} mentions across {chunk_count} chunks "
            f"from {doc_count} source documents. "
            f"Centrality: PageRank {pr_rank}/{total_entities} | "
            f"Betweenness {bt_rank}/{total_entities} | "
            f"Closeness {cl_rank}/{total_entities}. "
            f"Community: {community_text}. "
            f"Key relationships: {top_rels_text}. "
            f"Key claims: {top_claims_text}."
        )

        record = EntityProfileRecord(
            entity_id=entity_id,
            label=label,
            entity_type=entity_type,
            domain=domain,
            aliases_json=json.dumps(aliases, separators=(",", ":")),
            deterministic_summary=deterministic_summary,
            llm_summary=None,
            chunk_count=chunk_count,
            doc_count=doc_count,
            mention_count=mention_count,
            claim_count=claim_count,
            top_predicates_json=json.dumps(top_preds, separators=(",", ":")),
            top_claims_json=json.dumps(top_claims, separators=(",", ":")),
            pagerank=pagerank,
            betweenness=betweenness,
            closeness=closeness,
            in_degree=in_degree,
            out_degree=out_degree,
            eigenvector=eigenvector,
            community_id=community_id,
            source_hash=source_hash,
            generated_at=timestamp,
        )
        upsert_entity_profile(connection, record)
        profiles_built += 1

    _log.info("entity profiles built", profiles_built=profiles_built)
    return profiles_built


def build_community_reports(
    connection: sqlite3.Connection,
    community_assignments: dict[str, int],
    centrality: dict[str, CentralityScores],
    *,
    force: bool = False,
) -> int:
    """Build deterministic community reports.

    Returns:
        Number of reports built.
    """
    # Group entities by community
    communities: dict[int, list[str]] = {}
    for entity_id, community_id in community_assignments.items():
        communities.setdefault(community_id, []).append(entity_id)

    timestamp = datetime.now(UTC).isoformat()
    reports_built = 0

    for community_id, member_ids in communities.items():
        sorted_members = sorted(member_ids)

        # Top entities by PageRank within community
        ranked = sorted(
            member_ids,
            key=lambda eid: (
                centrality.get(
                    eid,
                    CentralityScores(
                        entity_id=eid,
                        pagerank=0,
                        betweenness=0,
                        closeness=0,
                        in_degree=0,
                        out_degree=0,
                        eigenvector=0,
                    ),
                ).pagerank
            ),
            reverse=True,
        )
        top_entities = [
            {"entity_id": eid, "pagerank": centrality[eid].pagerank}
            for eid in ranked[:10]
            if eid in centrality
        ]

        # Top claims from community members
        claims = load_claims_for_entities(connection, member_ids, limit=10)
        top_claims = [{"claim_text": c.claim_text, "confidence": c.confidence} for c in claims[:5]]

        # Intra-community edge count
        intra_edges = connection.execute(
            """
            SELECT COUNT(*) AS cnt FROM relations
            WHERE subject_entity_id IN ({placeholders})
              AND object_entity_id IN ({placeholders})
            """.format(placeholders=",".join("?" * len(member_ids))),
            [*member_ids, *member_ids],
        ).fetchone()
        intra_edge_count = int(intra_edges["cnt"]) if intra_edges else 0

        # Member source hashes for staleness detection
        member_hashes = []
        for eid in sorted_members:
            row = connection.execute(
                "SELECT source_hash FROM entity_profiles WHERE entity_id = ?", (eid,)
            ).fetchone()
            if row:
                member_hashes.append(str(row["source_hash"]))

        source_hash = blake3_hex(*sorted_members, *sorted(member_hashes))

        # Deterministic summary
        member_labels = [eid.replace("_", " ").title() for eid in ranked[:5]]
        claims_text = "; ".join(str(c["claim_text"])[:60] for c in top_claims[:3]) or "none"

        deterministic_summary = (
            f"Community {community_id} contains {len(member_ids)} entities. "
            f"Top members: {', '.join(member_labels)}. "
            f"Intra-community edges: {intra_edge_count}. "
            f"Key claims: {claims_text}."
        )

        record = CommunityReportRecord(
            community_id=community_id,
            community_level=0,
            member_count=len(member_ids),
            member_entity_ids_json=json.dumps(sorted_members, separators=(",", ":")),
            deterministic_summary=deterministic_summary,
            llm_summary=None,
            top_entities_json=json.dumps(top_entities, separators=(",", ":")),
            top_claims_json=json.dumps(top_claims, separators=(",", ":")),
            intra_community_edge_count=intra_edge_count,
            source_hash=source_hash,
            generated_at=timestamp,
        )
        upsert_community_report(connection, record)
        reports_built += 1

    _log.info("community reports built", reports_built=reports_built)
    return reports_built


def enrich_entity_profiles_with_llm(
    connection: sqlite3.Connection,
    config: RuntimeConfig,
    *,
    force: bool = False,
) -> int:
    """Generate LLM prose summaries for entities and communities.

    Returns:
        Number of summaries generated.
    """
    kg_cfg = config.knowledge_graph
    if not kg_cfg.llm_enrichment or kg_cfg.llm_enrichment_backend == "none":
        _log.info("LLM enrichment disabled")
        return 0

    from lxd.stores.sqlite import load_all_community_reports, load_all_entity_profiles

    enriched = 0

    # Enrich entity profiles
    profiles = load_all_entity_profiles(connection)
    for profile in profiles:
        if profile.llm_summary is not None and not force:
            continue

        prompt = (
            f"Write a 150–300 word prose summary of this entity for instructional design professionals.\n\n"
            f"Entity: {profile.label}\n"
            f"Type: {profile.entity_type}\n"
            f"Domain: {profile.domain}\n\n"
            f"Context:\n{profile.deterministic_summary}\n\n"
            f"Top predicates: {profile.top_predicates_json}\n"
            f"Top claims: {profile.top_claims_json}\n"
        )

        llm_summary = _call_llm_enrichment(prompt, config)
        if llm_summary:
            connection.execute(
                "UPDATE entity_profiles SET llm_summary = ? WHERE entity_id = ?",
                (llm_summary, profile.entity_id),
            )
            connection.commit()
            enriched += 1

    # Enrich community reports
    reports = load_all_community_reports(connection)
    for report in reports:
        if report.llm_summary is not None and not force:
            continue

        prompt = (
            f"Write a 200–400 word narrative summary of this entity community "
            f"for instructional design professionals.\n\n"
            f"Community {report.community_id} ({report.member_count} members)\n\n"
            f"Context:\n{report.deterministic_summary}\n\n"
            f"Top entities: {report.top_entities_json}\n"
            f"Top claims: {report.top_claims_json}\n"
        )

        llm_summary = _call_llm_enrichment(prompt, config)
        if llm_summary:
            connection.execute(
                "UPDATE community_reports SET llm_summary = ? WHERE community_id = ?",
                (llm_summary, report.community_id),
            )
            connection.commit()
            enriched += 1

    _log.info("LLM enrichment complete", enriched=enriched)
    return enriched


def _call_llm_enrichment(prompt: str, config: RuntimeConfig) -> str | None:
    """Call LLM for enrichment summary."""
    kg_cfg = config.knowledge_graph

    if kg_cfg.llm_enrichment_backend == "openai":
        try:
            return _call_openai_enrichment(prompt, config)
        except Exception as exc:
            _log.warning("OpenAI enrichment failed, trying fallback: %s", exc)
            try:
                return _call_ollama_enrichment(prompt, config)
            except Exception as fallback_exc:
                _log.warning("Ollama enrichment fallback failed: %s", fallback_exc)
            return None

    if kg_cfg.llm_enrichment_backend == "ollama":
        try:
            return _call_ollama_enrichment(prompt, config)
        except Exception as exc:
            _log.warning("Ollama enrichment failed: %s", exc)
            return None

    return None


def _call_openai_enrichment(prompt: str, config: RuntimeConfig) -> str | None:
    import openai

    kg_cfg = config.knowledge_graph
    openai_cfg = config.openai
    api_key_env = openai_cfg.api_key_env if openai_cfg else "OPENAI_API_KEY"
    api_key = os.environ.get(api_key_env)
    if not api_key:
        return None

    client = openai.OpenAI(api_key=api_key, timeout=float(kg_cfg.llm_enrichment_timeout_secs))
    response = client.chat.completions.create(
        model=kg_cfg.llm_enrichment_model,
        messages=[
            {
                "role": "system",
                "content": "You are an expert in instructional design and learning science. Write clear, informative summaries.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=kg_cfg.llm_enrichment_temperature,
        max_tokens=500,
    )
    content = response.choices[0].message.content
    return content.strip() if content else None


def _call_ollama_enrichment(prompt: str, config: RuntimeConfig) -> str | None:
    import ollama

    kg_cfg = config.knowledge_graph
    client = ollama.Client(
        host=str(config.ollama.url),
        timeout=float(kg_cfg.llm_enrichment_timeout_secs),
    )
    response = client.chat(
        model=kg_cfg.llm_enrichment_fallback_model,
        messages=[
            {
                "role": "system",
                "content": "You are an expert in instructional design and learning science. Write clear, informative summaries.",
            },
            {"role": "user", "content": prompt},
        ],
        options={"temperature": kg_cfg.llm_enrichment_temperature},
    )
    content = (
        response["message"]["content"] if isinstance(response, dict) else response.message.content
    )
    return content.strip() if content else None


def _rank_position(
    entity_id: str,
    centrality: dict[str, CentralityScores],
    metric: str,
) -> int:
    """Return 1-based rank position for an entity on a given metric."""
    values = sorted(
        ((eid, getattr(scores, metric)) for eid, scores in centrality.items()),
        key=lambda pair: pair[1],
        reverse=True,
    )
    for idx, (eid, _) in enumerate(values):
        if eid == entity_id:
            return idx + 1
    return len(values)
