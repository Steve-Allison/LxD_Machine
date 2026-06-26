"""Build deterministic entity profiles and optional LLM enrichment."""

import asyncio
import json
import sqlite3
from datetime import UTC, datetime
from typing import Any

import structlog

from lxd.domain.ids import blake3_hex
from lxd.ingest.llm_client import call_with_fallback_async, run_concurrent_extraction
from lxd.ontology.entity_graph import CentralityScores
from lxd.settings.models import RuntimeConfig
from lxd.stores.models import CommunityReportRecord, EntityProfileRecord
from lxd.stores.sqlite.chunks import load_entity_mention_stats
from lxd.stores.sqlite.claims import load_claims_for_entities
from lxd.stores.sqlite.kg_profiles import (
    load_all_community_reports,
    load_all_entity_profiles,
    load_community_members,
    load_community_report_source_hashes,
    load_entity_profile_source_hashes,
    upsert_community_report,
    upsert_entity_profile,
)
from lxd.stores.sqlite.kg_relations import load_top_predicates_for_entity

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
    chunk_ids_by_entity = _load_chunk_ids_by_entity(connection)

    # Precompute rank dicts once. Profile source_hash uses ranks (small ints)
    # rather than raw centrality floats, so imperceptible 4th-decimal score
    # shifts that don't reshuffle the ranking don't trigger a profile rebuild
    # (and the cascading LLM re-enrichment).
    pr_ranks = _compute_ranks(centrality, "pagerank")
    bt_ranks = _compute_ranks(centrality, "betweenness")
    cl_ranks = _compute_ranks(centrality, "closeness")
    total_entities = len(centrality)

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

        # Chunk IDs for source hash are preloaded in one grouped query above
        # to avoid an N+1 pattern across the ontology.
        chunk_ids = chunk_ids_by_entity.get(entity_id, [])
        claim_ids = sorted(c.claim_id for c in claims)

        # Source hash composed of rank positions (small ints) rather than
        # raw centrality floats. The rank position is what the deterministic
        # summary and downstream LLM enrichment actually see; sub-rank float
        # noise contributes nothing semantically but used to trigger a full
        # profile + LLM enrichment rebuild on every graph update.
        pr_rank = pr_ranks.get(entity_id, total_entities)
        bt_rank = bt_ranks.get(entity_id, total_entities)
        cl_rank = cl_ranks.get(entity_id, total_entities)
        source_hash = blake3_hex(
            *chunk_ids,
            str(pr_rank),
            str(bt_rank),
            str(cl_rank),
            str(in_degree),
            str(out_degree),
            str(community_id),
            *claim_ids,
        )

        # Incremental: skip if source hash unchanged
        if entity_id in existing_hashes and existing_hashes[entity_id] == source_hash:
            continue

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


def _load_chunk_ids_by_entity(connection: sqlite3.Connection) -> dict[str, list[str]]:
    """Return ``{entity_id: sorted unique chunk_ids}`` in a single query.

    Replaces the per-entity ``SELECT DISTINCT chunk_id FROM mention_rows ...``
    loop in :func:`build_entity_profiles`. Chunk IDs are sorted lexicographically
    within each entity so callers can feed them directly into ``blake3_hex``
    without an additional sort.
    """
    rows = connection.execute(
        """
        SELECT entity_id, chunk_id
        FROM mention_rows
        GROUP BY entity_id, chunk_id
        ORDER BY entity_id, chunk_id
        """
    ).fetchall()
    grouped: dict[str, list[str]] = {}
    for row in rows:
        grouped.setdefault(str(row["entity_id"]), []).append(str(row["chunk_id"]))
    return grouped


def build_community_reports(
    connection: sqlite3.Connection,
    community_assignments: dict[str, int],
    centrality: dict[str, CentralityScores],
    *,
    force: bool = False,
    community_level: int = 0,
    parent_of: dict[int, int | None] | None = None,
) -> int:
    """Build deterministic community reports for one level of the hierarchy.

    ``community_level`` defaults to 0 (finest) — callers that produced a
    hierarchical partition pass the level and a ``parent_of`` map so each
    report records its parent. For single-level builds (default), the parent
    field is null.

    Returns:
        Number of reports built.
    """
    # Group entities by community
    communities: dict[int, list[str]] = {}
    for entity_id, community_id in community_assignments.items():
        communities.setdefault(community_id, []).append(entity_id)

    existing_hashes = load_community_report_source_hashes(connection) if not force else {}

    timestamp = datetime.now(UTC).isoformat()
    reports_built = 0
    parent_lookup = parent_of or {}

    for community_id, member_ids in communities.items():
        sorted_members = sorted(member_ids)

        # Compute source_hash *before* the deterministic-summary build so an
        # unchanged community skips both the work and the upsert. The upsert
        # clobbers llm_summary to NULL, which used to cascade an LLM
        # re-enrichment on every report regardless of whether anything had
        # actually changed; the early skip stops the cascade.
        if sorted_members:
            placeholders = ",".join("?" * len(sorted_members))
            hash_rows = connection.execute(
                f"SELECT entity_id, source_hash FROM entity_profiles "
                f"WHERE entity_id IN ({placeholders})",
                sorted_members,
            ).fetchall()
            member_hashes = [str(row["source_hash"]) for row in hash_rows]
        else:
            member_hashes = []

        source_hash = blake3_hex(*sorted_members, *sorted(member_hashes))

        report_key = (community_id, community_level)
        if existing_hashes.get(report_key) == source_hash:
            continue

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
            community_level=community_level,
            parent_community_id=parent_lookup.get(community_id),
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
    """Generate LLM prose summaries for entities and communities (async concurrent).

    Returns:
        Number of summaries generated.
    """
    return asyncio.run(_enrich_async(connection, config, force=force))


_ENRICHMENT_SYSTEM_PROMPT = (
    "You are an expert in instructional design and learning science. "
    "Write clear, informative summaries."
)


async def _enrich_async(
    connection: sqlite3.Connection,
    config: RuntimeConfig,
    *,
    force: bool = False,
) -> int:
    """Async concurrent enrichment of profiles and community reports."""
    kg_cfg = config.knowledge_graph
    api_key_env = config.openai.api_key_env if config.openai else "OPENAI_API_KEY"
    ollama_host = str(config.ollama.url)
    enriched = 0

    # --- Entity profiles ---
    profiles = load_all_entity_profiles(connection)
    profiles_to_enrich = [p for p in profiles if force or p.llm_summary is None]

    if profiles_to_enrich:

        async def _enrich_profile(profile: EntityProfileRecord) -> tuple[str, str | None]:
            prompt = (
                f"Write a 150–300 word prose summary of this entity "
                f"for instructional design professionals.\n\n"
                f"Entity: {profile.label}\n"
                f"Type: {profile.entity_type}\n"
                f"Domain: {profile.domain}\n\n"
                f"Context:\n{profile.deterministic_summary}\n\n"
                f"Top predicates: {profile.top_predicates_json}\n"
                f"Top claims: {profile.top_claims_json}\n"
            )
            raw = await call_with_fallback_async(
                system_prompt=_ENRICHMENT_SYSTEM_PROMPT,
                user_prompt=prompt,
                primary_backend=kg_cfg.llm_enrichment_backend,
                openai_model=kg_cfg.llm_enrichment_model,
                ollama_model=kg_cfg.llm_enrichment_fallback_model,
                temperature=kg_cfg.llm_enrichment_temperature,
                openai_timeout=float(kg_cfg.llm_enrichment_timeout_secs),
                ollama_timeout=float(kg_cfg.llm_enrichment_timeout_secs),
                max_tokens=500,
                api_key_env=api_key_env,
                ollama_host=ollama_host,
                ollama_format=None,
            )
            summary = raw.strip() if raw else None
            return (profile.entity_id, summary)

        def _commit_profiles(results: list[tuple[str, str | None]]) -> None:
            nonlocal enriched
            for entity_id, summary in results:
                if summary:
                    connection.execute(
                        "UPDATE entity_profiles SET llm_summary = ? WHERE entity_id = ?",
                        (summary, entity_id),
                    )
                    enriched += 1
            connection.commit()

        await run_concurrent_extraction(
            profiles_to_enrich,
            _enrich_profile,
            max_concurrent=kg_cfg.claim_extraction_max_concurrent,
            sub_batch_size=kg_cfg.claim_extraction_sub_batch_size,
            commit_fn=_commit_profiles,
            label="profile_enrichment",
        )

    # --- Community reports ---
    reports = load_all_community_reports(connection)
    reports_to_enrich = [r for r in reports if force or r.llm_summary is None]

    if reports_to_enrich:

        async def _enrich_report(report: CommunityReportRecord) -> tuple[int, str | None]:
            prompt = (
                f"Write a 200–400 word narrative summary of this entity community "
                f"for instructional design professionals.\n\n"
                f"Community {report.community_id} ({report.member_count} members)\n\n"
                f"Context:\n{report.deterministic_summary}\n\n"
                f"Top entities: {report.top_entities_json}\n"
                f"Top claims: {report.top_claims_json}\n"
            )
            raw = await call_with_fallback_async(
                system_prompt=_ENRICHMENT_SYSTEM_PROMPT,
                user_prompt=prompt,
                primary_backend=kg_cfg.llm_enrichment_backend,
                openai_model=kg_cfg.llm_enrichment_model,
                ollama_model=kg_cfg.llm_enrichment_fallback_model,
                temperature=kg_cfg.llm_enrichment_temperature,
                openai_timeout=float(kg_cfg.llm_enrichment_timeout_secs),
                ollama_timeout=float(kg_cfg.llm_enrichment_timeout_secs),
                max_tokens=600,
                api_key_env=api_key_env,
                ollama_host=ollama_host,
                ollama_format=None,
            )
            summary = raw.strip() if raw else None
            return (report.community_id, summary)

        def _commit_reports(results: list[tuple[int, str | None]]) -> None:
            nonlocal enriched
            for community_id, summary in results:
                if summary:
                    connection.execute(
                        "UPDATE community_reports SET llm_summary = ? WHERE community_id = ?",
                        (summary, community_id),
                    )
                    enriched += 1
            connection.commit()

        await run_concurrent_extraction(
            reports_to_enrich,
            _enrich_report,
            max_concurrent=kg_cfg.claim_extraction_max_concurrent,
            sub_batch_size=kg_cfg.claim_extraction_sub_batch_size,
            commit_fn=_commit_reports,
            label="community_enrichment",
        )

    _log.info("LLM enrichment complete", enriched=enriched)
    return enriched


def _compute_ranks(
    centrality: dict[str, CentralityScores],
    metric: str,
) -> dict[str, int]:
    """Return ``{entity_id: 1-based rank position}`` for one centrality metric.

    Ties resolve in the underlying sort's stable order. Used by
    :func:`build_entity_profiles` to feed rank positions into the profile
    ``source_hash`` instead of raw floats — sub-rank float noise no longer
    triggers a profile rebuild.
    """
    ranked = sorted(
        centrality.items(),
        key=lambda item: getattr(item[1], metric),
        reverse=True,
    )
    return {entity_id: idx + 1 for idx, (entity_id, _) in enumerate(ranked)}
