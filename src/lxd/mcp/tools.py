"""Define MCP tools that expose corpus and ontology operations."""

from __future__ import annotations

from typing import Any

from lxd.app.bootstrap import AppContext
from lxd.app.status import load_committed_status
from lxd.ingest.pipeline import IngestPlan
from lxd.ontology.graph import direct_neighbors
from lxd.retrieval.expansion import expand_entity_ids
from lxd.retrieval.query_pipeline import search_chunks
from lxd.stores.sqlite import (
    build_store_paths,
    connect_sqlite,
    find_chunks_by_entity_mentions,
    initialize_schema,
    load_corpus_relations_for_entity,
)


def corpus_status_tool(app_context: AppContext, plan: IngestPlan) -> dict[str, object]:
    """Return corpus and ontology status for MCP clients.

    Args:
        app_context: Application context that provides runtime configuration.
        plan: Precomputed ingest plan and ontology snapshot.

    Returns:
        Corpus and ontology status payload for MCP responses.
    """
    store_paths = build_store_paths(app_context.config.paths.data_path)
    if store_paths.sqlite_path.exists():
        connection = connect_sqlite(store_paths.sqlite_path)
        try:
            initialize_schema(connection)
            status_snapshot = load_committed_status(
                connection,
                config=app_context.config,
                plan_provider=lambda: plan,
            )
        finally:
            connection.close()
        if status_snapshot is not None:
            summary = status_snapshot.summary
            return {
                "corpus_counts": {
                    "total": summary.corpus_file_count,
                    "text": summary.text_file_count,
                    "asset": summary.asset_file_count,
                },
                "retrieval_role_counts": summary.retrieval_role_counts,
                "chunk_count": summary.chunk_count,
                "mention_count": summary.mention_count,
                "ontology_file_count": summary.ontology_file_count,
                "entity_count": status_snapshot.entity_count,
                "matcher_term_count": summary.matcher_term_count,
                "ontology_snapshot_hash": summary.ontology_snapshot_hash,
                "matcher_termset_hash": summary.matcher_termset_hash,
                "ontology_coverage_path_count": summary.ontology_coverage_path_count,
                "ontology_graph_relation_count": summary.ontology_graph_relation_count,
                "ontology_validation_issue_count": summary.ontology_validation_issue_count,
                "ontology_validation_issue_samples": summary.ontology_validation_issue_samples,
                "config_drift_warnings": summary.config_drift_warnings,
            }
    asset_count = sum(1 for item in plan.scanned_files if item.source_type == "image_png")
    return {
        "corpus_counts": {
            "total": len(plan.scanned_files),
            "text": len(plan.scanned_files) - asset_count,
            "asset": asset_count,
        },
        "retrieval_role_counts": {"searchable": 0, "asset_only": 0, "not_searchable": 0},
        "chunk_count": 0,
        "mention_count": 0,
        "ontology_file_count": len(plan.ontology.sources),
        "entity_count": len(plan.ontology.entity_definitions),
        "matcher_term_count": len(plan.ontology.matcher_records),
        "ontology_snapshot_hash": plan.ontology.snapshot_hash,
        "matcher_termset_hash": plan.ontology.matcher_termset_hash,
        "ontology_coverage_path_count": plan.ontology.coverage_report.discovered_path_count,
        "ontology_graph_relation_count": len(plan.ontology.relation_records),
        "ontology_validation_issue_count": len(plan.ontology.validation_issues),
        "ontology_validation_issue_samples": [
            issue.message for issue in plan.ontology.validation_issues[:10]
        ],
        "config_drift_warnings": [],
    }


def get_entity_types_tool(plan: IngestPlan) -> list[str]:
    """List canonical ontology entity IDs.

    Args:
        plan: Precomputed ingest plan and ontology snapshot.

    Returns:
        Sorted canonical ontology entity IDs.
    """
    return sorted(entity["canonical_id"] for entity in plan.ontology.entity_definitions)


def get_related_concepts_tool(plan: IngestPlan, entity_id: str) -> list[dict[str, Any]]:
    """Return direct ontology neighbors for an entity.

    Args:
        plan: Precomputed ingest plan and ontology snapshot.
        entity_id: Canonical ontology entity identifier.

    Returns:
        Direct ontology neighbor entries for the entity.
    """
    _require_non_empty(entity_id, "entity_id")
    if entity_id not in plan.ontology.graph:
        return []
    return direct_neighbors(plan.ontology.graph, entity_id)


def search_corpus_tool(
    app_context: AppContext,
    terms: str,
    domain: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    """Search indexed chunks for query terms.

    Args:
        app_context: Application context that provides runtime configuration.
        terms: User query string to search for.
        domain: Optional source domain filter.
        limit: Maximum number of records to return.

    Returns:
        Matching chunk records formatted for MCP.
    """
    outcome = search_chunks(
        question=terms,
        config=app_context.config,
        domain=domain,
        limit=limit,
    )
    return [
        {
            "chunk_id": item.chunk_id,
            "document_id": item.document_id,
            "citation_label": item.citation_label,
            "source_rel_path": item.source_rel_path,
            "score": item.score,
            "text": item.text,
            "metadata_json": item.metadata_json,
        }
        for item in outcome.ranked
    ]


def find_documents_for_concept_tool(
    app_context: AppContext,
    plan: IngestPlan,
    entity_id: str,
    hops: int = 1,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Find chunks mentioning an entity and related concepts.

    Args:
        app_context: Application context that provides runtime configuration.
        plan: Precomputed ingest plan and ontology snapshot.
        entity_id: Canonical ontology entity identifier.
        hops: Graph expansion depth from the seed entity.
        limit: Maximum number of records to return.

    Returns:
        Matching chunks for the entity expansion set.
    """
    _require_non_empty(entity_id, "entity_id")
    if entity_id not in plan.ontology.graph:
        return []
    related_ids = expand_entity_ids(
        plan.ontology.graph,
        [entity_id],
        hops=hops,
        max_entities=50,
    )
    all_entity_ids = list({entity_id, *related_ids})
    store_paths = build_store_paths(app_context.config.paths.data_path)
    connection = connect_sqlite(store_paths.sqlite_path)
    try:
        initialize_schema(connection)
        results = find_chunks_by_entity_mentions(connection, all_entity_ids, limit=limit)
    finally:
        connection.close()
    return [
        {
            "chunk_id": item.chunk_id,
            "document_id": item.document_id,
            "citation_label": item.citation_label,
            "source_rel_path": item.source_rel_path,
            "score": item.score,
            "entity_match_count": item.entity_match_count,
            "matched_from_total": item.total_entity_ids,
            "text": item.text,
            "metadata_json": item.metadata_json,
        }
        for item in results
    ]


def get_corpus_relations_tool(
    app_context: AppContext,
    entity_id: str,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Load extracted corpus relations for an entity.

    Args:
        app_context: Application context that provides runtime configuration.
        entity_id: Canonical ontology entity identifier.
        limit: Maximum number of records to return.

    Returns:
        Relation rows involving the requested entity.
    """
    _require_non_empty(entity_id, "entity_id")
    store_paths = build_store_paths(app_context.config.paths.data_path)
    if not store_paths.sqlite_path.exists():
        return []
    connection = connect_sqlite(store_paths.sqlite_path)
    try:
        initialize_schema(connection)
        return load_corpus_relations_for_entity(connection, entity_id, limit=limit)
    finally:
        connection.close()


def get_entity_summary_tool(app_context: AppContext, entity_id: str) -> dict[str, Any]:
    """Return the full entity profile for an entity."""
    _require_non_empty(entity_id, "entity_id")
    store_paths = build_store_paths(app_context.config.paths.data_path)
    if not store_paths.sqlite_path.exists():
        return {}
    connection = connect_sqlite(store_paths.sqlite_path)
    try:
        initialize_schema(connection)
        from lxd.stores.sqlite import load_entity_profile

        profile = load_entity_profile(connection, entity_id)
        if profile is None:
            return {}
        return {
            "entity_id": profile.entity_id,
            "label": profile.label,
            "entity_type": profile.entity_type,
            "domain": profile.domain,
            "aliases": profile.aliases_json,
            "deterministic_summary": profile.deterministic_summary,
            "llm_summary": profile.llm_summary,
            "chunk_count": profile.chunk_count,
            "doc_count": profile.doc_count,
            "mention_count": profile.mention_count,
            "claim_count": profile.claim_count,
            "top_predicates": profile.top_predicates_json,
            "top_claims": profile.top_claims_json,
            "pagerank": profile.pagerank,
            "betweenness": profile.betweenness,
            "closeness": profile.closeness,
            "in_degree": profile.in_degree,
            "out_degree": profile.out_degree,
            "eigenvector": profile.eigenvector,
            "community_id": profile.community_id,
        }
    finally:
        connection.close()


def get_community_context_tool(app_context: AppContext, entity_id: str) -> dict[str, Any]:
    """Return the community report for an entity's community."""
    _require_non_empty(entity_id, "entity_id")
    store_paths = build_store_paths(app_context.config.paths.data_path)
    if not store_paths.sqlite_path.exists():
        return {}
    connection = connect_sqlite(store_paths.sqlite_path)
    try:
        initialize_schema(connection)
        from lxd.stores.sqlite import load_community_report, load_entity_profile

        profile = load_entity_profile(connection, entity_id)
        if profile is None or profile.community_id is None:
            return {}
        report = load_community_report(connection, profile.community_id)
        if report is None:
            return {}
        return {
            "community_id": report.community_id,
            "member_count": report.member_count,
            "member_entity_ids": report.member_entity_ids_json,
            "deterministic_summary": report.deterministic_summary,
            "llm_summary": report.llm_summary,
            "top_entities": report.top_entities_json,
            "top_claims": report.top_claims_json,
            "intra_community_edge_count": report.intra_community_edge_count,
        }
    finally:
        connection.close()


def get_similar_entities_tool(
    app_context: AppContext, entity_id: str, limit: int = 10
) -> list[dict[str, Any]]:
    """Return similar entities via LanceDB vector search on entity embeddings."""
    _require_non_empty(entity_id, "entity_id")
    store_paths = build_store_paths(app_context.config.paths.data_path)
    if not store_paths.sqlite_path.exists():
        return []
    connection = connect_sqlite(store_paths.sqlite_path)
    try:
        initialize_schema(connection)
        import json

        from lxd.stores.lancedb import connect_lancedb, open_entity_table, search_similar_entities
        from lxd.stores.sqlite import load_chunk_ids_for_entity, load_entity_profile

        profile = load_entity_profile(connection, entity_id)
        if profile is None:
            return []

        # Get entity embedding vector by computing from chunk embeddings
        chunk_ids = load_chunk_ids_for_entity(connection, entity_id, limit=20)
        if not chunk_ids:
            return []

        vector_size = app_context.config.models.embed_dims
        placeholders = ",".join("?" * len(chunk_ids))
        chunk_rows = connection.execute(
            f"SELECT vector_json FROM chunk_rows WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        ).fetchall()

        vectors: list[list[float]] = []
        for row in chunk_rows:
            vec = json.loads(str(row["vector_json"]))
            if isinstance(vec, list) and len(vec) == vector_size:
                vectors.append(vec)

        if not vectors:
            return []

        query_vector = [sum(v[i] for v in vectors) / len(vectors) for i in range(vector_size)]

        db = connect_lancedb(store_paths.lancedb_path)
        try:
            entity_table = open_entity_table(db, vector_size=vector_size)
            results = search_similar_entities(
                entity_table,
                query_vector=query_vector,
                limit=limit + 1,
            )
            # Exclude self
            return [r for r in results if r["entity_id"] != entity_id][:limit]
        except FileNotFoundError:
            return []
    finally:
        connection.close()


def search_entities_tool(
    app_context: AppContext, query: str, limit: int = 20
) -> list[dict[str, Any]]:
    """Search entity profiles by label/alias substring, ranked by PageRank."""
    _require_non_empty(query, "query")
    store_paths = build_store_paths(app_context.config.paths.data_path)
    if not store_paths.sqlite_path.exists():
        return []
    connection = connect_sqlite(store_paths.sqlite_path)
    try:
        initialize_schema(connection)
        from lxd.stores.sqlite import search_entity_profiles

        profiles = search_entity_profiles(connection, query, limit=limit)
        return [
            {
                "entity_id": p.entity_id,
                "label": p.label,
                "entity_type": p.entity_type,
                "pagerank": p.pagerank,
                "community_id": p.community_id,
                "mention_count": p.mention_count,
            }
            for p in profiles
        ]
    finally:
        connection.close()


def inspect_evidence_tool(app_context: AppContext, relation_id: str) -> list[dict[str, Any]]:
    """Return all evidence records for a canonical relation."""
    _require_non_empty(relation_id, "relation_id")
    store_paths = build_store_paths(app_context.config.paths.data_path)
    if not store_paths.sqlite_path.exists():
        return []
    connection = connect_sqlite(store_paths.sqlite_path)
    try:
        initialize_schema(connection)
        from lxd.stores.sqlite import load_evidence_for_relation

        records = load_evidence_for_relation(connection, relation_id)
        return [
            {
                "evidence_id": r.evidence_id,
                "relation_id": r.relation_id,
                "chunk_id": r.chunk_id,
                "surface_subject": r.surface_subject,
                "surface_object": r.surface_object,
                "evidence_text": r.evidence_text[:500],
                "confidence": r.confidence,
                "extraction_model": r.extraction_model,
            }
            for r in records
        ]
    finally:
        connection.close()


def find_path_between_entities_tool(
    app_context: AppContext,
    plan: IngestPlan,
    source: str,
    target: str,
    max_hops: int = 5,
) -> dict[str, Any]:
    """Find shortest unweighted path between two entities."""
    _require_non_empty(source, "source")
    _require_non_empty(target, "target")
    import networkx as nx

    graph = plan.ontology.graph
    if source not in graph or target not in graph:
        return {"path": [], "edges": [], "hops": 0}
    try:
        path = nx.shortest_path(graph, source, target)
        if len(path) - 1 > max_hops:
            return {"path": [], "edges": [], "hops": 0, "note": f"Path exceeds max_hops={max_hops}"}
        edges: list[dict[str, str]] = []
        for i in range(len(path) - 1):
            edge_data = graph.get_edge_data(path[i], path[i + 1])
            if edge_data:
                first_key = next(iter(edge_data))
                edges.append(
                    {
                        "source": str(path[i]),
                        "target": str(path[i + 1]),
                        "relation_type": str(edge_data[first_key].get("relation_type", "")),
                    }
                )
        return {"path": [str(n) for n in path], "edges": edges, "hops": len(path) - 1}
    except nx.NetworkXNoPath:
        return {"path": [], "edges": [], "hops": 0}


def find_weighted_path_tool(
    app_context: AppContext,
    plan: IngestPlan,
    source: str,
    target: str,
) -> dict[str, Any]:
    """Find confidence-weighted Dijkstra shortest path between two entities."""
    _require_non_empty(source, "source")
    _require_non_empty(target, "target")
    import networkx as nx

    graph = plan.ontology.graph

    if source not in graph or target not in graph:
        return {"path": [], "edges": [], "total_weight": 0.0}

    # Create weighted graph where weight = 1.0 - confidence
    weighted = nx.DiGraph()
    for u, v, data in graph.edges(data=True):
        confidence = data.get("weight", 0.5) if data.get("origin_kind") == "corpus" else 0.5
        weight = 1.0 - max(0.0, min(1.0, confidence))
        if weighted.has_edge(u, v):
            existing = weighted[u][v]["weight"]
            weighted[u][v]["weight"] = min(existing, weight)
        else:
            weighted.add_edge(u, v, weight=weight)

    try:
        path = nx.dijkstra_path(weighted, source, target, weight="weight")
        total_weight = nx.dijkstra_path_length(weighted, source, target, weight="weight")
        edges: list[dict[str, Any]] = []
        for i in range(len(path) - 1):
            edges.append(
                {
                    "source": str(path[i]),
                    "target": str(path[i + 1]),
                    "weight": weighted[path[i]][path[i + 1]]["weight"],
                }
            )
        return {
            "path": [str(n) for n in path],
            "edges": edges,
            "total_weight": total_weight,
        }
    except nx.NetworkXNoPath:
        return {"path": [], "edges": [], "total_weight": 0.0}


def get_hub_entities_tool(app_context: AppContext, limit: int = 20) -> list[dict[str, Any]]:
    """Return top entities by PageRank."""
    store_paths = build_store_paths(app_context.config.paths.data_path)
    if not store_paths.sqlite_path.exists():
        return []
    connection = connect_sqlite(store_paths.sqlite_path)
    try:
        initialize_schema(connection)
        from lxd.stores.sqlite import load_top_entities_by_pagerank

        profiles = load_top_entities_by_pagerank(connection, limit=limit)
        return [
            {
                "entity_id": p.entity_id,
                "label": p.label,
                "pagerank": p.pagerank,
                "community_id": p.community_id,
            }
            for p in profiles
        ]
    finally:
        connection.close()


def find_bridge_entities_tool(app_context: AppContext, limit: int = 20) -> list[dict[str, Any]]:
    """Return top entities by betweenness centrality."""
    store_paths = build_store_paths(app_context.config.paths.data_path)
    if not store_paths.sqlite_path.exists():
        return []
    connection = connect_sqlite(store_paths.sqlite_path)
    try:
        initialize_schema(connection)
        from lxd.stores.sqlite import load_top_entities_by_betweenness

        profiles = load_top_entities_by_betweenness(connection, limit=limit)
        return [
            {
                "entity_id": p.entity_id,
                "label": p.label,
                "betweenness": p.betweenness,
                "community_id": p.community_id,
            }
            for p in profiles
        ]
    finally:
        connection.close()


def find_foundational_entities_tool(
    app_context: AppContext, limit: int = 20
) -> list[dict[str, Any]]:
    """Return top entities by closeness centrality."""
    store_paths = build_store_paths(app_context.config.paths.data_path)
    if not store_paths.sqlite_path.exists():
        return []
    connection = connect_sqlite(store_paths.sqlite_path)
    try:
        initialize_schema(connection)
        from lxd.stores.sqlite import load_top_entities_by_closeness

        profiles = load_top_entities_by_closeness(connection, limit=limit)
        return [
            {
                "entity_id": p.entity_id,
                "label": p.label,
                "closeness": p.closeness,
                "community_id": p.community_id,
            }
            for p in profiles
        ]
    finally:
        connection.close()


def get_entity_graph_stats_tool(app_context: AppContext) -> dict[str, Any]:
    """Return knowledge graph statistics."""
    store_paths = build_store_paths(app_context.config.paths.data_path)
    if not store_paths.sqlite_path.exists():
        return {"graph_version": 0, "entity_profiles": 0, "communities": 0}
    connection = connect_sqlite(store_paths.sqlite_path)
    try:
        initialize_schema(connection)
        from lxd.stores.sqlite import (
            count_canonical_relations,
            count_claims,
            count_communities,
            count_community_reports,
            count_entity_profiles,
            count_relation_evidence,
            load_graph_metadata,
        )

        metadata = load_graph_metadata(connection)
        return {
            "graph_version": int(metadata.get("graph_version", "0")),
            "last_build_at": metadata.get("last_build_at", "never"),
            "entity_profiles": count_entity_profiles(connection),
            "communities": count_communities(connection),
            "community_reports": count_community_reports(connection),
            "canonical_relations": count_canonical_relations(connection),
            "relation_evidence": count_relation_evidence(connection),
            "claims": count_claims(connection),
        }
    finally:
        connection.close()


def search_knowledge_tool(
    app_context: AppContext,
    question: str,
    domain: str | None = None,
) -> dict[str, Any]:
    """Run the full answer pipeline with graph-augmented synthesis."""
    _require_non_empty(question, "question")
    from lxd.retrieval.query_pipeline import answer_question

    envelope = answer_question(question=question, config=app_context.config, domain=domain)
    return {
        "answer_status": envelope.answer_status.value,
        "answer_text": envelope.answer_text,
        "citations": envelope.citations,
        "warnings": envelope.warnings,
        "metadata": envelope.metadata,
    }


def search_knowledge_deep_tool(
    app_context: AppContext,
    question: str,
    domain: str | None = None,
) -> dict[str, Any]:
    """Run the full answer pipeline with graph context data returned alongside the answer."""
    _require_non_empty(question, "question")
    from lxd.retrieval.query_pipeline import answer_question

    envelope = answer_question(question=question, config=app_context.config, domain=domain)

    # Also load graph context data for the matched entities
    matched_entity_ids = envelope.metadata.get("matched_entity_ids", [])
    graph_data: dict[str, Any] = {"level": "none", "entity_profiles": [], "claims": []}

    if isinstance(matched_entity_ids, list) and matched_entity_ids:
        store_paths = build_store_paths(app_context.config.paths.data_path)
        if store_paths.sqlite_path.exists():
            connection = connect_sqlite(store_paths.sqlite_path)
            try:
                initialize_schema(connection)
                from lxd.retrieval.graph_routing import build_graph_context

                context = build_graph_context(connection, matched_entity_ids, app_context.config)
                graph_data = {
                    "level": context.level,
                    "entity_profiles": [
                        {
                            "entity_id": p.entity_id,
                            "label": p.label,
                            "entity_type": p.entity_type,
                            "deterministic_summary": p.deterministic_summary,
                            "llm_summary": p.llm_summary,
                            "pagerank": p.pagerank,
                            "community_id": p.community_id,
                        }
                        for p in context.entity_profiles
                    ],
                    "community_reports": [
                        {
                            "community_id": r.community_id,
                            "member_count": r.member_count,
                            "deterministic_summary": r.deterministic_summary,
                            "llm_summary": r.llm_summary,
                        }
                        for r in context.community_reports
                    ],
                    "claims": [
                        {
                            "claim_text": c.claim_text,
                            "claim_type": c.claim_type,
                            "confidence": c.confidence,
                            "subject_entity_id": c.subject_entity_id,
                            "object_entity_id": c.object_entity_id,
                        }
                        for c in context.claims
                    ],
                }
            finally:
                connection.close()

    return {
        "answer_status": envelope.answer_status.value,
        "answer_text": envelope.answer_text,
        "citations": envelope.citations,
        "warnings": envelope.warnings,
        "metadata": envelope.metadata,
        "graph_context": graph_data,
    }


def get_graph_overview_tool(app_context: AppContext) -> dict[str, Any]:
    """Return knowledge graph overview including stats and build state."""
    store_paths = build_store_paths(app_context.config.paths.data_path)
    if not store_paths.sqlite_path.exists():
        return {
            "graph_version": 0,
            "entity_profiles": 0,
            "communities": 0,
            "community_reports": 0,
            "canonical_relations": 0,
            "relation_evidence": 0,
            "claims": 0,
        }
    connection = connect_sqlite(store_paths.sqlite_path)
    try:
        initialize_schema(connection)
        from lxd.stores.sqlite import (
            count_canonical_relations,
            count_claims,
            count_communities,
            count_community_reports,
            count_entity_profiles,
            count_relation_evidence,
            load_graph_metadata,
        )

        metadata = load_graph_metadata(connection)
        return {
            "graph_version": int(metadata.get("graph_version", "0")),
            "last_build_at": metadata.get("last_build_at", "never"),
            "community_algorithm": metadata.get(
                "community_algorithm",
                app_context.config.knowledge_graph.community_algorithm,
            ),
            "entity_profiles": count_entity_profiles(connection),
            "communities": count_communities(connection),
            "community_reports": count_community_reports(connection),
            "canonical_relations": count_canonical_relations(connection),
            "relation_evidence": count_relation_evidence(connection),
            "claims": count_claims(connection),
        }
    finally:
        connection.close()


def _require_non_empty(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must be non-empty.")
