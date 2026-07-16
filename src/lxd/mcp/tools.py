"""Define MCP tools that expose corpus and ontology operations."""

from itertools import pairwise

import networkx as nx

from lxd.app.bootstrap import AppContext
from lxd.app.status import load_committed_status
from lxd.ingest.pipeline.orchestrator import IngestPlan
from lxd.mcp.models import (
    BridgeEntity,
    ChunkSearchResult,
    CommunityContext,
    ConceptDocumentMatch,
    CorpusCounts,
    CorpusRelation,
    CorpusStatusResponse,
    EntityGraphStats,
    EntityNeighbor,
    EntitySearchResult,
    EntitySummary,
    FoundationalEntity,
    GraphContextClaim,
    GraphContextCommunityReport,
    GraphContextData,
    GraphContextEntityProfile,
    GraphOverview,
    HubEntity,
    KnowledgeAnswer,
    KnowledgeAnswerDeep,
    KnowledgeAnswerMetadata,
    PathBetweenEntities,
    PathEdge,
    RelationEvidence,
    SentenceCitationView,
    SimilarEntity,
    WeightedEdge,
    WeightedPath,
)
from lxd.ontology.graph import direct_neighbors
from lxd.retrieval.expansion import expand_entity_ids
from lxd.retrieval.graph_routing import build_graph_context
from lxd.retrieval.query_pipeline import answer_question, search_chunks
from lxd.stores.lancedb import (
    connect_lancedb,
    load_vectors_by_chunk_ids,
    open_chunk_table,
    open_entity_table,
    search_similar_entities,
)
from lxd.stores.sqlite._pool import pooled_connection
from lxd.stores.sqlite.chunks import (
    find_chunks_by_entity_mentions,
    load_chunk_ids_for_entity,
    load_corpus_relations_for_entity,
)
from lxd.stores.sqlite.claims import count_claims
from lxd.stores.sqlite.connection import build_store_paths
from lxd.stores.sqlite.kg_profiles import (
    count_communities,
    count_community_reports,
    count_entity_profiles,
    load_community_report,
    load_entity_profile,
    load_top_entities_by_betweenness,
    load_top_entities_by_closeness,
    load_top_entities_by_pagerank,
    search_entity_profiles,
)
from lxd.stores.sqlite.kg_relations import (
    count_canonical_relations,
    count_relation_evidence,
    load_evidence_for_relation,
    load_graph_metadata,
)


def corpus_status_tool(app_context: AppContext, plan: IngestPlan) -> CorpusStatusResponse:
    """Return corpus and ontology status for MCP clients."""
    store_paths = build_store_paths(app_context.config.paths.data_path)
    if store_paths.sqlite_path.exists():
        with pooled_connection(store_paths.sqlite_path) as connection:
            status_snapshot = load_committed_status(
                connection,
                config=app_context.config,
                plan_provider=lambda: plan,
            )
        if status_snapshot is not None:
            summary = status_snapshot.summary
            return CorpusStatusResponse(
                corpus_counts=CorpusCounts(
                    total=summary.corpus_file_count,
                    text=summary.text_file_count,
                    asset=summary.asset_file_count,
                ),
                retrieval_role_counts=summary.retrieval_role_counts,
                chunk_count=summary.chunk_count,
                mention_count=summary.mention_count,
                ontology_file_count=summary.ontology_file_count,
                entity_count=status_snapshot.entity_count,
                matcher_term_count=summary.matcher_term_count,
                ontology_snapshot_hash=summary.ontology_snapshot_hash,
                matcher_termset_hash=summary.matcher_termset_hash,
                ontology_coverage_path_count=summary.ontology_coverage_path_count,
                ontology_graph_relation_count=summary.ontology_graph_relation_count,
                ontology_validation_issue_count=summary.ontology_validation_issue_count,
                ontology_validation_issue_samples=summary.ontology_validation_issue_samples,
                config_drift_warnings=summary.config_drift_warnings,
            )
    asset_count = sum(1 for item in plan.scanned_files if item.source_type == "image_png")
    return CorpusStatusResponse(
        corpus_counts=CorpusCounts(
            total=len(plan.scanned_files),
            text=len(plan.scanned_files) - asset_count,
            asset=asset_count,
        ),
        retrieval_role_counts={"searchable": 0, "asset_only": 0, "not_searchable": 0},
        chunk_count=0,
        mention_count=0,
        ontology_file_count=len(plan.ontology.sources),
        entity_count=len(plan.ontology.entity_definitions),
        matcher_term_count=len(plan.ontology.matcher_records),
        ontology_snapshot_hash=plan.ontology.snapshot_hash,
        matcher_termset_hash=plan.ontology.matcher_termset_hash,
        ontology_coverage_path_count=plan.ontology.coverage_report.discovered_path_count,
        ontology_graph_relation_count=len(plan.ontology.relation_records),
        ontology_validation_issue_count=len(plan.ontology.validation_issues),
        ontology_validation_issue_samples=[
            issue.message for issue in plan.ontology.validation_issues[:10]
        ],
        config_drift_warnings=[],
    )


def get_entity_types_tool(plan: IngestPlan) -> list[str]:
    """List canonical ontology entity IDs."""
    return sorted(entity["canonical_id"] for entity in plan.ontology.entity_definitions)


def get_related_concepts_tool(plan: IngestPlan, entity_id: str) -> list[EntityNeighbor]:
    """Return direct ontology neighbors for an entity."""
    _require_non_empty(entity_id, "entity_id")
    if entity_id not in plan.ontology.graph:
        return []
    return [
        EntityNeighbor(
            entity_id=str(n["entity_id"]),
            relation=str(n["relation"]),
            direction=str(n["direction"]),
        )
        for n in direct_neighbors(plan.ontology.graph, entity_id)
    ]


def search_corpus_tool(
    app_context: AppContext,
    terms: str,
    domain: str | None,
    limit: int,
) -> list[ChunkSearchResult]:
    """Search indexed chunks for query terms."""
    outcome = search_chunks(
        question=terms,
        config=app_context.config,
        domain=domain,
        limit=limit,
    )
    return [
        ChunkSearchResult(
            chunk_id=item.chunk_id,
            document_id=item.document_id,
            citation_label=item.citation_label,
            source_rel_path=item.source_rel_path,
            score=item.score,
            text=item.text,
            metadata_json=item.metadata_json,
        )
        for item in outcome.ranked
    ]


def find_documents_for_concept_tool(
    app_context: AppContext,
    plan: IngestPlan,
    entity_id: str,
    hops: int = 1,
    limit: int = 10,
) -> list[ConceptDocumentMatch]:
    """Find chunks mentioning an entity and related concepts."""
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
    with pooled_connection(store_paths.sqlite_path) as connection:
        results = find_chunks_by_entity_mentions(connection, all_entity_ids, limit=limit)
    return [
        ConceptDocumentMatch(
            chunk_id=item.chunk_id,
            document_id=item.document_id,
            citation_label=item.citation_label,
            source_rel_path=item.source_rel_path,
            score=item.score,
            entity_match_count=item.entity_match_count,
            matched_from_total=item.total_entity_ids,
            text=item.text,
            metadata_json=item.metadata_json,
        )
        for item in results
    ]


def get_corpus_relations_tool(
    app_context: AppContext,
    entity_id: str,
    limit: int = 50,
) -> list[CorpusRelation]:
    """Load extracted corpus relations for an entity."""
    _require_non_empty(entity_id, "entity_id")
    store_paths = build_store_paths(app_context.config.paths.data_path)
    if not store_paths.sqlite_path.exists():
        return []
    with pooled_connection(store_paths.sqlite_path) as connection:
        rows = load_corpus_relations_for_entity(connection, entity_id, limit=limit)
    return [
        CorpusRelation(
            subject=str(row["subject"]),
            predicate=str(row["predicate"]),
            object=str(row["object"]),
            confidence=float(row["confidence"]),
            source_rel_path=str(row["source_rel_path"]),
            chunk_id=str(row["chunk_id"]),
        )
        for row in rows
    ]


def get_entity_summary_tool(app_context: AppContext, entity_id: str) -> EntitySummary | None:
    """Return the full entity profile for an entity."""
    _require_non_empty(entity_id, "entity_id")
    store_paths = build_store_paths(app_context.config.paths.data_path)
    if not store_paths.sqlite_path.exists():
        return None
    with pooled_connection(store_paths.sqlite_path) as connection:
        profile = load_entity_profile(connection, entity_id)
    if profile is None:
        return None
    return EntitySummary(
        entity_id=profile.entity_id,
        label=profile.label,
        entity_type=profile.entity_type,
        domain=profile.domain,
        aliases=profile.aliases_json,
        deterministic_summary=profile.deterministic_summary,
        llm_summary=profile.llm_summary,
        chunk_count=profile.chunk_count,
        doc_count=profile.doc_count,
        mention_count=profile.mention_count,
        claim_count=profile.claim_count,
        top_predicates=profile.top_predicates_json,
        top_claims=profile.top_claims_json,
        pagerank=profile.pagerank,
        betweenness=profile.betweenness,
        closeness=profile.closeness,
        in_degree=profile.in_degree,
        out_degree=profile.out_degree,
        eigenvector=profile.eigenvector,
        community_id=profile.community_id,
    )


def get_community_context_tool(app_context: AppContext, entity_id: str) -> CommunityContext | None:
    """Return the community report for an entity's community."""
    _require_non_empty(entity_id, "entity_id")
    store_paths = build_store_paths(app_context.config.paths.data_path)
    if not store_paths.sqlite_path.exists():
        return None
    with pooled_connection(store_paths.sqlite_path) as connection:
        profile = load_entity_profile(connection, entity_id)
        if profile is None or profile.community_id is None:
            return None
        report = load_community_report(connection, profile.community_id)
    if report is None:
        return None
    return CommunityContext(
        community_id=report.community_id,
        member_count=report.member_count,
        member_entity_ids=report.member_entity_ids_json,
        deterministic_summary=report.deterministic_summary,
        llm_summary=report.llm_summary,
        top_entities=report.top_entities_json,
        top_claims=report.top_claims_json,
        intra_community_edge_count=report.intra_community_edge_count,
    )


def get_similar_entities_tool(
    app_context: AppContext, entity_id: str, limit: int = 10
) -> list[SimilarEntity]:
    """Return similar entities via LanceDB vector search on entity embeddings."""
    _require_non_empty(entity_id, "entity_id")
    store_paths = build_store_paths(app_context.config.paths.data_path)
    if not store_paths.sqlite_path.exists():
        return []
    with pooled_connection(store_paths.sqlite_path) as connection:
        profile = load_entity_profile(connection, entity_id)
        if profile is None:
            return []

        chunk_ids = load_chunk_ids_for_entity(connection, entity_id, limit=20)
        if not chunk_ids:
            return []

        vector_size = app_context.config.models.embed_dims
        db = connect_lancedb(store_paths.lancedb_path)
        try:
            chunk_table = open_chunk_table(db, vector_size=vector_size)
        except FileNotFoundError:
            return []
        vectors_by_id = load_vectors_by_chunk_ids(chunk_table, chunk_ids)
        vectors = [v for v in vectors_by_id.values() if len(v) == vector_size]
        if not vectors:
            return []

        query_vector = [sum(v[i] for v in vectors) / len(vectors) for i in range(vector_size)]

        try:
            entity_table = open_entity_table(db, vector_size=vector_size)
            results = search_similar_entities(
                entity_table,
                query_vector=query_vector,
                limit=limit + 1,
            )
        except FileNotFoundError:
            return []
        return [
            SimilarEntity(
                entity_id=str(r["entity_id"]),
                label=str(r["label"]),
                community_id=r["community_id"],
                score=float(r["score"]),
            )
            for r in results
            if r["entity_id"] != entity_id
        ][:limit]


def search_entities_tool(
    app_context: AppContext, query: str, limit: int = 20
) -> list[EntitySearchResult]:
    """Search entity profiles by label/alias substring, ranked by PageRank."""
    _require_non_empty(query, "query")
    store_paths = build_store_paths(app_context.config.paths.data_path)
    if not store_paths.sqlite_path.exists():
        return []
    with pooled_connection(store_paths.sqlite_path) as connection:
        profiles = search_entity_profiles(connection, query, limit=limit)
    return [
        EntitySearchResult(
            entity_id=p.entity_id,
            label=p.label,
            entity_type=p.entity_type,
            pagerank=p.pagerank,
            community_id=p.community_id,
            mention_count=p.mention_count,
        )
        for p in profiles
    ]


def inspect_evidence_tool(app_context: AppContext, relation_id: str) -> list[RelationEvidence]:
    """Return all evidence records for a canonical relation."""
    _require_non_empty(relation_id, "relation_id")
    store_paths = build_store_paths(app_context.config.paths.data_path)
    if not store_paths.sqlite_path.exists():
        return []
    with pooled_connection(store_paths.sqlite_path) as connection:
        records = load_evidence_for_relation(connection, relation_id)
    return [
        RelationEvidence(
            evidence_id=r.evidence_id,
            relation_id=r.relation_id,
            chunk_id=r.chunk_id,
            surface_subject=r.surface_subject,
            surface_object=r.surface_object,
            evidence_text=r.evidence_text[:500],
            confidence=r.confidence,
            extraction_model=r.extraction_model,
        )
        for r in records
    ]


def find_path_between_entities_tool(
    plan: IngestPlan,
    source: str,
    target: str,
    max_hops: int = 5,
) -> PathBetweenEntities:
    """Find shortest unweighted path between two entities."""
    _require_non_empty(source, "source")
    _require_non_empty(target, "target")

    graph = plan.ontology.graph
    if source not in graph or target not in graph:
        return PathBetweenEntities(path=[], edges=[], hops=0)
    try:
        path = nx.shortest_path(graph, source, target)
    except nx.NetworkXNoPath:
        return PathBetweenEntities(path=[], edges=[], hops=0)
    if len(path) - 1 > max_hops:
        return PathBetweenEntities(
            path=[],
            edges=[],
            hops=0,
            note=f"Path exceeds max_hops={max_hops}",
        )
    edges: list[PathEdge] = []
    for u, v in pairwise(path):
        edge_data = graph.get_edge_data(u, v)
        if edge_data:
            first_key = next(iter(edge_data))
            edges.append(
                PathEdge(
                    source=str(u),
                    target=str(v),
                    relation_type=str(edge_data[first_key].get("relation_type", "")),
                )
            )
    return PathBetweenEntities(
        path=[str(n) for n in path],
        edges=edges,
        hops=len(path) - 1,
    )


def find_weighted_path_tool(
    plan: IngestPlan,
    source: str,
    target: str,
) -> WeightedPath:
    """Find confidence-weighted Dijkstra shortest path between two entities."""
    _require_non_empty(source, "source")
    _require_non_empty(target, "target")

    graph = plan.ontology.graph

    if source not in graph or target not in graph:
        return WeightedPath(path=[], edges=[], total_weight=0.0)

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
    except nx.NetworkXNoPath:
        return WeightedPath(path=[], edges=[], total_weight=0.0)
    edges: list[WeightedEdge] = [
        WeightedEdge(
            source=str(u),
            target=str(v),
            weight=float(weighted[u][v]["weight"]),
        )
        for u, v in pairwise(path)
    ]
    return WeightedPath(
        path=[str(n) for n in path],
        edges=edges,
        total_weight=float(total_weight),
    )


def get_hub_entities_tool(app_context: AppContext, limit: int = 20) -> list[HubEntity]:
    """Return top entities by PageRank."""
    store_paths = build_store_paths(app_context.config.paths.data_path)
    if not store_paths.sqlite_path.exists():
        return []
    with pooled_connection(store_paths.sqlite_path) as connection:
        profiles = load_top_entities_by_pagerank(connection, limit=limit)
    return [
        HubEntity(
            entity_id=p.entity_id,
            label=p.label,
            pagerank=p.pagerank,
            community_id=p.community_id,
        )
        for p in profiles
    ]


def find_bridge_entities_tool(app_context: AppContext, limit: int = 20) -> list[BridgeEntity]:
    """Return top entities by betweenness centrality."""
    store_paths = build_store_paths(app_context.config.paths.data_path)
    if not store_paths.sqlite_path.exists():
        return []
    with pooled_connection(store_paths.sqlite_path) as connection:
        profiles = load_top_entities_by_betweenness(connection, limit=limit)
    return [
        BridgeEntity(
            entity_id=p.entity_id,
            label=p.label,
            betweenness=p.betweenness,
            community_id=p.community_id,
        )
        for p in profiles
    ]


def find_foundational_entities_tool(
    app_context: AppContext, limit: int = 20
) -> list[FoundationalEntity]:
    """Return top entities by closeness centrality."""
    store_paths = build_store_paths(app_context.config.paths.data_path)
    if not store_paths.sqlite_path.exists():
        return []
    with pooled_connection(store_paths.sqlite_path) as connection:
        profiles = load_top_entities_by_closeness(connection, limit=limit)
    return [
        FoundationalEntity(
            entity_id=p.entity_id,
            label=p.label,
            closeness=p.closeness,
            community_id=p.community_id,
        )
        for p in profiles
    ]


def get_entity_graph_stats_tool(app_context: AppContext) -> EntityGraphStats:
    """Return knowledge graph statistics."""
    store_paths = build_store_paths(app_context.config.paths.data_path)
    if not store_paths.sqlite_path.exists():
        return EntityGraphStats(
            graph_version=0,
            last_build_at="never",
            entity_profiles=0,
            communities=0,
            community_reports=0,
            canonical_relations=0,
            relation_evidence=0,
            claims=0,
        )
    with pooled_connection(store_paths.sqlite_path) as connection:
        metadata = load_graph_metadata(connection)
        return EntityGraphStats(
            graph_version=int(metadata.get("graph_version", "0")),
            last_build_at=metadata.get("last_build_at", "never"),
            entity_profiles=count_entity_profiles(connection),
            communities=count_communities(connection),
            community_reports=count_community_reports(connection),
            canonical_relations=count_canonical_relations(connection),
            relation_evidence=count_relation_evidence(connection),
            claims=count_claims(connection),
        )


def search_knowledge_tool(
    app_context: AppContext,
    question: str,
    domain: str | None = None,
) -> KnowledgeAnswer:
    """Run the full answer pipeline with graph-augmented synthesis."""
    _require_non_empty(question, "question")

    envelope = answer_question(question=question, config=app_context.config, domain=domain)
    return KnowledgeAnswer(
        answer_status=envelope.answer_status.value,
        answer_text=envelope.answer_text,
        citations=envelope.citations,
        sentence_citations=[
            SentenceCitationView(text=sc.text, citation_labels=sc.citation_labels)
            for sc in envelope.sentence_citations
        ],
        warnings=envelope.warnings,
        metadata=KnowledgeAnswerMetadata.model_validate(envelope.metadata),
    )


def search_knowledge_deep_tool(
    app_context: AppContext,
    question: str,
    domain: str | None = None,
) -> KnowledgeAnswerDeep:
    """Run the full answer pipeline with graph context data returned alongside the answer."""
    _require_non_empty(question, "question")

    envelope = answer_question(question=question, config=app_context.config, domain=domain)

    matched_entity_ids = envelope.metadata.get("matched_entity_ids", [])
    graph_data = GraphContextData(level="none")

    if isinstance(matched_entity_ids, list) and matched_entity_ids:
        store_paths = build_store_paths(app_context.config.paths.data_path)
        if store_paths.sqlite_path.exists():
            with pooled_connection(store_paths.sqlite_path) as connection:
                context = build_graph_context(connection, matched_entity_ids, app_context.config)
            graph_data = GraphContextData(
                level=context.level,
                entity_profiles=[
                    GraphContextEntityProfile(
                        entity_id=p.entity_id,
                        label=p.label,
                        entity_type=p.entity_type,
                        deterministic_summary=p.deterministic_summary,
                        llm_summary=p.llm_summary,
                        pagerank=p.pagerank,
                        community_id=p.community_id,
                    )
                    for p in context.entity_profiles
                ],
                community_reports=[
                    GraphContextCommunityReport(
                        community_id=r.community_id,
                        member_count=r.member_count,
                        deterministic_summary=r.deterministic_summary,
                        llm_summary=r.llm_summary,
                    )
                    for r in context.community_reports
                ],
                claims=[
                    GraphContextClaim(
                        claim_text=c.claim_text,
                        claim_type=c.claim_type,
                        confidence=c.confidence,
                        subject_entity_id=c.subject_entity_id,
                        object_entity_id=c.object_entity_id,
                    )
                    for c in context.claims
                ],
            )

    return KnowledgeAnswerDeep(
        answer_status=envelope.answer_status.value,
        answer_text=envelope.answer_text,
        citations=envelope.citations,
        sentence_citations=[
            SentenceCitationView(text=sc.text, citation_labels=sc.citation_labels)
            for sc in envelope.sentence_citations
        ],
        warnings=envelope.warnings,
        metadata=KnowledgeAnswerMetadata.model_validate(envelope.metadata),
        graph_context=graph_data,
    )


def get_graph_overview_tool(app_context: AppContext) -> GraphOverview:
    """Return knowledge graph overview including stats and build state."""
    store_paths = build_store_paths(app_context.config.paths.data_path)
    if not store_paths.sqlite_path.exists():
        return GraphOverview(
            graph_version=0,
            last_build_at="never",
            community_algorithm=None,
            entity_profiles=0,
            communities=0,
            community_reports=0,
            canonical_relations=0,
            relation_evidence=0,
            claims=0,
        )
    with pooled_connection(store_paths.sqlite_path) as connection:
        metadata = load_graph_metadata(connection)
        return GraphOverview(
            graph_version=int(metadata.get("graph_version", "0")),
            last_build_at=metadata.get("last_build_at", "never"),
            community_algorithm=metadata.get(
                "community_algorithm",
                app_context.config.knowledge_graph.community_algorithm,
            ),
            entity_profiles=count_entity_profiles(connection),
            communities=count_communities(connection),
            community_reports=count_community_reports(connection),
            canonical_relations=count_canonical_relations(connection),
            relation_evidence=count_relation_evidence(connection),
            claims=count_claims(connection),
        )


def _require_non_empty(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must be non-empty.")
