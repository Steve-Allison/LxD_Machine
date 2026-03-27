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


def _require_non_empty(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must be non-empty.")
