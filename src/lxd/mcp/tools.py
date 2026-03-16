from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from lxd.app.bootstrap import AppContext, bootstrap_app
from lxd.ingest.pipeline import IngestPlan, build_ingest_plan
from lxd.ontology.graph import direct_neighbors
from lxd.retrieval.expansion import expand_entity_ids
from lxd.retrieval.query_pipeline import search_chunks
from lxd.stores.sqlite import (
    build_store_paths,
    connect_sqlite,
    find_chunks_by_entity_mentions,
    initialize_schema,
    load_ingest_config_snapshot,
    load_ontology_snapshot,
    store_has_committed_state,
    summarize_store,
)

_app_context: AppContext | None = None
_ingest_plan: IngestPlan | None = None


def initialize_tools(
    cwd: Path | None = None,
    *,
    profile: str | None = None,
    config_path: Path | None = None,
) -> None:
    global _app_context, _ingest_plan
    _app_context = bootstrap_app(cwd or Path.cwd(), profile=profile, config_path=config_path)
    _ingest_plan = None


def corpus_status_tool() -> dict[str, object]:
    context = _context()
    store_paths = build_store_paths(context.config.paths.data_path)
    if store_paths.sqlite_path.exists():
        connection = connect_sqlite(store_paths.sqlite_path)
        try:
            initialize_schema(connection)
            if store_has_committed_state(connection):
                ontology_snapshot = load_ontology_snapshot(connection)
                if _needs_live_ontology_fallback(ontology_snapshot):
                    plan = _plan()
                    summary = summarize_store(
                        connection,
                        ontology_file_count=len(plan.ontology.sources),
                        matcher_term_count=len(plan.ontology.matcher_records),
                        matcher_termset_hash=plan.ontology.matcher_termset_hash,
                        ontology_snapshot_hash=plan.ontology.snapshot_hash,
                        ontology_coverage_path_count=plan.ontology.coverage_report.discovered_path_count,
                        ontology_graph_relation_count=len(plan.ontology.relation_records),
                        ontology_validation_issue_count=len(plan.ontology.validation_issues),
                        ontology_validation_issue_samples=[
                            issue.message for issue in plan.ontology.validation_issues[:10]
                        ],
                        config_drift_warnings=_config_drift_warnings(connection, context),
                    )
                    ontology_snapshot = None
                else:
                    summary = summarize_store(
                        connection,
                        ontology_file_count=ontology_snapshot.source_file_count if ontology_snapshot else 0,
                        matcher_term_count=ontology_snapshot.matcher_term_count if ontology_snapshot else 0,
                        matcher_termset_hash=(
                            ontology_snapshot.matcher_termset_hash if ontology_snapshot else None
                        ),
                        ontology_snapshot_hash=(
                            ontology_snapshot.snapshot_hash if ontology_snapshot else None
                        ),
                        ontology_coverage_path_count=(
                            ontology_snapshot.coverage_path_count if ontology_snapshot else 0
                        ),
                        ontology_graph_relation_count=(
                            ontology_snapshot.graph_relation_count if ontology_snapshot else 0
                        ),
                        ontology_validation_issue_count=(
                            ontology_snapshot.validation_issue_count if ontology_snapshot else 0
                        ),
                        ontology_validation_issue_samples=(
                            json.loads(ontology_snapshot.validation_issues_json)
                            if ontology_snapshot is not None
                            else []
                        ),
                        config_drift_warnings=_config_drift_warnings(connection, context),
                    )
            else:
                summary = None
                ontology_snapshot = None
        finally:
            connection.close()
        if summary is not None:
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
                "entity_count": ontology_snapshot.entity_count if ontology_snapshot else 0,
                "matcher_term_count": summary.matcher_term_count,
                "ontology_snapshot_hash": summary.ontology_snapshot_hash,
                "matcher_termset_hash": summary.matcher_termset_hash,
                "ontology_coverage_path_count": summary.ontology_coverage_path_count,
                "ontology_graph_relation_count": summary.ontology_graph_relation_count,
                "ontology_validation_issue_count": summary.ontology_validation_issue_count,
                "ontology_validation_issue_samples": summary.ontology_validation_issue_samples,
                "config_drift_warnings": summary.config_drift_warnings,
            }
    plan = _plan()
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


def get_entity_types_tool() -> list[str]:
    plan = _plan()
    return sorted(entity["canonical_id"] for entity in plan.ontology.entity_definitions)


def get_related_concepts_tool(entity_id: str) -> list[dict[str, Any]]:
    plan = _plan()
    _validate_non_empty(entity_id, "entity_id")
    if entity_id not in plan.ontology.graph:
        return []
    return direct_neighbors(plan.ontology.graph, entity_id)


def search_corpus_tool(terms: str, domain: str | None, limit: int) -> list[dict[str, Any]]:
    context = _context()
    outcome = search_chunks(
        question=terms,
        config=context.config,
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
    entity_id: str,
    hops: int = 1,
    limit: int = 10,
) -> list[dict[str, Any]]:
    _validate_non_empty(entity_id, "entity_id")
    plan = _plan()
    if entity_id not in plan.ontology.graph:
        return []
    related_ids = expand_entity_ids(
        plan.ontology.graph,
        [entity_id],
        hops=hops,
        max_entities=50,
    )
    all_entity_ids = list({entity_id, *related_ids})
    context = _context()
    store_paths = build_store_paths(context.config.paths.data_path)
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


def _context() -> AppContext:
    if _app_context is None:
        initialize_tools(Path.cwd())
    if _app_context is None:
        raise RuntimeError("MCP tool context failed to initialize.")
    return _app_context


def _plan() -> IngestPlan:
    global _ingest_plan
    context = _context()
    if _ingest_plan is None:
        _ingest_plan = build_ingest_plan(context.config)
    return _ingest_plan


def _validate_non_empty(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must be non-empty.")


def _config_drift_warnings(connection: sqlite3.Connection, context: AppContext) -> list[str]:
    stored = load_ingest_config_snapshot(connection)
    if not stored:
        return []
    current = {
        "paths.corpus_path": str(context.config.paths.corpus_path),
        "paths.ontology_path": str(context.config.paths.ontology_path),
        "paths.data_path": str(context.config.paths.data_path),
        "chunking.chunk_overlap": str(context.config.chunking.chunk_overlap),
        "chunking.chunk_size": str(context.config.chunking.chunk_size),
        "chunking.min_tokens": str(context.config.chunking.min_tokens),
        "chunking.strategy": context.config.chunking.strategy,
        "chunking.tokenizer_backend": context.config.chunking.tokenizer_backend,
        "chunking.tokenizer_name": context.config.chunking.tokenizer_name,
        "models.embed": context.config.models.embed,
        "models.embed_dims": str(context.config.models.embed_dims),
    }
    warnings: list[str] = []
    for key, current_value in current.items():
        stored_value = stored.get(key)
        if stored_value is None:
            warnings.append(f"Committed ingest config is missing '{key}'.")
            continue
        if stored_value != current_value:
            warnings.append(f"Config drift: {key} stored={stored_value} current={current_value}.")
    return warnings


def _needs_live_ontology_fallback(ontology_snapshot: object) -> bool:
    if ontology_snapshot is None:
        return False
    return (
        getattr(ontology_snapshot, "source_file_count", 0) > 0
        and getattr(ontology_snapshot, "coverage_path_count", 0) == 0
        and getattr(ontology_snapshot, "graph_relation_count", 0) == 0
    )
