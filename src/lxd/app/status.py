from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass

from lxd.ingest.pipeline import IngestPlan
from lxd.settings.models import RuntimeConfig
from lxd.stores.models import CorpusStatusSummary, OntologySnapshotRecord
from lxd.stores.sqlite import (
    load_ingest_config_snapshot,
    load_ontology_snapshot,
    store_has_committed_state,
    summarize_store,
)


@dataclass(frozen=True)
class StatusSnapshot:
    summary: CorpusStatusSummary
    entity_count: int


def current_ingest_config(config: RuntimeConfig) -> dict[str, str]:
    return {
        "paths.corpus_path": str(config.paths.corpus_path),
        "paths.ontology_path": str(config.paths.ontology_path),
        "paths.data_path": str(config.paths.data_path),
        "chunking.chunk_overlap": str(config.chunking.chunk_overlap),
        "chunking.chunk_size": str(config.chunking.chunk_size),
        "chunking.min_tokens": str(config.chunking.min_tokens),
        "chunking.strategy": config.chunking.strategy,
        "chunking.tokenizer_backend": config.chunking.tokenizer_backend,
        "chunking.tokenizer_name": config.chunking.tokenizer_name,
        "models.embed": config.models.embed,
        "models.embed_backend": config.models.embed_backend,
        "models.embed_dims": str(config.models.embed_dims),
    }


def config_drift_warnings(connection: sqlite3.Connection, config: RuntimeConfig) -> list[str]:
    stored = load_ingest_config_snapshot(connection)
    if not stored:
        return []
    warnings: list[str] = []
    for key, current_value in current_ingest_config(config).items():
        stored_value = stored.get(key)
        if stored_value is None:
            warnings.append(f"Committed ingest config is missing '{key}'.")
            continue
        if stored_value != current_value:
            warnings.append(f"Config drift: {key} stored={stored_value} current={current_value}.")
    return warnings


def needs_live_ontology_fallback(ontology_snapshot: OntologySnapshotRecord | None) -> bool:
    return (
        ontology_snapshot is not None
        and ontology_snapshot.source_file_count > 0
        and ontology_snapshot.coverage_path_count == 0
        and ontology_snapshot.graph_relation_count == 0
    )


def load_committed_status(
    connection: sqlite3.Connection,
    *,
    config: RuntimeConfig,
    plan_provider: Callable[[], IngestPlan],
) -> StatusSnapshot | None:
    if not store_has_committed_state(connection):
        return None

    drift_warnings = config_drift_warnings(connection, config)
    ontology_snapshot = load_ontology_snapshot(connection)
    if needs_live_ontology_fallback(ontology_snapshot):
        plan = plan_provider()
        return StatusSnapshot(
            summary=summarize_store(
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
                config_drift_warnings=drift_warnings,
            ),
            entity_count=len(plan.ontology.entity_definitions),
        )

    validation_issue_samples = (
        _parse_validation_issues_json(ontology_snapshot.validation_issues_json)
        if ontology_snapshot is not None
        else []
    )
    return StatusSnapshot(
        summary=summarize_store(
            connection,
            ontology_file_count=ontology_snapshot.source_file_count if ontology_snapshot else 0,
            matcher_term_count=ontology_snapshot.matcher_term_count if ontology_snapshot else 0,
            matcher_termset_hash=ontology_snapshot.matcher_termset_hash if ontology_snapshot else None,
            ontology_snapshot_hash=ontology_snapshot.snapshot_hash if ontology_snapshot else None,
            ontology_coverage_path_count=ontology_snapshot.coverage_path_count if ontology_snapshot else 0,
            ontology_graph_relation_count=ontology_snapshot.graph_relation_count if ontology_snapshot else 0,
            ontology_validation_issue_count=ontology_snapshot.validation_issue_count if ontology_snapshot else 0,
            ontology_validation_issue_samples=validation_issue_samples,
            config_drift_warnings=drift_warnings,
        ),
        entity_count=ontology_snapshot.entity_count if ontology_snapshot is not None else 0,
    )


def _parse_validation_issues_json(payload: str) -> list[str]:
    try:
        decoded = json.loads(payload)
    except ValueError:
        return []
    if not isinstance(decoded, list):
        return []
    return [str(item) for item in decoded]
