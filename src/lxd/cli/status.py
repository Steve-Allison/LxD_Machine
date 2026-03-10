from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import typer

from lxd.app.bootstrap import bootstrap_app
from lxd.ingest.pipeline import build_ingest_plan
from lxd.settings.models import RuntimeConfig
from lxd.stores.sqlite import (
    build_store_paths,
    connect_sqlite,
    initialize_schema,
    load_ingest_config_snapshot,
    load_ontology_snapshot,
    store_has_committed_state,
    summarize_store,
)

PROFILE_OPTION = typer.Option(None, "--profile")
CONFIG_OPTION = typer.Option(None, "--config", dir_okay=False, resolve_path=True)


def status_command(
    profile: str | None = PROFILE_OPTION,
    config: Path | None = CONFIG_OPTION,
) -> None:
    context = bootstrap_app(Path.cwd(), profile=profile, config_path=config)
    store_paths = build_store_paths(context.config.paths.data_path)
    if store_paths.sqlite_path.exists():
        connection = connect_sqlite(store_paths.sqlite_path)
        try:
            initialize_schema(connection)
            if store_has_committed_state(connection):
                ontology_snapshot = load_ontology_snapshot(connection)
                if _needs_live_ontology_fallback(ontology_snapshot):
                    plan = build_ingest_plan(context.config)
                    ontology_snapshot = None
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
                        config_drift_warnings=[],
                    )
                    config_drift_warnings = _config_drift_warnings(connection, context.config)
                    summary = summary.__class__(
                        corpus_file_count=summary.corpus_file_count,
                        text_file_count=summary.text_file_count,
                        asset_file_count=summary.asset_file_count,
                        retrieval_role_counts=summary.retrieval_role_counts,
                        chunk_count=summary.chunk_count,
                        mention_count=summary.mention_count,
                        ontology_file_count=summary.ontology_file_count,
                        matcher_term_count=summary.matcher_term_count,
                        matcher_termset_hash=summary.matcher_termset_hash,
                        ontology_snapshot_hash=summary.ontology_snapshot_hash,
                        ontology_coverage_path_count=summary.ontology_coverage_path_count,
                        ontology_graph_relation_count=summary.ontology_graph_relation_count,
                        ontology_validation_issue_count=summary.ontology_validation_issue_count,
                        ontology_validation_issue_samples=summary.ontology_validation_issue_samples,
                        config_drift_warnings=config_drift_warnings,
                    )
                else:
                    config_drift_warnings = _config_drift_warnings(connection, context.config)
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
                        config_drift_warnings=config_drift_warnings,
                    )
            else:
                summary = None
        finally:
            connection.close()
        if summary is not None:
            typer.echo(f"Config file: {context.config_path}")
            typer.echo(f"SQLite store: {store_paths.sqlite_path}")
            typer.echo(f"LanceDB store: {store_paths.lancedb_path}")
            typer.echo(f"Corpus files tracked: {summary.corpus_file_count}")
            typer.echo(f"Text files tracked: {summary.text_file_count}")
            typer.echo(f"Asset files tracked: {summary.asset_file_count}")
            typer.echo(f"Retrieval searchable: {summary.retrieval_role_counts['searchable']}")
            typer.echo(f"Retrieval asset_only: {summary.retrieval_role_counts['asset_only']}")
            typer.echo(f"Retrieval not_searchable: {summary.retrieval_role_counts['not_searchable']}")
            typer.echo(f"Chunks stored: {summary.chunk_count}")
            typer.echo(f"Mentions stored: {summary.mention_count}")
            typer.echo(f"Ontology snapshot hash: {summary.ontology_snapshot_hash}")
            typer.echo(f"Matcher termset hash: {summary.matcher_termset_hash}")
            typer.echo(f"Ontology coverage paths: {summary.ontology_coverage_path_count}")
            typer.echo(f"Ontology graph relations: {summary.ontology_graph_relation_count}")
            typer.echo(
                f"Ontology validation issues: {summary.ontology_validation_issue_count}"
            )
            for sample in summary.ontology_validation_issue_samples:
                typer.echo(f"Ontology issue: {sample}")
            for warning in summary.config_drift_warnings:
                typer.echo(f"Warning: {warning}")
            return
    plan = build_ingest_plan(context.config)
    snapshot_path = context.config.paths.data_path / "ingest_snapshot.json"
    if snapshot_path.exists():
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
        typer.echo(f"Config file: {context.config_path}")
        typer.echo(f"Snapshot path: {snapshot_path}")
        typer.echo(f"Corpus total: {payload['corpus_counts']['total']}")
        typer.echo(f"Corpus text: {payload['corpus_counts']['text']}")
        typer.echo(f"Corpus assets: {payload['corpus_counts']['asset']}")
        typer.echo(f"Chunks stored: {payload['chunk_count']}")
        typer.echo(f"Mentions stored: {payload['mention_count']}")
        typer.echo(f"Ontology snapshot hash: {payload['ontology_snapshot_hash']}")
        typer.echo(f"Matcher termset hash: {payload['matcher_termset_hash']}")
        typer.echo(f"Ontology coverage paths: {payload.get('ontology_coverage_path_count', 0)}")
        typer.echo(f"Ontology graph relations: {payload.get('ontology_graph_relation_count', 0)}")
        typer.echo(f"Ontology validation issues: {payload.get('ontology_validation_issue_count', 0)}")
        for sample in payload.get("ontology_validation_issue_samples", []):
            typer.echo(f"Ontology issue: {sample}")
        return
    asset_count = sum(1 for item in plan.scanned_files if item.source_type == "image_png")
    typer.echo(f"Config file: {context.config_path}")
    typer.echo(f"Corpus path: {context.config.paths.corpus_path}")
    typer.echo(f"Ontology path: {context.config.paths.ontology_path}")
    typer.echo(f"Scanned text files: {len(plan.scanned_files) - asset_count}")
    typer.echo(f"Scanned asset files: {asset_count}")
    typer.echo(f"Ontology snapshot hash: {plan.ontology.snapshot_hash}")
    typer.echo(f"Matcher termset hash: {plan.ontology.matcher_termset_hash}")
    typer.echo(
        f"Ontology validation issues: {len(plan.ontology.validation_issues)}"
    )
    for issue in plan.ontology.validation_issues[:10]:
        typer.echo(f"Ontology issue: {issue.message}")


def _config_drift_warnings(
    connection: sqlite3.Connection, config: RuntimeConfig
) -> list[str]:
    stored = load_ingest_config_snapshot(connection)
    if not stored:
        return []
    current = {
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
        "models.embed_dims": str(config.models.embed_dims),
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
