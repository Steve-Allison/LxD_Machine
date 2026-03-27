"""Implement the CLI command for status reporting."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from lxd.app.bootstrap import bootstrap_app
from lxd.app.status import load_committed_status
from lxd.ingest.pipeline import build_ingest_plan
from lxd.stores.sqlite import (
    build_store_paths,
    connect_sqlite,
    initialize_schema,
)

PROFILE_OPTION = typer.Option(None, "--profile")
CONFIG_OPTION = typer.Option(None, "--config", dir_okay=False, resolve_path=True)


def status_command(
    profile: str | None = PROFILE_OPTION,
    config: Path | None = CONFIG_OPTION,
) -> None:
    """Print committed corpus status, snapshot status, or live plan status.

    Args:
        profile: Optional config profile name (`config.<profile>.yaml`).
        config: Optional explicit config file path.

    Side Effects:
        Reads SQLite and/or JSON snapshot state from disk and prints status lines to stdout.
    """
    context = bootstrap_app(Path.cwd(), profile=profile, config_path=config)
    store_paths = build_store_paths(context.config.paths.data_path)
    if store_paths.sqlite_path.exists():
        connection = connect_sqlite(store_paths.sqlite_path)
        try:
            initialize_schema(connection)
            status_snapshot = load_committed_status(
                connection,
                config=context.config,
                plan_provider=lambda: build_ingest_plan(context.config),
            )
        finally:
            connection.close()
        if status_snapshot is not None:
            summary = status_snapshot.summary
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
            typer.echo(f"Entity definitions: {status_snapshot.entity_count}")
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
