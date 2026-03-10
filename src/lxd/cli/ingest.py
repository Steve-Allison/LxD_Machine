from __future__ import annotations

from pathlib import Path

import typer

from lxd.app.bootstrap import bootstrap_app
from lxd.ingest.pipeline import run_ingest

PROFILE_OPTION = typer.Option(None, "--profile")
CONFIG_OPTION = typer.Option(None, "--config", dir_okay=False, resolve_path=True)


def ingest_command(
    full: bool = typer.Option(
        False, "--full", help="Force a fresh live rescan before writing the snapshot."
    ),
    profile: str | None = PROFILE_OPTION,
    config: Path | None = CONFIG_OPTION,
) -> None:
    context = bootstrap_app(Path.cwd(), profile=profile, config_path=config)
    result = run_ingest(context.config, full_rebuild=full)
    typer.echo(f"Ingest run: {result.run_id}")
    typer.echo(f"Config file: {context.config_path}")
    typer.echo(f"Corpus files tracked: {result.summary.corpus_file_count}")
    typer.echo(f"Text files tracked: {result.summary.text_file_count}")
    typer.echo(f"Asset files tracked: {result.summary.asset_file_count}")
    typer.echo(f"Chunks stored: {result.summary.chunk_count}")
    typer.echo(f"Mentions stored: {result.summary.mention_count}")
    typer.echo(f"Entity definitions: {result.entity_count}")
    typer.echo(f"Matcher terms: {result.summary.matcher_term_count}")
    typer.echo(f"Matcher termset hash: {result.summary.matcher_termset_hash}")
    typer.echo(f"Text sources re-embedded: {result.reembedded_text_sources}")
    typer.echo(f"Move-detected sources reused: {result.reused_move_sources}")
    typer.echo(f"Snapshot written: {result.snapshot_path}")
    if result.warnings:
        for warning in result.warnings:
            typer.echo(f"Warning: {warning}")
    if full:
        typer.echo("Full mode requested; snapshot reflects a full live rescan.")
