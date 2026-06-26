"""Implement the CLI command for corpus ingestion.

After a successful corpus pass, ``ingest_command`` auto-chains into
``build_graph_command`` by default so a single ``pixi run ingest`` gets
the corpus query-ready end-to-end (claims, communities, entity
profiles, community reports). Power users can opt out with ``--no-graph``
for corpus-only ingest. The destructive-rebuild confirmation prompt
inside ``build_graph_command`` still fires when ``--full`` is set —
the auto-chain inherits, not replaces, that gate.
"""

from pathlib import Path
from typing import Final

import typer

from lxd.app.bootstrap import bootstrap_app
from lxd.cli.graph import build_graph_command
from lxd.ingest.pipeline.orchestrator import run_ingest

PROFILE_OPTION: Final = typer.Option(None, "--profile")
CONFIG_OPTION: Final = typer.Option(None, "--config", dir_okay=False, resolve_path=True)


def ingest_command(
    full: bool = typer.Option(
        False, "--full", help="Force a fresh live rescan before writing the snapshot."
    ),
    with_graph: bool = typer.Option(
        True,
        "--with-graph/--no-graph",
        help=(
            "When True (default), auto-build the knowledge graph after the "
            "corpus pass completes. Use --no-graph for corpus-only ingest "
            "(claims / communities / profiles / reports are NOT built)."
        ),
    ),
    profile: str | None = PROFILE_OPTION,
    config: Path | None = CONFIG_OPTION,
) -> None:
    """Run corpus ingestion (auto-chains to build-graph by default).

    Args:
        full: When `True`, force a full rescan AND a full graph rebuild
            (the destructive prompt inside build_graph_command still fires).
        with_graph: Auto-chain to build_graph_command after the corpus
            pass succeeds. Default True. Pass ``--no-graph`` for corpus-only.
        profile: Optional config profile name (`config.<profile>.yaml`).
        config: Optional explicit config file path.

    Side Effects:
        Executes the corpus ingestion pipeline, writes store state, prints
        summary lines to stdout, and (when with_graph=True) runs the full
        knowledge-graph build phases.
    """
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

    if not with_graph:
        typer.echo(
            "--no-graph: skipping knowledge-graph build. "
            "Run `pixi run build-graph` separately to populate the KG."
        )
        return

    typer.echo("")
    typer.echo("== Auto-chaining to knowledge-graph build ==")
    build_graph_command(
        full=full,
        enrich=False,
        dry_run=False,
        batch=False,
        phase=None,
        profile=profile,
        config_path=config,
    )
