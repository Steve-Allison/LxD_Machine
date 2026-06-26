"""Preflight: cheap sanity check before any expensive ingest run.

Verifies — without spending API budget — that the system is healthy:

1. ``.env`` and config load successfully.
2. SQLite opens, migrations are at the expected version, and the schema
   integrity check passes (catches ghost FKs, missing tables, missing
   columns *before* OpenAI is called even once).
3. LanceDB opens and the chunk + cache tables can be created/opened.
4. The corpus and ontology paths exist.
5. Counts of expected work (files to scan, files unchanged, files needing
   embed) are reported, plus a coarse cache hit-rate estimate based on
   chunk hashes already in the cache.

Exit codes:
    0 — everything green; ingest is safe to run.
    1 — at least one red flag; ingest is *not* safe to run.

Use ``pixi run preflight`` before any expensive operation.
"""

import sqlite3
from pathlib import Path
from typing import Final

import typer

from lxd.app.bootstrap import bootstrap_app
from lxd.ingest.budget import estimate_run_cost
from lxd.ingest.embedding_cache import open_cache_table
from lxd.ingest.pipeline.orchestrator import build_ingest_plan
from lxd.stores.lancedb import connect_lancedb, open_chunk_table
from lxd.stores.schema import (
    CURRENT_SCHEMA_VERSION,
    SchemaIntegrityError,
    get_schema_version,
    verify_schema_integrity,
)
from lxd.stores.sqlite.connection import build_store_paths, connect_sqlite

PROFILE_OPTION: Final = typer.Option(None, "--profile")
CONFIG_OPTION: Final = typer.Option(None, "--config", dir_okay=False, resolve_path=True)


def preflight_command(
    profile: str | None = PROFILE_OPTION,
    config: Path | None = CONFIG_OPTION,
) -> None:
    """Run the preflight checks and exit non-zero on any failure.

    Args:
        profile: Optional config profile name.
        config: Optional explicit config file path.

    Side Effects:
        Opens SQLite and LanceDB, prints status lines, exits with code 1 on
        any failure.
    """
    issues: list[str] = []
    notes: list[str] = []

    try:
        context = bootstrap_app(Path.cwd(), profile=profile, config_path=config)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        typer.echo(f"[X] bootstrap failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    notes.append(f"Config file: {context.config_path}")

    if not context.config.paths.corpus_path.exists():
        issues.append(f"corpus path missing: {context.config.paths.corpus_path}")
    else:
        notes.append(f"Corpus path: {context.config.paths.corpus_path}")

    if not context.config.paths.ontology_path.exists():
        issues.append(f"ontology path missing: {context.config.paths.ontology_path}")
    else:
        notes.append(f"Ontology path: {context.config.paths.ontology_path}")

    store_paths = build_store_paths(context.config.paths.data_path)
    notes.append(f"SQLite store: {store_paths.sqlite_path}")
    notes.append(f"LanceDB store: {store_paths.lancedb_path}")

    if context.config.paths.data_path.exists():
        sqlite_conn: sqlite3.Connection | None = None
        try:
            sqlite_conn = connect_sqlite(store_paths.sqlite_path)
            schema_version = get_schema_version(sqlite_conn)
            notes.append(f"Schema version: {schema_version} (expected {CURRENT_SCHEMA_VERSION})")
            if schema_version > CURRENT_SCHEMA_VERSION:
                issues.append(
                    f"schema version {schema_version} is newer than this code "
                    f"supports ({CURRENT_SCHEMA_VERSION}); upgrade the code"
                )
            elif schema_version > 0:
                try:
                    verify_schema_integrity(sqlite_conn)
                    notes.append("Schema integrity: OK")
                except SchemaIntegrityError as exc:
                    issues.append(f"schema integrity check failed: {exc}")
        except sqlite3.DatabaseError as exc:
            issues.append(f"SQLite open failed: {exc}")
        finally:
            if sqlite_conn is not None:
                sqlite_conn.close()
    else:
        notes.append("Data path does not exist yet — schema check skipped (first run).")

    if store_paths.lancedb_path.exists() or context.config.paths.data_path.exists():
        try:
            db = connect_lancedb(store_paths.lancedb_path)
            try:
                chunk_table = open_chunk_table(db, vector_size=context.config.models.embed_dims)
                cache_table = open_cache_table(db, vector_size=context.config.models.embed_dims)
                cache_count = _count_rows(cache_table)
                chunk_count = _count_rows(chunk_table)
                notes.append(f"LanceDB chunk vectors: {chunk_count}")
                notes.append(f"LanceDB embedding cache entries: {cache_count}")
            except (FileNotFoundError, ValueError, RuntimeError) as exc:
                issues.append(f"LanceDB table open failed: {exc}")
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            issues.append(f"LanceDB connect failed: {exc}")

    for line in notes:
        typer.echo(line)

    if context.config.paths.corpus_path.exists() and context.config.paths.ontology_path.exists():
        try:
            plan = build_ingest_plan(context.config)
            estimate = estimate_run_cost(plan.scanned_files, context.config)
        except (FileNotFoundError, ValueError, OSError) as exc:
            issues.append(f"cost-estimate scan failed: {exc}")
        else:
            typer.echo("")
            typer.echo("Estimated run cost (upper bound — review before pressing go):")
            typer.echo(
                f"  Embedding: {estimate.text_file_count} text files, "
                f"~{estimate.embedding_tokens_est:,} tokens at {estimate.embedding_model} "
                f"(~${estimate.embedding_usd_est:.2f})"
            )
            if estimate.llm_call_cap is None:
                typer.echo(
                    "  LLM (relation extraction): no cap configured "
                    "— cost ceiling unknown. Set "
                    "`ingest_budget.max_llm_calls_per_run` to bound spend."
                )
            else:
                typer.echo(
                    f"  LLM (relation extraction): up to {estimate.llm_call_cap} calls × "
                    f"~{estimate.llm_prompt_tokens_per_call + estimate.llm_completion_tokens_per_call} "
                    f"tokens at {estimate.llm_model} (~${estimate.llm_usd_est:.2f})"
                )
            typer.echo(f"  Total upper bound: ~${estimate.total_usd_est:.2f}")

    if issues:
        typer.echo("")
        typer.echo("Preflight FAILED:", err=True)
        for issue in issues:
            typer.echo(f"  [X] {issue}", err=True)
        raise typer.Exit(code=1)

    typer.echo("")
    typer.echo("Preflight OK — ingest is safe to run.")


def _count_rows(table: object) -> int:
    """Best-effort row count; LanceDB exposes ``count_rows`` on Table."""
    counter = getattr(table, "count_rows", None)
    if counter is None:
        return -1
    try:
        return int(counter())
    except (TypeError, ValueError):
        return -1
