"""Backfill captions for previously ingested ``asset_only`` PNG sources.

Turns PNG assets that were registered before ``multimodal.captions_enabled``
was turned on (or whose caption attempt failed at ingest time) into
searchable chunks, without re-running a full corpus ingest.

Preflight-as-gate (project convention, see ``.claude/rules/ingest-discipline.md``):
this command prints the plan — candidate count, model, one API call per
image — and stops before spending anything unless ``--yes`` is passed.
"""

import contextlib
import sqlite3
from pathlib import Path
from typing import Final

import structlog
import typer
from rich.console import Console
from rich.table import Table

from lxd.app.bootstrap import bootstrap_app
from lxd.domain.ids import blake3_hex
from lxd.domain.status import LifecycleStatus, RetrievalStatus
from lxd.ingest.captions import caption_asset_source
from lxd.stores.lancedb import (
    connect_lancedb,
    load_source_chunk_rows,
    open_chunk_table,
    refresh_fts_index,
    restore_source_chunk_rows,
)
from lxd.stores.lancedb import replace_source_chunks as replace_vector_source_chunks
from lxd.stores.models import ManifestRecord
from lxd.stores.sqlite.chunks import replace_source_chunks as replace_sqlite_source_chunks
from lxd.stores.sqlite.connection import build_store_paths, connect_sqlite, initialize_schema
from lxd.stores.sqlite.manifest import load_manifest_index, upsert_manifest_record

_log = structlog.get_logger(__name__)
_console = Console()

_PROFILE_OPTION: Final = typer.Option(None, "--profile", help="Config profile name")
_CONFIG_OPTION: Final = typer.Option(None, "--config", help="Config file path")
_LIMIT_OPTION: Final = typer.Option(
    None, "--limit", min=1, help="Cap the number of images captioned this run."
)
_YES_OPTION: Final = typer.Option(
    False,
    "--yes",
    help="Proceed without an interactive confirmation prompt.",
)

# Runs above this size always require an explicit --yes; below it, an
# interactive confirm() is enough. Mirrors build_graph_command's --full gate.
_AUTO_CONFIRM_THRESHOLD: Final = 20


def caption_assets_command(
    limit: int | None = _LIMIT_OPTION,
    yes: bool = _YES_OPTION,
    profile: str | None = _PROFILE_OPTION,
    config_path: Path | None = _CONFIG_OPTION,
) -> None:
    """Caption asset_only PNG sources already tracked in the manifest.

    Args:
        limit: Cap the number of images captioned this run (safety valve
            for large corpora / cost control).
        yes: Skip the interactive confirmation. Required (not optional)
            once the candidate count exceeds ``_AUTO_CONFIRM_THRESHOLD``.
        profile: Optional config profile name (`config.<profile>.yaml`).
        config_path: Optional explicit config file path.
    """
    ctx = bootstrap_app(Path.cwd(), profile=profile, config_path=config_path)
    config = ctx.config

    if not config.multimodal.captions_enabled:
        _console.print(
            "[red]multimodal.captions_enabled is false in config.yaml.[/red] "
            "Set it to true before backfilling captions — the code path is "
            "complete but stays inert until you opt in."
        )
        raise typer.Exit(code=1)

    store_paths = build_store_paths(config.paths.data_path)
    connection = connect_sqlite(store_paths.sqlite_path)
    initialize_schema(connection)

    manifest = load_manifest_index(connection)
    candidates = sorted(
        (
            record
            for record in manifest.values()
            if record.source_type == "image_png"
            and record.retrieval_status == RetrievalStatus.ASSET_ONLY
            and record.lifecycle_status != LifecycleStatus.DELETED
        ),
        key=lambda record: record.source_rel_path,
    )
    total_candidates = len(candidates)
    if limit is not None:
        candidates = candidates[:limit]

    plan_table = Table(title="Caption backfill plan")
    plan_table.add_column("Metric", style="bold")
    plan_table.add_column("Value")
    plan_table.add_row("asset_only PNGs found", str(total_candidates))
    plan_table.add_row("Queued this run (--limit)", str(len(candidates)))
    plan_table.add_row("Caption model", config.multimodal.caption_model)
    plan_table.add_row("Estimated API calls", str(len(candidates)))
    _console.print(plan_table)

    if not candidates:
        _console.print("Nothing to caption.")
        connection.close()
        return

    if not yes:
        if len(candidates) > _AUTO_CONFIRM_THRESHOLD:
            _console.print(
                f"[bold red]{len(candidates)} images queued — this exceeds the "
                f"auto-confirm threshold ({_AUTO_CONFIRM_THRESHOLD}). "
                "Pass --yes to proceed.[/bold red]"
            )
            connection.close()
            raise typer.Exit(code=1)
        if not typer.confirm(f"Caption {len(candidates)} image(s) now?"):
            connection.close()
            raise typer.Abort()

    vector_db = connect_lancedb(store_paths.lancedb_path)
    vector_table = open_chunk_table(vector_db, vector_size=config.models.embed_dims)

    captioned = 0
    empty_or_failed = 0
    missing_files = 0

    try:
        for record in candidates:
            absolute_path = Path(record.absolute_path)
            if not absolute_path.exists():
                _console.print(
                    f"[yellow]Missing file, skipping:[/yellow] {record.source_rel_path}"
                )
                missing_files += 1
                continue

            document_id = record.document_id or blake3_hex(
                record.source_rel_path, record.content_hash
            )
            caption_chunk = caption_asset_source(
                absolute_path=absolute_path,
                source_rel_path=record.source_rel_path,
                source_type=record.source_type,
                source_domain=record.source_domain,
                content_hash=record.content_hash,
                document_id=document_id,
                config=config,
            )
            if caption_chunk is None:
                empty_or_failed += 1
                continue

            prior_vectors = load_source_chunk_rows(vector_table, record.source_rel_path)
            replace_vector_source_chunks(vector_table, record.source_rel_path, [caption_chunk])
            try:
                replace_sqlite_source_chunks(
                    connection,
                    source_rel_path=record.source_rel_path,
                    chunk_records=[caption_chunk],
                    mention_records=[],
                    relation_records=[],
                )
            except sqlite3.Error as exc:
                with contextlib.suppress(FileNotFoundError, ValueError, RuntimeError):
                    restore_source_chunk_rows(vector_table, record.source_rel_path, prior_vectors)
                _console.print(
                    f"[red]Persistence failed for {record.source_rel_path}:[/red] {exc}"
                )
                empty_or_failed += 1
                continue

            committed = ManifestRecord(
                source_rel_path=record.source_rel_path,
                absolute_path=record.absolute_path,
                source_type=record.source_type,
                source_domain=record.source_domain,
                document_id=document_id,
                file_size_bytes=record.file_size_bytes,
                content_hash=record.content_hash,
                parent_source_rel_path=record.parent_source_rel_path,
                chunk_count=1,
                last_seen_at=record.last_seen_at,
                last_processed_at=record.last_processed_at,
                last_committed_at=record.last_committed_at,
                error_message=None,
                lifecycle_status=LifecycleStatus.COMPLETE,
                retrieval_status=RetrievalStatus.SEARCHABLE,
            )
            upsert_manifest_record(connection, committed)
            captioned += 1
            _console.print(f"[green]Captioned:[/green] {record.source_rel_path}")

        if captioned:
            refresh_fts_index(vector_table)
    finally:
        connection.close()

    result_table = Table(title="Caption backfill result")
    result_table.add_column("Outcome", style="bold")
    result_table.add_column("Count")
    result_table.add_row("Captioned (now searchable)", str(captioned))
    result_table.add_row("Empty caption / generation failed", str(empty_or_failed))
    result_table.add_row("Missing files", str(missing_files))
    result_table.add_row("Remaining asset_only (this run)", str(total_candidates - captioned))
    _console.print(result_table)
