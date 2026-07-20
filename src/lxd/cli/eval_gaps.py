"""CLI command that turns retrieval-eval failures into gap tickets.

Runs the same retrieval evaluation as `lxd.cli.eval`, persists the run to
`retrieval_eval_runs.jsonl`, then writes one JSON gap ticket per failing
case under `<data_path>/gaps/`. Never edits the wiki or ontology — tickets
are for a human reviewer to triage.
"""

from pathlib import Path
from typing import Final

import typer

from lxd.app.bootstrap import bootstrap_app
from lxd.domain.time import utc_now
from lxd.eval.gaps import build_gap_tickets, write_gap_tickets
from lxd.retrieval.eval import append_eval_run, build_eval_run, load_eval_cases, run_eval

PROFILE_OPTION: Final = typer.Option(None, "--profile")
CONFIG_OPTION: Final = typer.Option(None, "--config", dir_okay=False, resolve_path=True)
PERSIST_OPTION: Final = typer.Option(
    True,
    "--persist/--no-persist",
    help="Append this run to <data_path>/retrieval_eval_runs.jsonl (default: on).",
)
TAG_OPTION: Final = typer.Option(
    "",
    "--tag",
    help="Optional label stored alongside this run in the history file.",
)


def eval_gaps_command(
    profile: str | None = PROFILE_OPTION,
    config: Path | None = CONFIG_OPTION,
    persist: bool = PERSIST_OPTION,
    tag: str = TAG_OPTION,
) -> None:
    """Run retrieval evaluation and write gap tickets for every failing case.

    Args:
        profile: Optional config profile name (`config.<profile>.yaml`).
        config: Optional explicit config file path.
        persist: Whether to append this run to the JSONL history file.
        tag: Optional caller-supplied label stored alongside the run.

    Raises:
        typer.BadParameter: If the evaluation set file is missing.

    Side Effects:
        Reads config and eval-set files, executes retrieval evaluation,
        (when `persist` is true) appends a run record to
        `<data_path>/retrieval_eval_runs.jsonl`, and writes gap tickets to
        `<data_path>/gaps/`.
    """
    context = bootstrap_app(Path.cwd(), profile=profile, config_path=config)
    eval_set = Path.cwd() / "tests" / "eval" / "eval_set.json"
    if not eval_set.exists():
        raise typer.BadParameter(f"Missing eval set: {eval_set}")
    cases = load_eval_cases(eval_set)

    run_started_at = utc_now()
    summary = run_eval(cases, config=context.config)
    run_finished_at = utc_now()

    typer.echo(f"Config file: {context.config_path}")
    typer.echo(f"Eval questions: {summary.question_count}")
    typer.echo(f"Mean Recall@10: {summary.mean_recall_at_10:.3f}")
    typer.echo(f"Mean MRR@10: {summary.mean_mrr_at_10:.3f}")

    if persist:
        run = build_eval_run(
            summary,
            run_started_at=run_started_at,
            run_finished_at=run_finished_at,
            run_tag=tag,
        )
        history_path = context.config.paths.data_path / "retrieval_eval_runs.jsonl"
        append_eval_run(run, history_path)
        typer.echo(f"History appended: {history_path}")

    gaps_dir = context.config.paths.data_path / "gaps"
    tickets = build_gap_tickets(summary)
    written = write_gap_tickets(tickets, gaps_dir)

    typer.echo(f"Gap tickets: {len(tickets)} derived, {len(written)} written (upserted)")
    typer.echo(f"Gap tickets directory: {gaps_dir}")
