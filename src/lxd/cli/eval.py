"""Implement the CLI command for retrieval evaluation."""

from __future__ import annotations

from pathlib import Path

import typer

from lxd.app.bootstrap import bootstrap_app
from lxd.retrieval.eval import load_eval_cases, run_eval

PROFILE_OPTION = typer.Option(None, "--profile")
CONFIG_OPTION = typer.Option(None, "--config", dir_okay=False, resolve_path=True)


def eval_command(
    profile: str | None = PROFILE_OPTION,
    config: Path | None = CONFIG_OPTION,
) -> None:
    """Run retrieval evaluation against the configured corpus.

    Args:
        profile: Optional config profile name (`config.<profile>.yaml`).
        config: Optional explicit config file path.

    Raises:
        typer.BadParameter: If the evaluation set file is missing.

    Side Effects:
        Reads config and eval-set files, executes retrieval evaluation, and writes results to stdout.
    """
    context = bootstrap_app(Path.cwd(), profile=profile, config_path=config)
    eval_set = Path.cwd() / "tests" / "eval" / "eval_set.json"
    if not eval_set.exists():
        raise typer.BadParameter(f"Missing eval set: {eval_set}")
    cases = load_eval_cases(eval_set)
    summary = run_eval(cases, config=context.config)
    typer.echo(f"Config file: {context.config_path}")
    typer.echo(f"Eval questions: {summary.question_count}")
    typer.echo(f"Mean Recall@10: {summary.mean_recall_at_10:.3f}")
    typer.echo(f"Mean MRR@10: {summary.mean_mrr_at_10:.3f}")
    warning_count = sum(len(case.warnings) for case in summary.cases)
    if warning_count:
        typer.echo(f"Warnings: {warning_count}")
    failures = [
        case
        for case in summary.cases
        if case.recall_at_10 < 1.0 or case.mrr_at_10 == 0.0 or case.warnings
    ]
    if failures:
        typer.echo("Failing cases:")
        for case in failures:
            typer.echo(f"- {case.question}")
            typer.echo(f"  expected: {case.expected}")
            typer.echo(f"  ranked[:5]: {case.ranked[:5]}")
            if case.warnings:
                typer.echo(f"  warnings: {case.warnings}")
