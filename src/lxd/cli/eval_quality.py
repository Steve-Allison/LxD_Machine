"""CLI command for RAGAS-style quality eval.

Distinct from ``lxd.cli.eval`` (which runs recall/MRR retrieval eval):
this one runs the full answer pipeline and judges synthesis quality via
LLM-graded faithfulness, answer relevance, and context precision.
"""

import asyncio
from pathlib import Path
from typing import Final

import typer

from lxd.app.bootstrap import bootstrap_app
from lxd.eval import (
    append_run_to_history,
    format_console_report,
    load_golden_set,
    run_quality_eval,
)
from lxd.eval.report import write_report_json

PROFILE_OPTION: Final = typer.Option(None, "--profile")
CONFIG_OPTION: Final = typer.Option(None, "--config", dir_okay=False, resolve_path=True)
GOLDEN_OPTION: Final = typer.Option(
    None,
    "--golden",
    dir_okay=False,
    resolve_path=True,
    help="Path to the golden quality set (defaults to tests/eval/golden_quality_set.json).",
)
JUDGE_MODEL_OPTION: Final = typer.Option(
    "gpt-4o-mini",
    "--judge-model",
    help="OpenAI chat model used as the eval judge.",
)
OUT_OPTION: Final = typer.Option(
    None,
    "--out",
    dir_okay=False,
    resolve_path=True,
    help="Path to write the full JSON report (in addition to the appended history file).",
)


def eval_quality_command(
    profile: str | None = PROFILE_OPTION,
    config: Path | None = CONFIG_OPTION,
    golden: Path | None = GOLDEN_OPTION,
    judge_model: str = JUDGE_MODEL_OPTION,
    out: Path | None = OUT_OPTION,
) -> None:
    """Run RAGAS-style quality eval on the answer pipeline.

    Runs every question in the golden set against ``answer_question``, then
    computes faithfulness, answer relevance, and context precision via the
    judge LLM. Results are appended to
    ``<data_path>/eval_quality_runs.jsonl`` so historical scores can be
    diffed between runs.

    Args:
        profile: Optional config profile name.
        config: Optional explicit config file path.
        golden: Override path to the golden set JSON. Defaults to
            ``tests/eval/golden_quality_set.json``.
        judge_model: OpenAI chat model used as the eval judge.
        out: Optional path to write the full JSON report (in addition to
            the history file).

    Raises:
        typer.BadParameter: If the golden set file is missing.
    """
    context = bootstrap_app(Path.cwd(), profile=profile, config_path=config)

    golden_path = golden or Path.cwd() / "tests" / "eval" / "golden_quality_set.json"
    if not golden_path.exists():
        raise typer.BadParameter(f"Missing golden set: {golden_path}")

    golden_set = load_golden_set(golden_path)
    if not golden_set:
        raise typer.BadParameter(f"Golden set at {golden_path} is empty.")

    typer.echo(f"Config file: {context.config_path}")
    typer.echo(f"Golden set:  {golden_path}  ({len(golden_set)} questions)")
    typer.echo(f"Judge model: {judge_model}")
    typer.echo("Running eval — this calls the LLM judge once per claim / question / context.")
    typer.echo("")

    report = asyncio.run(
        run_quality_eval(
            golden_set=golden_set,
            config=context.config,
            judge_model=judge_model,
        )
    )

    history_path = context.config.paths.data_path / "eval_quality_runs.jsonl"
    append_run_to_history(report, history_path)
    if out is not None:
        write_report_json(report, out)

    typer.echo(format_console_report(report))
    typer.echo("")
    typer.echo(f"History appended: {history_path}")
    if out is not None:
        typer.echo(f"Full report:      {out}")
