"""Expose the top-level CLI entrypoint."""

import typer

from lxd.cli.caption_assets import caption_assets_command
from lxd.cli.eval import eval_command
from lxd.cli.eval_gaps import eval_gaps_command
from lxd.cli.eval_quality import eval_quality_command
from lxd.cli.graph import (
    batch_status_command,
    build_graph_command,
    collect_batch_command,
    graph_status_command,
)
from lxd.cli.ingest import ingest_command
from lxd.cli.preflight import preflight_command
from lxd.cli.status import status_command

app = typer.Typer(no_args_is_help=True)
app.command("ingest")(ingest_command)
app.command("status")(status_command)
app.command("eval")(eval_command)
app.command("eval-gaps")(eval_gaps_command)
app.command("eval-quality")(eval_quality_command)
app.command("build-graph")(build_graph_command)
app.command("graph-status")(graph_status_command)
app.command("collect-batch")(collect_batch_command)
app.command("batch-status")(batch_status_command)
app.command("preflight")(preflight_command)
app.command("caption-assets")(caption_assets_command)


def main() -> None:
    """Execute the Typer application entrypoint."""
    app()


if __name__ == "__main__":
    main()
