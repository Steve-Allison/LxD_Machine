"""Expose the top-level CLI entrypoint."""

from __future__ import annotations

import typer

from lxd.cli.eval import eval_command
from lxd.cli.graph import (
    batch_status_command,
    build_graph_command,
    collect_batch_command,
    graph_status_command,
)
from lxd.cli.ingest import ingest_command
from lxd.cli.status import status_command

app = typer.Typer(no_args_is_help=True)
app.command("ingest")(ingest_command)
app.command("status")(status_command)
app.command("eval")(eval_command)
app.command("build-graph")(build_graph_command)
app.command("graph-status")(graph_status_command)
app.command("collect-batch")(collect_batch_command)
app.command("batch-status")(batch_status_command)


def main() -> None:
    """Execute the Typer application entrypoint."""
    app()


if __name__ == "__main__":
    main()
