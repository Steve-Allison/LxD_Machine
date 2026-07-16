"""Golden-transcript regression test for the MCP tool manifest.

Serialises the client-facing surface of every tool registered by
:func:`lxd.mcp.server.create_server` — name, description-presence,
input-schema parameter/required lists, output-schema, and semantic
annotations — into a JSON manifest, then diffs against a committed
``tests/golden/mcp_tool_manifest.json``.

Rationale:
    * Detects accidental tool removals, renames, input-schema changes,
      output-schema changes, and annotation-hint drift without needing a
      full MCP round-trip.
    * Clients bind to *output* shape as tightly as to input names, so
      output_schema drift is a breaking change the golden must catch.
    * ToolAnnotations (readOnlyHint / idempotentHint / openWorldHint)
      are part of the contract the client uses to reason about safety;
      hint drift is a breaking change the golden must catch too.

Updating the golden file:
    Run ``pytest tests/integration/test_mcp_tool_manifest.py
    --update-golden`` to overwrite the manifest with the current server
    output. CI never writes the file; any drift fails the build.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from lxd.mcp.server import create_server

pytestmark = [pytest.mark.integration]


GOLDEN_PATH = Path(__file__).resolve().parents[1] / "golden" / "mcp_tool_manifest.json"


async def _collect_manifest() -> list[dict[str, Any]]:
    """Return a sorted list of tool descriptors from a fresh server."""
    mcp = create_server()
    tools = await mcp.list_tools()
    manifest: list[dict[str, Any]] = []
    for tool in tools:
        schema = dict(tool.parameters) if tool.parameters else {}
        props = schema.get("properties", {})
        required = list(schema.get("required", []))
        output_schema = dict(tool.output_schema) if tool.output_schema else {}
        annotations_model = getattr(tool, "annotations", None)
        annotations: dict[str, Any] = (
            annotations_model.model_dump(exclude_none=True) if annotations_model else {}
        )
        manifest.append(
            {
                "name": tool.name,
                "has_description": bool((tool.description or "").strip()),
                "parameters": sorted(props.keys()),
                "required": sorted(required),
                "output_schema": output_schema,
                "annotations": annotations,
            }
        )
    manifest.sort(key=lambda item: item["name"])
    return manifest


async def test_mcp_tool_manifest_matches_golden(
    request: pytest.FixtureRequest,
) -> None:
    """Assert that the live MCP tool manifest matches the committed snapshot.

    Fails loudly when a tool is added, removed, renamed, or has a changed
    input schema — all of which are breaking changes for clients.
    """
    current = await _collect_manifest()

    if getattr(request.config.option, "update_golden", False):
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(
            json.dumps(current, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return

    if not GOLDEN_PATH.exists():
        pytest.fail(
            f"Missing MCP tool golden file at {GOLDEN_PATH}. "
            "Re-run with --update-golden to create it."
        )

    expected = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    assert current == expected, (
        "MCP tool manifest drifted from the golden file. "
        "If the change is intentional, re-run with --update-golden."
    )
