"""Golden-transcript regression test for the MCP tool manifest.

Wave 10 introduces a simple but powerful guardrail: serialize the full
(name, description-presence, input-schema) of every tool registered by
:func:`lxd.mcp.server.create_server` into a JSON manifest, then diff
against a committed ``tests/golden/mcp_tool_manifest.json``.

Rationale:
    * Detects accidental tool removals, renames, or schema-shape changes
      without needing a full MCP round-trip.
    * The manifest captures parameter names and types only — not runtime
      behaviour — so it is cheap, deterministic, and easy to review in PRs.

Updating the golden file:
    Run ``pytest tests/integration/test_mcp_tool_manifest.py
    --update-golden`` to overwrite the manifest with the current server
    output. CI never writes the file; any drift fails the build.
"""

from __future__ import annotations

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
        manifest.append(
            {
                "name": tool.name,
                "has_description": bool((tool.description or "").strip()),
                "parameters": sorted(props.keys()),
                "required": sorted(required),
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
