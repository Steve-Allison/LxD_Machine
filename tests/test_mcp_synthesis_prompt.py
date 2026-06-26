"""Tests for the `lxd_synthesis_preamble` MCP prompt and the shared preamble (B-STACK-5)."""

from __future__ import annotations

import pytest
from mcp.types import TextContent

from lxd.synthesis.answering import (
    SYNTHESIS_PREAMBLE_BASE,
    SYNTHESIS_PREAMBLE_GRAPH_CONTEXT,
    SYNTHESIS_PREAMBLE_TRANSITIVE_SOURCES,
    synthesis_preamble,
)


def test_preamble_base_is_always_present() -> None:
    text = synthesis_preamble(has_transitive_sources=False, has_graph_context=False)
    assert text == SYNTHESIS_PREAMBLE_BASE


def test_preamble_includes_transitive_sources_when_flag_set() -> None:
    text = synthesis_preamble(has_transitive_sources=True, has_graph_context=False)
    assert SYNTHESIS_PREAMBLE_TRANSITIVE_SOURCES in text
    assert SYNTHESIS_PREAMBLE_GRAPH_CONTEXT not in text


def test_preamble_includes_graph_context_when_flag_set() -> None:
    text = synthesis_preamble(has_transitive_sources=False, has_graph_context=True)
    assert SYNTHESIS_PREAMBLE_GRAPH_CONTEXT in text
    assert SYNTHESIS_PREAMBLE_TRANSITIVE_SOURCES not in text


def test_preamble_includes_both_sub_sections_when_both_flags_set() -> None:
    text = synthesis_preamble(has_transitive_sources=True, has_graph_context=True)
    assert SYNTHESIS_PREAMBLE_BASE in text
    assert SYNTHESIS_PREAMBLE_TRANSITIVE_SOURCES in text
    assert SYNTHESIS_PREAMBLE_GRAPH_CONTEXT in text


@pytest.mark.asyncio
async def test_mcp_prompt_lists_synthesis_preamble() -> None:
    """`lxd_synthesis_preamble` is registered and listed by FastMCP."""
    from lxd.mcp.server import create_server

    mcp = create_server()
    prompts = await mcp.list_prompts()
    names = [p.name for p in prompts]

    assert "lxd_synthesis_preamble" in names, (
        f"`lxd_synthesis_preamble` should be registered; saw {names}."
    )


@pytest.mark.asyncio
async def test_mcp_prompt_returns_full_preamble_text() -> None:
    """Rendering the prompt returns the same text as `synthesis_preamble(True, True)`."""
    from lxd.mcp.server import create_server

    mcp = create_server()
    rendered = await mcp.render_prompt("lxd_synthesis_preamble")

    expected = synthesis_preamble(has_transitive_sources=True, has_graph_context=True)
    rendered_text = "\n".join(
        msg.content.text for msg in rendered.messages if isinstance(msg.content, TextContent)
    )
    assert expected in rendered_text, (
        f"Rendered prompt should contain the full preamble; saw:\n{rendered_text}\n"
        f"expected:\n{expected}"
    )
