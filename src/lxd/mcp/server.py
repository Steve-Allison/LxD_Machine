from __future__ import annotations

import argparse
from pathlib import Path

from fastmcp import FastMCP

from lxd.mcp.tools import (
    corpus_status_tool,
    find_documents_for_concept_tool,
    get_entity_types_tool,
    get_related_concepts_tool,
    initialize_tools,
    search_corpus_tool,
)

mcp = FastMCP("lxd-machine")


@mcp.tool()
def corpus_status() -> dict[str, object]:
    return corpus_status_tool()


@mcp.tool()
def get_entity_types() -> list[str]:
    return get_entity_types_tool()


@mcp.tool()
def get_related_concepts(entity_id: str) -> list[dict[str, object]]:
    return get_related_concepts_tool(entity_id)


@mcp.tool()
def search_corpus(
    terms: str, domain: str | None = None, limit: int = 10
) -> list[dict[str, object]]:
    return search_corpus_tool(terms, domain, limit)


@mcp.tool()
def find_documents_for_concept(
    entity_id: str, hops: int = 1, limit: int = 10
) -> list[dict[str, object]]:
    return find_documents_for_concept_tool(entity_id, hops, limit)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile")
    parser.add_argument("--config")
    args = parser.parse_args()
    initialize_tools(
        Path.cwd(),
        profile=args.profile,
        config_path=Path(args.config).resolve() if args.config else None,
    )
    mcp.run()


if __name__ == "__main__":
    main()
