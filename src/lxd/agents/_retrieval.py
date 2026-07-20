"""Shared retrieval formatting helpers for the design and critique agents.

Kept out of :mod:`lxd.agents.design` / :mod:`lxd.agents.critique` so
neither module needs to import from the other just to reuse this
formatting logic.
"""

from lxd.retrieval.query_pipeline import RankedChunk


def format_evidence_block(ranked: list[RankedChunk]) -> str:
    """Render ranked chunks as a citation-labelled evidence block for LLM prompts."""
    if not ranked:
        return "(no evidence retrieved)"
    return "\n\n".join(f"[{item.citation_label}]\n{item.text}" for item in ranked)


def citation_labels(ranked: list[RankedChunk]) -> list[str]:
    """Return the citation label of each ranked chunk, in retrieval order."""
    return [item.citation_label for item in ranked]
