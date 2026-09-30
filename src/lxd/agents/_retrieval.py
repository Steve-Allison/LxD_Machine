"""Shared retrieval formatting helpers for the design and critique agents.

Kept out of :mod:`lxd.agents.design` / :mod:`lxd.agents.critique` so
neither module needs to import from the other just to reuse this
formatting logic.
"""

from lxd.ingest.wiki_metadata import is_citable_source
from lxd.retrieval.query_pipeline import RankedChunk


def format_evidence_block(ranked: list[RankedChunk]) -> str:
    """Render ranked chunks as a citation-labelled evidence block for LLM prompts.

    ``_index.md`` is the wiki category catalog and is omitted so a table of
    contents does not become a cited source.
    """
    citable = [item for item in ranked if is_citable_source(item.source_rel_path)]
    if not citable:
        return "(no evidence retrieved)"
    return "\n\n".join(f"[{item.citation_label}]\n{item.text}" for item in citable)


def citation_labels(ranked: list[RankedChunk]) -> list[str]:
    """Return the citation label of each citable ranked chunk, in retrieval order."""
    return [item.citation_label for item in ranked if is_citable_source(item.source_rel_path)]
