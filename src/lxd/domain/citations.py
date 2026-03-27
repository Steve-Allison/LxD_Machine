"""Build normalized citation labels for retrieval evidence."""

from __future__ import annotations


def make_citation_label(source_rel_path: str, page_no: int | None = None) -> str:
    """Build a stable citation label from source path and optional page number.

    Args:
        source_rel_path: Repository-relative source path used as the base label.
        page_no: Optional 1-based page number to append as a fragment.

    Returns:
        `source_rel_path` when `page_no` is `None`; otherwise `<path>#page=<page_no>`.
    """
    if page_no is None:
        return source_rel_path
    return f"{source_rel_path}#page={page_no}"
