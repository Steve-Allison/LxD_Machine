from __future__ import annotations


def make_citation_label(source_rel_path: str, page_no: int | None = None) -> str:
    if page_no is None:
        return source_rel_path
    return f"{source_rel_path}#page={page_no}"
