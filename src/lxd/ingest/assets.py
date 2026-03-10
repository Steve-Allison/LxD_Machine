from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AssetParentLink:
    parent_rel_path: str | None
    link_method: str
    page_no: int | None


def infer_asset_parent(asset_rel_path: str) -> AssetParentLink:
    asset_path = Path(asset_rel_path)
    stem = asset_path.stem
    page_no = _extract_page_number(stem)
    parent_dir = asset_path.parent
    if parent_dir.name.endswith("_images"):
        base_name = parent_dir.name[: -len("_images")]
        docling_candidate = parent_dir.parent / f"{base_name}.docling.json"
        if docling_candidate.exists():
            return AssetParentLink(str(docling_candidate), "docling_sibling_images_dir", page_no)
        markdown_candidate = parent_dir.parent / f"{base_name}.md"
        if markdown_candidate.exists():
            return AssetParentLink(str(markdown_candidate), "markdown_sibling_images_dir", page_no)
    return AssetParentLink(None, "unresolved", page_no)


def _extract_page_number(stem: str) -> int | None:
    for token in stem.replace("-", "_").split("_"):
        if token.startswith("page") and token[4:].isdigit():
            return int(token[4:])
        if token.isdigit():
            return int(token)
    return None
