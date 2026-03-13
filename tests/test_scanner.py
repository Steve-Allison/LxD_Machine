from __future__ import annotations

from pathlib import Path

from lxd.ingest.scanner import classify_source_type

_TEXT_EXTS = [".md", ".docling.json", ".docling.md"]
_ASSET_EXTS = [".png"]


def test_classify_plain_markdown() -> None:
    assert classify_source_type(Path("notes.md"), _TEXT_EXTS, _ASSET_EXTS) == "markdown"


def test_classify_docling_json() -> None:
    assert classify_source_type(Path("report.docling.json"), _TEXT_EXTS, _ASSET_EXTS) == "docling_json"


def test_classify_docling_md() -> None:
    assert classify_source_type(Path("report.docling.md"), _TEXT_EXTS, _ASSET_EXTS) == "docling_md"


def test_classify_image_png() -> None:
    assert classify_source_type(Path("diagram.png"), _TEXT_EXTS, _ASSET_EXTS) == "image_png"


def test_classify_unknown_extension_returns_none() -> None:
    assert classify_source_type(Path("readme.txt"), _TEXT_EXTS, _ASSET_EXTS) is None


def test_classify_docling_md_not_confused_with_plain_md() -> None:
    # .docling.md ends with .md — must still be classified as docling_md, not markdown
    result = classify_source_type(Path("deep/nested/analysis.docling.md"), _TEXT_EXTS, _ASSET_EXTS)
    assert result == "docling_md"
