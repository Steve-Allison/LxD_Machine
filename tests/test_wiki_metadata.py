"""Tests for the wiki frontmatter and ``[[slug]]`` cross-reference parser.

Real-world samples are pulled from the live curated wiki — the parser must
handle the actual edge cases (parenthetical-in-filename, post-extension
annotation, image embeds, pipe-aliased links) without inventing or losing
sources.
"""

from __future__ import annotations

from lxd.ingest.wiki_metadata import (
    WikiPageMetadata,
    extract_wiki_links,
    parse_wiki_metadata,
)


def test_parse_full_frontmatter_block() -> None:
    text = (
        "# ADDIE Model\n\n"
        "**Summary**: Five-phase ID framework.\n"
        "**Scope**: Owns analysis-design-develop-implement-evaluate.\n"
        "**Sources**: theory_addie_model.md, 2025 LX Toolkit.pdf\n"
        "**Last updated**: 2026-05-02\n"
        "**Last reviewed**: 2026-05-02\n\n"
        "Body content with [[backward-design]] reference.\n"
    )
    md = parse_wiki_metadata(text)
    assert md.summary == "Five-phase ID framework."
    assert md.scope == "Owns analysis-design-develop-implement-evaluate."
    assert md.cited_sources == ("theory_addie_model.md", "2025 LX Toolkit.pdf")
    assert md.last_updated == "2026-05-02"
    assert md.last_reviewed == "2026-05-02"
    assert md.wiki_links == ("backward-design",)


def test_parse_preserves_parens_inside_filename() -> None:
    """``2025 Learning Experience (LX) Design.pdf`` is a real filename — the
    ``(LX)`` parens must NOT be stripped as if they were a post-citation
    annotation."""
    text = "**Sources**: 2025 Learning Experience (LX) Design.pdf, theory_x.md\n"
    md = parse_wiki_metadata(text)
    assert md.cited_sources == (
        "2025 Learning Experience (LX) Design.pdf",
        "theory_x.md",
    )


def test_parse_strips_post_extension_annotations_with_embedded_commas() -> None:
    """``learning-styles-debate.md`` carries annotations like
    ``"x.md (note that itself contains, commas)"``. Naive split-on-comma
    produces garbage; the parser must strip annotations *before* splitting."""
    text = (
        "**Sources**: foo.md (source that invokes VARK, flagged as contested), "
        "bar.pdf (source that invokes 4MAT, including superseded framing)\n"
    )
    md = parse_wiki_metadata(text)
    assert md.cited_sources == ("foo.md", "bar.pdf")


def test_parse_handles_backticked_filenames() -> None:
    text = "**Sources**: `Research - Foo.md`\n"
    md = parse_wiki_metadata(text)
    assert md.cited_sources == ("Research - Foo.md",)


def test_parse_missing_sources_line_yields_empty_tuple() -> None:
    text = "# Title\n\nNo frontmatter at all.\n"
    md = parse_wiki_metadata(text)
    assert md.cited_sources == ()
    assert md.summary is None
    assert md.is_empty is True


def test_extract_wiki_links_dedupes_in_first_seen_order() -> None:
    text = "See [[alpha]] and [[beta]], also [[alpha]] again. Maybe [[gamma]]."
    assert extract_wiki_links(text) == ("alpha", "beta", "gamma")


def test_extract_wiki_links_skips_image_embeds() -> None:
    """``![[file.png]]`` is an Obsidian image embed, not a cross-reference."""
    text = "Inline ![[diagram.png]] but text [[real-link]] counts."
    assert extract_wiki_links(text) == ("real-link",)


def test_extract_wiki_links_skips_anchor_only_links() -> None:
    """``[[#section]]`` is an intra-page anchor — not a page reference."""
    text = "See [[#methodology]] above and [[other-page]] for detail."
    assert extract_wiki_links(text) == ("other-page",)


def test_extract_wiki_links_handles_pipe_alias() -> None:
    """``[[slug|Human Title]]`` should yield only the slug."""
    text = "Refer to [[mayers-multimedia-principles|Mayer's Principles]] for context."
    assert extract_wiki_links(text) == ("mayers-multimedia-principles",)


def test_extract_wiki_links_normalises_case() -> None:
    """Slugs are lowercased so case-only variants don't double-count."""
    text = "[[ADDIE-Model]] and later [[addie-model]]."
    assert extract_wiki_links(text) == ("addie-model",)


def test_empty_input_returns_empty_metadata() -> None:
    md = parse_wiki_metadata("")
    assert md == WikiPageMetadata()
    assert md.is_empty is True


def test_parse_summary_with_inline_bold() -> None:
    """Summary may contain bold markers that aren't field delimiters."""
    text = "**Summary**: A **bold** word here.\n**Sources**: x.md\n"
    md = parse_wiki_metadata(text)
    assert md.summary == "A **bold** word here."
    assert md.cited_sources == ("x.md",)


def test_parse_against_real_addie_page() -> None:
    """Smoke test against the live wiki — guards against parser regressions
    on real-world prose."""
    from pathlib import Path

    page = Path("/Users/steveallison/Documents/_Knowledge/wiki/addie-model.md")
    if not page.exists():
        return  # Wiki not present on this machine; skip silently.
    md = parse_wiki_metadata(page.read_text(encoding="utf-8"))
    # Real expectations from the actual file:
    assert "Five-phase" in (md.summary or "") or "five-phase" in (md.summary or "")
    assert len(md.cited_sources) >= 5
    # The (LX) filename must be preserved verbatim:
    assert any("(LX)" in s for s in md.cited_sources)
    # At least the major cross-refs ADDIE owns:
    assert "backward-design" in md.wiki_links
    assert "kirkpatricks-evaluation-model" in md.wiki_links
