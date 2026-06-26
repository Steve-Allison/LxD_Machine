"""Parse wiki-style frontmatter and inline ``[[slug]]`` cross-references.

Responsibility:
    The curated wiki corpus uses a regular structure at the top of each page:

        # Title

        **Summary**: one-line description...
        **Scope**: ownership boundaries...
        **Sources**: file_a.md, file_b.pdf, file_c.docx
        **Last updated**: YYYY-MM-DD
        **Last reviewed**: YYYY-MM-DD

    Plus inline references using Obsidian/Bear-style ``[[slug]]`` notation.

    This module extracts those signals into structured metadata so the
    ingest pipeline can attach them to every chunk derived from the page,
    without re-parsing the markdown each time.

Design boundary:
    Pure parsing — no I/O, no logging, no mutation. Operates on strings.
    Empty / missing fields return ``None`` or empty tuples; callers decide
    whether absence is a problem.

Format expectations (validated against the live wiki, 144/147 pages):
    * Each field key appears at most once, on its own line.
    * The ``**Sources**:`` line is comma-separated; whitespace tolerated.
    * Parenthetical annotations on a source filename are stripped:
      ``"x.md (note about source)"`` -> ``"x.md"``.
    * ``[[slug]]`` references can include pipe-style aliases
      (``[[slug|display text]]``); only the slug is captured.
    * Section headers such as ``[[#anchor]]`` and image embeds ``![[file]]``
      are deliberately ignored.
"""

import re
from dataclasses import dataclass

_FRONTMATTER_FIELD_RE = re.compile(
    r"^\*\*(?P<key>Summary|Scope|Sources|Last updated|Last reviewed)\*\*:\s*(?P<value>.*?)\s*$",
    re.MULTILINE,
)

_FILE_EXTENSION_GROUP = r"(?:\.md|\.pdf|\.docx|\.json|\.txt)"

# Annotations only count when they FOLLOW a file extension — that's how the
# wiki distinguishes "filename includes parens" (e.g. "x (LX) Design.pdf")
# from "filename + post-citation note" (e.g. "x.md (source about VARK)").
_PAREN_ANNOTATION_RE = re.compile(rf"({_FILE_EXTENSION_GROUP})\s*\([^)]*\)", re.IGNORECASE)

# Match [[slug]] or [[slug|alias]] but NOT ![[file]] (image embed) or [[#anchor]].
# - Negative lookbehind for "!" to skip image embeds.
# - First captured chunk is the slug; alias after a pipe is ignored.
_WIKILINK_RE = re.compile(r"(?<!\!)\[\[(?!#)([^\[\]\|#]+?)(?:\|[^\[\]]*)?\]\]")


@dataclass(frozen=True, slots=True)
class WikiPageMetadata:
    """Structured metadata extracted from a wiki-formatted markdown page.

    Attributes:
        summary: First-line ``**Summary**:`` value, or ``None`` if absent.
        scope: ``**Scope**:`` value, or ``None`` if absent.
        cited_sources: Filenames listed on the ``**Sources**:`` line, in
            original order, with parenthetical annotations stripped and
            empty entries discarded. Empty when the line is missing.
        wiki_links: Slug references found in the page body, deduplicated
            (preserving first-seen order). Excludes image embeds (``![[..]]``)
            and section anchors (``[[#..]]``).
        last_updated: Raw ``**Last updated**:`` value (typically YYYY-MM-DD).
        last_reviewed: Raw ``**Last reviewed**:`` value.
    """

    summary: str | None = None
    scope: str | None = None
    cited_sources: tuple[str, ...] = ()
    wiki_links: tuple[str, ...] = ()
    last_updated: str | None = None
    last_reviewed: str | None = None

    @property
    def is_empty(self) -> bool:
        """True when no recognised wiki signal was extracted from the page."""
        return (
            self.summary is None
            and self.scope is None
            and not self.cited_sources
            and not self.wiki_links
            and self.last_updated is None
            and self.last_reviewed is None
        )


def parse_wiki_metadata(text: str) -> WikiPageMetadata:
    """Extract :class:`WikiPageMetadata` from a markdown page body.

    Args:
        text: Full text of the markdown source file (UTF-8 string).

    Returns:
        Parsed metadata. Fields are populated on a best-effort basis;
        missing patterns yield ``None`` / empty tuples rather than raising.
    """
    fields = {
        match.group("key"): match.group("value") for match in _FRONTMATTER_FIELD_RE.finditer(text)
    }
    return WikiPageMetadata(
        summary=_optional_field(fields.get("Summary")),
        scope=_optional_field(fields.get("Scope")),
        cited_sources=_parse_sources_line(fields.get("Sources")),
        wiki_links=extract_wiki_links(text),
        last_updated=_optional_field(fields.get("Last updated")),
        last_reviewed=_optional_field(fields.get("Last reviewed")),
    )


def extract_wiki_links(text: str) -> tuple[str, ...]:
    """Return de-duplicated ``[[slug]]`` references in first-seen order.

    Skips image embeds (``![[file]]``) and intra-page anchors (``[[#x]]``).
    Slugs are normalised to lowercase to match the wiki's filename convention.
    """
    seen: dict[str, None] = {}
    for match in _WIKILINK_RE.finditer(text):
        slug = match.group(1).strip().casefold()
        if slug and slug not in seen:
            seen[slug] = None
    return tuple(seen)


def _parse_sources_line(value: str | None) -> tuple[str, ...]:
    """Split a Sources line into clean filename tokens.

    Handles the one-off pattern in ``learning-styles-debate.md`` where
    sources carry parenthetical annotations: ``"x.md (note)"`` -> ``"x.md"``.
    """
    if value is None:
        return ()
    # Strip post-extension annotations BEFORE splitting on commas, because
    # annotation bodies themselves contain commas (e.g.
    # ``"x.md (note, with comma)"`` — naive split produces garbage).
    without_annotations = _PAREN_ANNOTATION_RE.sub(r"\1", value)
    cleaned = []
    for raw in without_annotations.split(","):
        stripped = raw.strip().strip("`")
        if stripped:
            cleaned.append(stripped)
    return tuple(cleaned)


def _optional_field(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None
