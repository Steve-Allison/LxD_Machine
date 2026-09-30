"""Parse the curated wiki's YAML frontmatter, scope line, and ``[[slug]]`` links.

The live wiki (flat ``*.md`` pages) opens each content page with a YAML
block:

    ---
    type: Concept
    category: "Instructional Design & Methodology"
    description: "..."
    owns:
      - "Phrase this page is authoritative for"
    defers:
      "Why another page owns the neighbour": "other-slug"
    sources:
      - resource: "/raw/Some Source.md"
    ---

The body still carries a ``**Scope**:`` line and ``[[slug]]`` links.
Older pages that used bold ``**Summary**:`` / ``**Sources**:`` lines are
still accepted; YAML wins when both are present.

This module is pure parsing. Callers attach the result to chunks and
relations.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import yaml

_FRONTMATTER_FIELD_RE: Final = re.compile(
    r"^\*\*(?P<key>Summary|Scope|Sources|Last updated|Last reviewed)\*\*:\s*(?P<value>.*?)\s*$",
    re.MULTILINE,
)
_FENCE_RE: Final = re.compile(r"\A---\n(?P<body>.*?)\n---\n", re.DOTALL)
_FILE_EXTENSION_GROUP: Final = r"(?:\.md|\.pdf|\.docx|\.json|\.txt)"
_PAREN_ANNOTATION_RE: Final = re.compile(rf"({_FILE_EXTENSION_GROUP})\s*\([^)]*\)", re.IGNORECASE)
_WIKILINK_RE: Final = re.compile(r"(?<!\!)\[\[(?!#)([^\[\]\|#]+?)(?:\|[^\[\]]*)?\]\]")
_EVIDENCE_GRADE_RE: Final = re.compile(r"(?<!\[)\[([a-z][a-z0-9]*(?:-[a-z0-9]+)*)\](?!\()")
_INDEX_HEADING_RE: Final = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_WIKI_PAGE_METADATA_KEY: Final = "wiki_page"
_WIKI_PARSE_VERSION_KEY: Final = "wiki_parse_version"
WIKI_PARSE_VERSION: Final = 2

SYNTHESIS_EXCLUDED_FILENAMES: Final = frozenset({"_index.md"})


@dataclass(frozen=True, slots=True)
class WikiDeferral:
    """One ``defers`` entry: a reason and the page slug that owns the neighbour."""

    reason: str
    slug: str


@dataclass(frozen=True, slots=True)
class WikiMatcherPhrase:
    """A surface form the wiki already resolves to an owner page slug."""

    term: str
    slug: str
    term_source: str


@dataclass(frozen=True, slots=True)
class WikiPageMetadata:
    """Structured metadata extracted from one wiki page.

    ``cited_sources`` are provenance paths (typically ``/raw/...``). They
    are citations, not files to ingest. ``owns`` are phrases this page is
    authoritative for. ``defers`` are typed hand-offs to other slugs.
    """

    summary: str | None = None
    scope: str | None = None
    cited_sources: tuple[str, ...] = ()
    wiki_links: tuple[str, ...] = ()
    last_updated: str | None = None
    last_reviewed: str | None = None
    page_type: str | None = None
    category: str | None = None
    owns: tuple[str, ...] = ()
    defers: tuple[WikiDeferral, ...] = ()
    evidence_grades: tuple[str, ...] = ()

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
            and self.page_type is None
            and self.category is None
            and not self.owns
            and not self.defers
        )


def parse_wiki_metadata(text: str) -> WikiPageMetadata:
    """Extract :class:`WikiPageMetadata` from a markdown page.

    Args:
        text: Full text of the markdown source file.

    Returns:
        Parsed metadata. Missing patterns yield ``None`` or empty tuples.
    """
    yaml_block, body = split_frontmatter(text)
    yaml_fields = _parse_yaml_block(yaml_block)
    bold = {
        match.group("key"): match.group("value") for match in _FRONTMATTER_FIELD_RE.finditer(body)
    }
    description = _optional_str(yaml_fields.get("description"))
    generated = yaml_fields.get("generated")
    generated_at = None
    if isinstance(generated, dict):
        generated_at = _optional_str(generated.get("at"))
    return WikiPageMetadata(
        summary=description or _optional_field(bold.get("Summary")),
        scope=_optional_field(bold.get("Scope")),
        cited_sources=_yaml_sources(yaml_fields.get("sources"))
        or _parse_sources_line(bold.get("Sources")),
        wiki_links=extract_wiki_links(body),
        last_updated=_optional_field(bold.get("Last updated")) or generated_at,
        last_reviewed=_optional_field(bold.get("Last reviewed")),
        page_type=_optional_str(yaml_fields.get("type")),
        category=_optional_str(yaml_fields.get("category")),
        owns=_string_tuple(yaml_fields.get("owns")),
        defers=_parse_defers(yaml_fields.get("defers")),
        evidence_grades=extract_evidence_grades(body),
    )


def split_frontmatter(text: str) -> tuple[str | None, str]:
    """Split a leading YAML fence from the markdown body.

    Args:
        text: Full markdown source.

    Returns:
        ``(yaml_text_or_none, body)``. Body is the original text when no
        fence is present.
    """
    match = _FENCE_RE.match(text)
    if match is None:
        return None, text
    return match.group("body"), text[match.end() :]


def strip_wiki_frontmatter(text: str) -> str:
    """Return markdown with the leading YAML fence removed.

    Args:
        text: Full markdown source.

    Returns:
        Body text. Unfenced text is returned unchanged.
    """
    return split_frontmatter(text)[1]


def extract_wiki_links(text: str) -> tuple[str, ...]:
    """Return de-duplicated ``[[slug]]`` references in first-seen order.

    Skips image embeds (``![[file]]``) and intra-page anchors (``[[#x]]``).
    Slugs are normalised to lowercase.
    """
    seen: dict[str, None] = {}
    for match in _WIKILINK_RE.finditer(text):
        slug = match.group(1).strip().casefold()
        if slug and slug not in seen:
            seen[slug] = None
    return tuple(seen)


def extract_evidence_grades(text: str) -> tuple[str, ...]:
    """Return unique evidence-grade tokens such as ``one-researcher``.

    The wiki writes these as single-bracket tags in prose. Wiki links and
    markdown links are not grades.

    Args:
        text: Page or chunk text to scan.

    Returns:
        Tags in first-seen order.
    """
    seen: dict[str, None] = {}
    for match in _EVIDENCE_GRADE_RE.finditer(text):
        tag = match.group(1)
        if tag not in seen:
            seen[tag] = None
    return tuple(seen)


def merge_wiki_page_metadata(
    metadata_json: str,
    page: WikiPageMetadata,
    chunk_text: str,
) -> str:
    """Attach the page contract and this chunk's evidence grades to metadata.

    Existing JSON object keys are preserved. Grades are taken from
    ``chunk_text``, not the whole page, so a chunk only carries a grade
    it actually contains.

    Args:
        metadata_json: Chunk metadata JSON, possibly ``"{}"``.
        page: Parsed page metadata.
        chunk_text: Text of the chunk being stored.

    Returns:
        JSON object string.
    """
    grades = extract_evidence_grades(chunk_text)
    payload = _load_object(metadata_json)
    payload[_WIKI_PARSE_VERSION_KEY] = WIKI_PARSE_VERSION
    if page.is_empty and not grades:
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    payload[_WIKI_PAGE_METADATA_KEY] = {
        "type": page.page_type,
        "category": page.category,
        "summary": page.summary,
        "owns": list(page.owns),
        "defers": [{"reason": item.reason, "slug": item.slug} for item in page.defers],
        "evidence_grades": list(grades),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def defers_from_metadata(metadata_json: str) -> tuple[WikiDeferral, ...]:
    """Read ``defers`` back out of a chunk's ``wiki_page`` metadata.

    Used when a moved file is cloned: the chunk text is already stored,
    and the deferral edges have to be rewritten for the new path.

    Args:
        metadata_json: Chunk metadata JSON.

    Returns:
        Deferrals stored under ``wiki_page.defers``. Empty when absent.
    """
    payload = _load_object(metadata_json).get(_WIKI_PAGE_METADATA_KEY)
    if not isinstance(payload, dict):
        return ()
    return _parse_defers_list(payload.get("defers"))


def has_current_wiki_parse(metadata_json: str) -> bool:
    """Return whether chunk metadata was written by the current wiki parser.

    Incremental ingest skips a file whose bytes have not changed. Bumping
    :data:`WIKI_PARSE_VERSION` forces those files through the parser once.
    """
    version = _load_object(metadata_json).get(_WIKI_PARSE_VERSION_KEY)
    return version == WIKI_PARSE_VERSION


def is_citable_source(source_rel_path: str) -> bool:
    """Return whether a source may be cited as answer evidence.

    ``_index.md`` is the category catalog. It stays in the corpus so the
    category check can read it, and it stays out of synthesis evidence.
    """
    return Path(source_rel_path).name not in SYNTHESIS_EXCLUDED_FILENAMES


def load_wiki_matcher_phrases(corpus_root: Path) -> tuple[WikiMatcherPhrase, ...]:
    """Collect glossary aliases and ``owns`` phrases from a wiki root.

    Args:
        corpus_root: Directory of flat wiki markdown pages.

    Returns:
        Surface forms pointing at an owner slug. Empty when the directory
        has no matching pages.
    """
    if not corpus_root.is_dir():
        return ()
    phrases: list[WikiMatcherPhrase] = []
    glossary = corpus_root / "_glossary.md"
    if glossary.is_file():
        phrases.extend(parse_glossary_phrases(glossary.read_text(encoding="utf-8")))
    for path in sorted(corpus_root.rglob("*.md")):
        if path.name.startswith("_"):
            continue
        page = parse_wiki_metadata(path.read_text(encoding="utf-8"))
        slug = path.stem.casefold()
        for term in page.owns:
            phrases.append(WikiMatcherPhrase(term=term, slug=slug, term_source="wiki_owns"))
    return tuple(phrases)


def parse_glossary_phrases(text: str) -> tuple[WikiMatcherPhrase, ...]:
    """Parse the canonical-term table in ``_glossary.md``.

    Args:
        text: Full glossary markdown.

    Returns:
        One phrase per canonical term and per alias, each aimed at the
        owner slug in the row. Rows without an owner link are skipped.
    """
    phrases: list[WikiMatcherPhrase] = []
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 3 or cells[0] in {"Canonical term", "---"} or set(cells[0]) <= {"-", " "}:
            continue
        if cells[0].startswith("---") or cells[0].startswith("-"):
            continue
        owner = _first_wikilink(cells[2])
        if owner is None:
            continue
        canonical = cells[0]
        if canonical:
            phrases.append(
                WikiMatcherPhrase(term=canonical, slug=owner, term_source="wiki_glossary")
            )
        for alias in _split_aliases(cells[1]):
            phrases.append(WikiMatcherPhrase(term=alias, slug=owner, term_source="wiki_glossary"))
    return tuple(phrases)


def parse_index_categories(text: str) -> frozenset[str]:
    """Return ``##`` headings from ``_index.md``, which name the categories.

    Args:
        text: Full index markdown.

    Returns:
        Heading text with trailing markup stripped.
    """
    categories: set[str] = set()
    for match in _INDEX_HEADING_RE.finditer(text):
        heading = match.group(1).strip().strip("*").strip()
        if heading:
            categories.add(heading)
    return frozenset(categories)


def wiki_category_warnings(corpus_root: Path) -> list[str]:
    """Warn when a page category is not a heading in ``_index.md``.

    Args:
        corpus_root: Wiki root directory.

    Returns:
        Zero or one warning string. Missing index yields no warning.
    """
    index_path = corpus_root / "_index.md"
    if not index_path.is_file():
        return []
    allowed = parse_index_categories(index_path.read_text(encoding="utf-8"))
    if not allowed:
        return []
    unknown: list[str] = []
    for path in sorted(corpus_root.rglob("*.md")):
        if path.name.startswith("_"):
            continue
        category = parse_wiki_metadata(path.read_text(encoding="utf-8")).category
        if category and category not in allowed:
            unknown.append(f"{path.name}={category}")
    if not unknown:
        return []
    sample = ", ".join(unknown[:8])
    return [f"{len(unknown)} wiki pages use a category absent from _index.md headings: {sample}"]


def _parse_yaml_block(block: str | None) -> dict[str, Any]:
    if not block or not block.strip():
        return {}
    loaded = yaml.safe_load(block)
    if not isinstance(loaded, dict):
        return {}
    return loaded


def _yaml_sources(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    sources: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            sources.append(item.strip())
            continue
        if isinstance(item, dict):
            resource = item.get("resource")
            if isinstance(resource, str) and resource.strip():
                sources.append(resource.strip())
    return tuple(sources)


def _parse_defers(value: Any) -> tuple[WikiDeferral, ...]:
    if isinstance(value, dict):
        deferrals: list[WikiDeferral] = []
        for reason, slug in value.items():
            reason_text = _optional_str(reason)
            slug_text = _optional_str(slug)
            if reason_text and slug_text:
                deferrals.append(WikiDeferral(reason=reason_text, slug=slug_text.casefold()))
        return tuple(deferrals)
    return _parse_defers_list(value)


def _parse_defers_list(value: Any) -> tuple[WikiDeferral, ...]:
    if not isinstance(value, list):
        return ()
    deferrals: list[WikiDeferral] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        reason_text = _optional_str(item.get("reason"))
        slug_text = _optional_str(item.get("slug"))
        if reason_text and slug_text:
            deferrals.append(WikiDeferral(reason=reason_text, slug=slug_text.casefold()))
    return tuple(deferrals)


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    items: list[str] = []
    for item in value:
        text = _optional_str(item)
        if text:
            items.append(text)
    return tuple(items)


def _parse_sources_line(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    without_annotations = _PAREN_ANNOTATION_RE.sub(r"\1", value)
    cleaned: list[str] = []
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


def _optional_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _load_object(metadata_json: str) -> dict[str, Any]:
    if not metadata_json.strip():
        return {}
    try:
        loaded = json.loads(metadata_json)
    except json.JSONDecodeError:
        return {}
    if isinstance(loaded, dict):
        return loaded
    return {}


def _first_wikilink(cell: str) -> str | None:
    match = _WIKILINK_RE.search(cell)
    if match is None:
        return None
    slug = match.group(1).strip().casefold()
    return slug or None


def _split_aliases(cell: str) -> tuple[str, ...]:
    aliases: list[str] = []
    for raw in cell.split(","):
        alias = raw.strip()
        if alias and alias != "—":
            aliases.append(alias)
    return tuple(aliases)
