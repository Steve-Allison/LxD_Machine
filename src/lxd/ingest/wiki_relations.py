"""Derive synthetic entity-graph relations from wiki ``[[slug]]`` cross-references.

Wiki pages already encode hand-curated cross-references between concepts
(e.g. ``addie-model.md`` links to ``[[backward-design]]``). When the
slug maps to a ``Central_Configs`` v5.0.0 ``coe:Cxxxxx`` entity via
``pref_label``, ``all_terms.written_form`` (``aliases``), or
``all_entities.wiki_slug``, we materialise that link as an
``extracted_relations`` row with predicate ``wiki_references`` so the
knowledge-graph build sees it alongside LLM-extracted relations — no
LLM cost, no hallucination risk.

Page-level citations (the ``Sources:`` frontmatter, persisted on each
chunk as ``cited_sources_json``) are NOT mapped here: source filenames
do not correspond to ontology entities, so they would not be useful
as entity-graph edges. The chunk-row column already carries them
through to retrieval and synthesis.
"""

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from lxd.domain.ids import blake3_hex
from lxd.ingest.wiki_metadata import WikiDeferral
from lxd.stores.models import ChunkRecord, ExtractedRelationRecord

_WIKI_REFERENCES_PREDICATE: Final = "wiki_references"
_DEFERS_TO_PREDICATE: Final = "defers_to"
_WIKI_RELATION_MODEL: Final = "wiki_metadata"
_WIKI_ENTITY_PREFIX: Final = "wiki:"
_APOSTROPHE_RE: Final = re.compile(r"['’`]")
_NON_ALNUM_RE: Final = re.compile(r"[^A-Za-z0-9]+")


@dataclass(frozen=True, slots=True)
class WikiRelationDerivationResult:
    """Outcome of deriving wiki-link relations for a batch of chunks."""

    relations: list[ExtractedRelationRecord]
    dangling_slugs: tuple[str, ...] = ()
    pages_without_subject: tuple[str, ...] = field(default_factory=tuple)


def build_slug_index(entity_definitions: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    """Build a ``slug -> coe:Cxxxxx`` index for wiki-link resolution.

    Indexes ``Central_Configs`` v5.0.0 entities in deterministic priority order:
      1. ``label`` (``all_entities.pref_label``)
      2. ``aliases`` (``all_terms.written_form``)
      3. ``wiki_slug`` (``all_entities.wiki_slug``)

    Opaque ``coe:Cxxxxx`` identifiers are never indexed as slugs themselves.
    """
    entities = [
        entity
        for entity in entity_definitions
        if isinstance(entity.get("canonical_id"), str) and str(entity["canonical_id"]).strip()
    ]
    index: dict[str, str] = {}

    # Pass 1: primary preferred labels
    for entity in entities:
        canonical = str(entity["canonical_id"]).strip()
        label = entity.get("label")
        if isinstance(label, str) and label.strip():
            for variant in _phrase_slug_variants(label):
                index.setdefault(variant, canonical)

    # Pass 2: controlled lexical terms (all_terms.written_form projected onto aliases)
    for entity in entities:
        canonical = str(entity["canonical_id"]).strip()
        aliases = entity.get("aliases")
        if isinstance(aliases, Iterable) and not isinstance(aliases, (str, bytes)):
            for alias in aliases:
                if isinstance(alias, str) and alias.strip():
                    for variant in _phrase_slug_variants(alias):
                        index.setdefault(variant, canonical)

    # Pass 3: explicit all_entities.wiki_slug bridge
    for entity in entities:
        canonical = str(entity["canonical_id"]).strip()
        wiki_slug = entity.get("wiki_slug")
        if isinstance(wiki_slug, str) and wiki_slug.strip():
            for variant in _phrase_slug_variants(wiki_slug):
                index.setdefault(variant, canonical)

    return index


def resolve_page_subject(source_rel_path: str, slug_index: Mapping[str, str]) -> str | None:
    """Resolve a wiki page's filename stem to an ontology canonical_id.

    Returns ``None`` when the stem is not an ontology entity. Callers that
    still need a graph id should use :func:`graph_entity_id`.
    """
    stem = Path(source_rel_path).stem
    return slug_index.get(stem) or slug_index.get(stem.lower())


def graph_entity_id(slug: str, slug_index: Mapping[str, str]) -> str:
    """Return the ontology id for ``slug``, or ``wiki:<slug>`` when it has none.

    The page graph is keyed by slug. An ontology match replaces the wiki
    id so the same concept is not stored twice.
    """
    resolved = slug_index.get(slug) or slug_index.get(slug.lower())
    if resolved:
        return resolved
    return f"{_WIKI_ENTITY_PREFIX}{slug}"


def derive_wiki_link_relations(
    *,
    chunk_records: Iterable[ChunkRecord],
    slug_index: Mapping[str, str],
    extracted_at: str,
) -> WikiRelationDerivationResult:
    """Emit ``wiki_references`` relations from chunk ``wiki_links``."""
    relations: list[ExtractedRelationRecord] = []
    dangling: set[str] = set()
    pages_without_subject: set[str] = set()
    seen: set[tuple[str, str]] = set()

    for chunk in chunk_records:
        if not chunk.wiki_links:
            continue
        subject_slug = Path(chunk.source_rel_path).stem.casefold()
        if resolve_page_subject(chunk.source_rel_path, slug_index) is None:
            pages_without_subject.add(chunk.source_rel_path)
        subject_id = graph_entity_id(subject_slug, slug_index)
        for slug in chunk.wiki_links:
            object_id = graph_entity_id(slug, slug_index)
            if subject_id == object_id:
                continue
            dedup_key = (chunk.chunk_id, object_id)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            relations.append(
                _wiki_relation(
                    chunk=chunk,
                    subject_id=subject_id,
                    predicate=_WIKI_REFERENCES_PREDICATE,
                    object_id=object_id,
                    qualifier="",
                    extracted_at=extracted_at,
                )
            )
    return WikiRelationDerivationResult(
        relations=relations,
        dangling_slugs=tuple(sorted(dangling)),
        pages_without_subject=tuple(sorted(pages_without_subject)),
    )


def derive_defer_relations(
    *,
    chunk_records: Iterable[ChunkRecord],
    defers: tuple[WikiDeferral, ...],
    slug_index: Mapping[str, str],
    extracted_at: str,
) -> list[ExtractedRelationRecord]:
    """Emit one ``defers_to`` edge per frontmatter deferral."""
    if not defers:
        return []
    first = next(iter(chunk_records), None)
    if first is None:
        return []
    subject_slug = Path(first.source_rel_path).stem.casefold()
    subject_id = graph_entity_id(subject_slug, slug_index)
    relations: list[ExtractedRelationRecord] = []
    seen: set[str] = set()
    for deferral in defers:
        object_id = graph_entity_id(deferral.slug, slug_index)
        if subject_id == object_id or object_id in seen:
            continue
        seen.add(object_id)
        relations.append(
            _wiki_relation(
                chunk=first,
                subject_id=subject_id,
                predicate=_DEFERS_TO_PREDICATE,
                object_id=object_id,
                qualifier=deferral.reason,
                extracted_at=extracted_at,
            )
        )
    return relations


def _wiki_relation(
    *,
    chunk: ChunkRecord,
    subject_id: str,
    predicate: str,
    object_id: str,
    qualifier: str,
    extracted_at: str,
) -> ExtractedRelationRecord:
    return ExtractedRelationRecord(
        relation_id=blake3_hex(
            chunk.chunk_id,
            subject_id,
            predicate,
            object_id,
            qualifier,
        ),
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        source_rel_path=chunk.source_rel_path,
        subject_entity_id=subject_id,
        predicate=predicate,
        object_entity_id=object_id,
        confidence=1.0,
        extraction_model=_WIKI_RELATION_MODEL,
        extracted_at=extracted_at,
        qualifier=qualifier,
    )


def _phrase_slug_variants(phrase: str) -> tuple[str, ...]:
    cleaned = phrase.strip()
    if not cleaned:
        return ()
    variants: list[str] = []
    for candidate_base in (cleaned, _APOSTROPHE_RE.sub("", cleaned)):
        kebab = _NON_ALNUM_RE.sub("-", candidate_base).strip("-")
        snake = kebab.replace("-", "_")
        for form in (
            candidate_base,
            candidate_base.lower(),
            kebab,
            kebab.lower(),
            snake,
            snake.lower(),
        ):
            if form and form not in variants:
                variants.append(form)
    return tuple(variants)
