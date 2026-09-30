"""Derive synthetic entity-graph relations from wiki ``[[slug]]`` cross-references.

Wiki pages already encode hand-curated cross-references between concepts
(e.g. ``addie-model.md`` links to ``[[backward-design]]``). When the
slug maps to an ontology canonical_id, we materialise that link as an
``extracted_relations`` row with predicate ``wiki_references`` so the
knowledge-graph build sees it alongside LLM-extracted relations — no
LLM cost, no hallucination risk.

Resolution rule: ontology canonical_ids are snake_case
(``backward_design``); wiki slugs are kebab-case
(``backward-design``). The slug index folds both forms so a wiki slug
resolves regardless of which convention the ontology uses for that
entity.

Page-level citations (the ``Sources:`` frontmatter, persisted on each
chunk as ``cited_sources_json``) are NOT mapped here: source filenames
do not correspond to ontology entities, so they would not be useful
as entity-graph edges. The chunk-row column already carries them
through to retrieval and synthesis.
"""

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


@dataclass(frozen=True, slots=True)
class WikiRelationDerivationResult:
    """Outcome of deriving wiki-link relations for a batch of chunks."""

    relations: list[ExtractedRelationRecord]
    dangling_slugs: tuple[str, ...] = ()
    pages_without_subject: tuple[str, ...] = field(default_factory=tuple)


def build_slug_index(entity_definitions: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    """Build a ``slug -> canonical_id`` index for wiki-link resolution.

    Each canonical_id is registered under multiple normalisations so
    wiki slugs in either kebab-case or snake_case resolve to the same
    canonical (snake_case) id. Earlier entries win on conflict.

    Args:
        entity_definitions: Ontology entity dicts as produced by
            :func:`lxd.ontology.loader.load_ontology`. Each must carry a
            ``canonical_id`` string.

    Returns:
        Mapping from normalised slug forms to canonical_id.
    """
    index: dict[str, str] = {}
    for entity in entity_definitions:
        canonical = entity.get("canonical_id")
        if not isinstance(canonical, str) or not canonical:
            continue
        for variant in _slug_variants(canonical):
            index.setdefault(variant, canonical)
        tail = canonical.rsplit("/", 1)[-1]
        if tail != canonical:
            for variant in _slug_variants(tail):
                index.setdefault(variant, canonical)
        label = entity.get("label")
        if isinstance(label, str) and label.strip():
            for variant in _slug_variants(label.strip().replace(" ", "_")):
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
    """Emit ``wiki_references`` relations from chunk ``wiki_links``.

    For each chunk on a wiki page whose filename resolves to an
    ontology entity, emit one :class:`ExtractedRelationRecord` per
    resolved ``[[slug]]`` cross-reference. Output is deduplicated within
    a chunk (a slug repeated twice in the same chunk yields one
    relation). Self-references are skipped. Unresolved slugs are
    reported under ``dangling_slugs`` for caller-side logging.

    Args:
        chunk_records: Chunk rows about to be persisted for this run.
        slug_index: Result of :func:`build_slug_index` over the active
            ontology.
        extracted_at: ISO-8601 UTC timestamp stamped on each row.

    Returns:
        Derived relations plus diagnostic counts (dangling slugs,
        pages whose filename did not resolve to any entity).
    """
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
    """Emit one ``defers_to`` edge per frontmatter deferral.

    Edges hang off the first chunk of the page so a page-level hand-off
    is not copied once per chunk. The reason is stored on ``qualifier``.
    Both ends use :func:`graph_entity_id`.

    Args:
        chunk_records: Chunk rows for a single source page.
        defers: Deferrals parsed from that page.
        slug_index: Ontology slug index from :func:`build_slug_index`.
        extracted_at: ISO-8601 UTC timestamp stamped on each row.

    Returns:
        Relation rows. Empty when the page has no chunks or no deferrals.
    """
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


def _slug_variants(canonical: str) -> tuple[str, ...]:
    lower = canonical.lower()
    return (
        canonical,
        canonical.replace("_", "-"),
        lower,
        lower.replace("_", "-"),
    )
