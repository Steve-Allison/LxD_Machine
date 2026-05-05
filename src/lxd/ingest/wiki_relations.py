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

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lxd.domain.ids import blake3_hex
from lxd.stores.models import ChunkRecord, ExtractedRelationRecord

_WIKI_REFERENCES_PREDICATE = "wiki_references"
_WIKI_RELATION_MODEL = "wiki_metadata"


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
    return index


def resolve_page_subject(source_rel_path: str, slug_index: Mapping[str, str]) -> str | None:
    """Resolve a wiki page's filename stem to a canonical_id.

    Returns ``None`` when the stem does not match any ontology entity —
    such pages cannot contribute ``wiki_references`` edges (no subject).
    """
    stem = Path(source_rel_path).stem
    return slug_index.get(stem) or slug_index.get(stem.lower())


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
        subject_id = resolve_page_subject(chunk.source_rel_path, slug_index)
        if subject_id is None:
            pages_without_subject.add(chunk.source_rel_path)
            continue
        for slug in chunk.wiki_links:
            object_id = slug_index.get(slug) or slug_index.get(slug.lower())
            if object_id is None:
                dangling.add(slug)
                continue
            if subject_id == object_id:
                continue
            dedup_key = (chunk.chunk_id, object_id)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            relations.append(
                ExtractedRelationRecord(
                    relation_id=blake3_hex(
                        chunk.chunk_id,
                        subject_id,
                        _WIKI_REFERENCES_PREDICATE,
                        object_id,
                    ),
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    source_rel_path=chunk.source_rel_path,
                    subject_entity_id=subject_id,
                    predicate=_WIKI_REFERENCES_PREDICATE,
                    object_entity_id=object_id,
                    confidence=1.0,
                    extraction_model=_WIKI_RELATION_MODEL,
                    extracted_at=extracted_at,
                )
            )
    return WikiRelationDerivationResult(
        relations=relations,
        dangling_slugs=tuple(sorted(dangling)),
        pages_without_subject=tuple(sorted(pages_without_subject)),
    )


def _slug_variants(canonical: str) -> tuple[str, ...]:
    lower = canonical.lower()
    return (
        canonical,
        canonical.replace("_", "-"),
        lower,
        lower.replace("_", "-"),
    )
