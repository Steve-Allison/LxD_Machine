"""Tests for the wiki-link → entity-graph relation derivation."""

from __future__ import annotations

from lxd.ingest.wiki_relations import (
    build_slug_index,
    derive_wiki_link_relations,
    resolve_page_subject,
)
from lxd.stores.models import ChunkRecord


def _entity(canonical_id: str) -> dict[str, str]:
    return {"canonical_id": canonical_id}


def _chunk(
    *,
    chunk_id: str,
    source_rel_path: str,
    wiki_links: tuple[str, ...] = (),
    cited_sources: tuple[str, ...] = (),
) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        source_rel_path=source_rel_path,
        source_filename=source_rel_path.rsplit("/", 1)[-1],
        source_type="markdown",
        source_domain="wiki",
        source_hash=f"hash-{chunk_id}",
        citation_label=f"{source_rel_path}#0",
        chunk_index=0,
        chunk_occurrence=0,
        token_count=10,
        text=f"text-{chunk_id}",
        chunk_hash=f"ch-{chunk_id}",
        score_hint="hint",
        metadata_json="{}",
        vector=[0.1, 0.2, 0.3],
        embedding_model="m",
        embedding_dims=3,
        cited_sources=cited_sources,
        wiki_links=wiki_links,
    )


def test_build_slug_index_folds_kebab_and_snake_case() -> None:
    """A canonical_id like ``backward_design`` must resolve from either
    snake_case or kebab-case slugs."""
    index = build_slug_index([_entity("backward_design"), _entity("ADDIE_Model")])
    assert index["backward_design"] == "backward_design"
    assert index["backward-design"] == "backward_design"
    assert index["ADDIE_Model"] == "ADDIE_Model"
    assert index["addie_model"] == "ADDIE_Model"
    assert index["addie-model"] == "ADDIE_Model"


def test_build_slug_index_skips_entities_without_canonical_id() -> None:
    """Entities missing a ``canonical_id`` are tolerated, not crashed on."""
    index = build_slug_index([{"canonical_id": "valid"}, {"label": "no-id"}, {}])
    assert index == {"valid": "valid"}


def test_resolve_page_subject_strips_extension_and_resolves() -> None:
    index = build_slug_index([_entity("addie_model")])
    assert resolve_page_subject("wiki/addie-model.md", index) == "addie_model"
    assert resolve_page_subject("wiki/unknown-page.md", index) is None


def test_derive_wiki_link_relations_emits_resolved_edges_only() -> None:
    """Resolved [[slug]] becomes a wiki_references edge; unresolved goes
    to dangling_slugs."""
    index = build_slug_index([_entity("addie_model"), _entity("backward_design")])
    chunks = [
        _chunk(
            chunk_id="c1",
            source_rel_path="wiki/addie-model.md",
            wiki_links=("backward-design", "unknown-concept"),
        ),
    ]
    result = derive_wiki_link_relations(
        chunk_records=chunks,
        slug_index=index,
        extracted_at="2026-05-05T00:00:00+00:00",
    )
    assert len(result.relations) == 1
    rel = result.relations[0]
    assert rel.subject_entity_id == "addie_model"
    assert rel.object_entity_id == "backward_design"
    assert rel.predicate == "wiki_references"
    assert rel.extraction_model == "wiki_metadata"
    assert rel.confidence == 1.0
    assert result.dangling_slugs == ("unknown-concept",)
    assert result.pages_without_subject == ()


def test_derive_wiki_link_relations_skips_self_references() -> None:
    """A page that links to itself via [[own-slug]] yields no edge."""
    index = build_slug_index([_entity("addie_model")])
    chunks = [
        _chunk(
            chunk_id="c1",
            source_rel_path="wiki/addie-model.md",
            wiki_links=("addie-model",),
        ),
    ]
    result = derive_wiki_link_relations(
        chunk_records=chunks,
        slug_index=index,
        extracted_at="t",
    )
    assert result.relations == []


def test_derive_wiki_link_relations_dedupes_within_chunk() -> None:
    """The same slug appearing twice in one chunk yields a single edge."""
    index = build_slug_index([_entity("addie_model"), _entity("backward_design")])
    chunks = [
        _chunk(
            chunk_id="c1",
            source_rel_path="wiki/addie-model.md",
            wiki_links=("backward-design", "backward-design"),
        ),
    ]
    result = derive_wiki_link_relations(chunk_records=chunks, slug_index=index, extracted_at="t")
    assert len(result.relations) == 1


def test_derive_wiki_link_relations_records_pages_without_subject() -> None:
    """A page whose filename has no ontology match is reported (so the
    user can decide whether to add an entity or accept the page as
    descriptive-only)."""
    index = build_slug_index([_entity("backward_design")])
    chunks = [
        _chunk(
            chunk_id="c1",
            source_rel_path="wiki/orphan-page.md",
            wiki_links=("backward-design",),
        ),
    ]
    result = derive_wiki_link_relations(chunk_records=chunks, slug_index=index, extracted_at="t")
    assert result.relations == []
    assert result.pages_without_subject == ("wiki/orphan-page.md",)


def test_derive_wiki_link_relations_handles_chunks_with_no_wiki_links() -> None:
    """Chunks without wiki_links are skipped silently — no error, no
    diagnostic noise."""
    index = build_slug_index([_entity("addie_model")])
    chunks = [
        _chunk(
            chunk_id="c1",
            source_rel_path="wiki/addie-model.md",
            wiki_links=(),
        ),
    ]
    result = derive_wiki_link_relations(chunk_records=chunks, slug_index=index, extracted_at="t")
    assert result.relations == []
    assert result.dangling_slugs == ()
    assert result.pages_without_subject == ()


def test_derive_wiki_link_relation_ids_are_deterministic() -> None:
    """The same (chunk_id, subject, predicate, object) tuple must hash
    to the same relation_id on every run — re-ingest must not generate
    duplicate KG edges."""
    index = build_slug_index([_entity("addie_model"), _entity("backward_design")])
    chunks = [
        _chunk(
            chunk_id="c1",
            source_rel_path="wiki/addie-model.md",
            wiki_links=("backward-design",),
        ),
    ]
    first = derive_wiki_link_relations(
        chunk_records=chunks, slug_index=index, extracted_at="t1"
    ).relations[0]
    second = derive_wiki_link_relations(
        chunk_records=chunks, slug_index=index, extracted_at="t2"
    ).relations[0]
    assert first.relation_id == second.relation_id
