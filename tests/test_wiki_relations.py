"""Tests for the wiki-link -> Central_Configs v5.0.0 coe:Cxxxxx relation derivation."""

from typing import Any

from lxd.ingest.wiki_metadata import WikiDeferral
from lxd.ingest.wiki_relations import (
    build_slug_index,
    derive_defer_relations,
    derive_wiki_link_relations,
    resolve_page_subject,
)
from lxd.stores.models import ChunkRecord


def _entity(
    canonical_id: str,
    *,
    label: str,
    aliases: list[str] | None = None,
    wiki_slug: str | None = None,
) -> dict[str, Any]:
    return {
        "canonical_id": canonical_id,
        "label": label,
        "aliases": aliases or [],
        "wiki_slug": wiki_slug,
    }


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


def test_build_slug_index_resolves_wiki_slug_pref_label_and_written_form_to_curie() -> None:
    """wiki_slug, pref_label, and all_terms.written_form resolve to coe:Cxxxxx; raw CURIEs never enter index."""
    index = build_slug_index(
        [
            _entity(
                "coe:C05201",
                label="Backward Design",
                aliases=["Understanding by Design", "UbD"],
                wiki_slug="backward-design",
            ),
            _entity(
                "coe:C05202",
                label="ADDIE Model",
                aliases=["ADDIE"],
                wiki_slug="addie-model",
            ),
        ]
    )
    assert index["backward-design"] == "coe:C05201"
    assert index["backward_design"] == "coe:C05201"
    assert index["understanding-by-design"] == "coe:C05201"
    assert index["ubd"] == "coe:C05201"
    assert index["addie-model"] == "coe:C05202"
    assert index["addie"] == "coe:C05202"
    assert "coe:C05201" not in index
    assert "coe-c05201" not in index


def test_build_slug_index_prioritizes_pref_label_before_shared_wiki_slug() -> None:
    """Parent framework keeps shared wiki_slug while child stages remain reachable by pref_label."""
    index = build_slug_index(
        [
            _entity(
                "coe:C06002",
                label="Minto Pyramid & SCQA Architecture",
                wiki_slug="pyramid-principle",
            ),
            _entity(
                "coe:C06025",
                label="SCQA Situation",
                wiki_slug="pyramid-principle",
            ),
        ]
    )
    assert index["pyramid-principle"] == "coe:C06002"
    assert index["scqa-situation"] == "coe:C06025"


def test_build_slug_index_skips_entities_without_canonical_id() -> None:
    """Entities missing a canonical_id are tolerated, not crashed on."""
    index = build_slug_index(
        [
            _entity("coe:C05105", label="Scaffolding"),
            {"label": "no-id"},
            {},
        ]
    )
    assert index["scaffolding"] == "coe:C05105"
    assert "no-id" not in index


def test_resolve_page_subject_strips_extension_and_resolves() -> None:
    index = build_slug_index([_entity("coe:C05202", label="ADDIE Model", wiki_slug="addie-model")])
    assert resolve_page_subject("wiki/addie-model.md", index) == "coe:C05202"
    assert resolve_page_subject("wiki/unknown-page.md", index) is None


def test_derive_wiki_link_relations_keeps_unmatched_slugs() -> None:
    """A slug with no ontology entity is still an edge, keyed wiki:<slug>."""
    index = build_slug_index(
        [
            _entity("coe:C05202", label="ADDIE Model", wiki_slug="addie-model"),
            _entity("coe:C05201", label="Backward Design", wiki_slug="backward-design"),
        ]
    )
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
    by_object = {rel.object_entity_id: rel for rel in result.relations}
    assert set(by_object) == {"coe:C05201", "wiki:unknown-concept"}
    rel = by_object["coe:C05201"]
    assert rel.subject_entity_id == "coe:C05202"
    assert rel.predicate == "wiki_references"
    assert rel.extraction_model == "wiki_metadata"
    assert rel.confidence == 1.0
    assert rel.qualifier == ""
    assert result.dangling_slugs == ()
    assert result.pages_without_subject == ()


def test_derive_wiki_link_relations_skips_self_references() -> None:
    """A page that links to itself via [[own-slug]] yields no edge."""
    index = build_slug_index([_entity("coe:C05202", label="ADDIE Model", wiki_slug="addie-model")])
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
    index = build_slug_index(
        [
            _entity("coe:C05202", label="ADDIE Model", wiki_slug="addie-model"),
            _entity("coe:C05201", label="Backward Design", wiki_slug="backward-design"),
        ]
    )
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
    """A page whose filename has no ontology match is reported under wiki:<slug>."""
    index = build_slug_index(
        [_entity("coe:C05201", label="Backward Design", wiki_slug="backward-design")]
    )
    chunks = [
        _chunk(
            chunk_id="c1",
            source_rel_path="wiki/orphan-page.md",
            wiki_links=("backward-design",),
        ),
    ]
    result = derive_wiki_link_relations(chunk_records=chunks, slug_index=index, extracted_at="t")
    assert len(result.relations) == 1
    assert result.relations[0].subject_entity_id == "wiki:orphan-page"
    assert result.relations[0].object_entity_id == "coe:C05201"
    assert result.pages_without_subject == ("wiki/orphan-page.md",)


def test_derive_wiki_link_relations_handles_chunks_with_no_wiki_links() -> None:
    index = build_slug_index([_entity("coe:C05202", label="ADDIE Model", wiki_slug="addie-model")])
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
    index = build_slug_index(
        [
            _entity("coe:C05202", label="ADDIE Model", wiki_slug="addie-model"),
            _entity("coe:C05201", label="Backward Design", wiki_slug="backward-design"),
        ]
    )
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


def test_derive_defer_relations_keeps_reason_and_resolves_slug() -> None:
    index = build_slug_index(
        [
            _entity("coe:C05202", label="ADDIE Model", wiki_slug="addie-model"),
            _entity("coe:C05201", label="Backward Design", wiki_slug="backward-design"),
        ]
    )
    chunks = [
        _chunk(chunk_id="c1", source_rel_path="addie-model.md"),
        _chunk(chunk_id="c2", source_rel_path="addie-model.md"),
    ]
    relations = derive_defer_relations(
        chunk_records=chunks,
        defers=(WikiDeferral(reason="Stage 1 design logic", slug="backward-design"),),
        slug_index=index,
        extracted_at="t",
    )
    assert len(relations) == 1
    assert relations[0].chunk_id == "c1"
    assert relations[0].predicate == "defers_to"
    assert relations[0].subject_entity_id == "coe:C05202"
    assert relations[0].object_entity_id == "coe:C05201"
    assert relations[0].qualifier == "Stage 1 design logic"
