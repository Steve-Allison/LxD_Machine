from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from lxd.retrieval.query_pipeline import (
    RankedChunk,
    _current_ingest_config,
    _lexical_signal_score,
    _merge_ranked_prefix,
    _unique_source_prefix,
)


def _chunk(chunk_id: str, *, source_rel_path: str | None = None, score_hint: str | None = None) -> RankedChunk:
    return RankedChunk(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        citation_label=chunk_id,
        source_rel_path=source_rel_path or f"{chunk_id}.md",
        source_path=f"/tmp/{(source_rel_path or f'{chunk_id}.md')}",
        source_filename=Path(source_rel_path or f"{chunk_id}.md").name,
        source_type="markdown",
        source_domain="guides",
        source_hash=f"hash-{chunk_id}",
        chunk_index=0,
        chunk_occurrence=0,
        token_count=10,
        text=f"text-{chunk_id}",
        score_hint=score_hint or chunk_id,
        metadata_json="{}",
        score=0.0,
    )


def test_merge_ranked_prefix_preserves_dense_tail() -> None:
    ranked = [_chunk("a"), _chunk("b"), _chunk("c"), _chunk("d"), _chunk("e")]
    ranked_prefix = [ranked[2], ranked[0], ranked[1]]

    merged = _merge_ranked_prefix(ranked, ranked_prefix)

    assert [item.chunk_id for item in merged] == ["c", "a", "b", "d", "e"]


def test_unique_source_prefix_deduplicates_sources() -> None:
    ranked = [
        _chunk("a1", source_rel_path="alpha.md"),
        _chunk("a2", source_rel_path="alpha.md"),
        _chunk("b1", source_rel_path="beta.md"),
        _chunk("c1", source_rel_path="gamma.md"),
    ]

    prefix = _unique_source_prefix(ranked, 3)

    assert [item.chunk_id for item in prefix] == ["a1", "b1", "c1"]


def test_lexical_signal_score_prefers_exact_named_source() -> None:
    target = _chunk(
        "target",
        source_rel_path="Theories/theory_addie_model.md",
        score_hint="ADDIE Model: A Comprehensive Summary of the Instructional Design Framework",
    )
    distractor = _chunk(
        "other",
        source_rel_path="Theories/Theory_John_Kellers_ARCS_model.md",
        score_hint="ARCS Model: Motivation and instructional design",
    )

    assert _lexical_signal_score("What is the ADDIE model?", target) > _lexical_signal_score(
        "What is the ADDIE model?", distractor
    )


def test_current_ingest_config_excludes_query_time_reranker_settings() -> None:
    config = SimpleNamespace(
        paths=SimpleNamespace(
            corpus_path=Path("/tmp/corpus"),
            ontology_path=Path("/tmp/ontology"),
            data_path=Path("/tmp/data"),
        ),
        chunking=SimpleNamespace(
            chunk_overlap=60,
            chunk_size=300,
            min_tokens=80,
            strategy="hybrid",
            tokenizer_backend="tiktoken",
            tokenizer_name="cl100k_base",
        ),
        models=SimpleNamespace(
            embed="nomic-embed-text",
            embed_dims=768,
            embed_backend="ollama",
            rerank="dengcao/Qwen3-Reranker-4B:Q4_K_M",
        ),
        reranker=SimpleNamespace(
            enabled=True,
            backend="llama_cpp",
            url="http://127.0.0.1:8012",
            endpoint="/v1/rerank",
        ),
    )

    snapshot = _current_ingest_config(config)

    assert "models.rerank" not in snapshot
    assert "reranker.enabled" not in snapshot
    assert "reranker.backend" not in snapshot
    assert "reranker.url" not in snapshot
    assert "reranker.endpoint" not in snapshot
