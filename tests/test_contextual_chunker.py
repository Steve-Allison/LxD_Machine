"""Tests for the contextual retrieval chunker.

The chunker has three integration points exercised here:

* ``augment_chunk_for_embedding`` — pure function, no I/O.
* ``generate_chunk_context`` — calls Ollama; we monkeypatch the
  client so the test runs without a live model.
* ``lookup_summaries`` / ``store_summaries`` — round-trip through a
  real LanceDB table (project no-mocks rule for internal stores).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import lancedb

from lxd.ingest import contextual_chunker
from lxd.ingest.contextual_chunker import (
    augment_chunk_for_embedding,
    generate_chunk_context,
    lookup_summaries,
    open_summary_cache_table,
    store_summaries,
)


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        chunking=SimpleNamespace(
            contextual_summary_enabled=True,
            contextual_summary_model="qwen3:14b",
            contextual_summary_temperature=0.0,
            contextual_summary_timeout_secs=60,
            contextual_summary_max_tokens=80,
        ),
        models=SimpleNamespace(llm_no_think=True),
        ollama=SimpleNamespace(url="http://localhost:11434"),
    )


def test_augment_chunk_for_embedding_prepends_summary() -> None:
    augmented = augment_chunk_for_embedding(
        "Backward design starts from the desired learning outcome.",
        "This chunk introduces backward design within the ADDIE methodology.",
    )
    assert augmented.startswith(
        "This chunk introduces backward design within the ADDIE methodology."
    )
    assert "Backward design starts from" in augmented


def test_augment_chunk_for_embedding_with_empty_summary_is_passthrough() -> None:
    """Generation failures yield empty summary — augmentation must be a no-op
    so the embedder still gets the original chunk text."""
    text = "Body text only."
    assert augment_chunk_for_embedding(text, "") == text


def test_summary_cache_round_trips_strings(tmp_path: Path) -> None:
    """Stored summaries reappear via ``lookup_summaries`` keyed by chunk_hash."""
    database = lancedb.connect(str(tmp_path / "lancedb"))
    table = open_summary_cache_table(database)
    written = store_summaries(
        table,
        chunk_hashes=["h1", "h2", "h3"],
        summaries=[
            "Summary one.",
            "Summary two.",
            "",  # empty: must NOT round-trip
        ],
        model="qwen3:14b",
    )
    assert written == 2
    result = lookup_summaries(table, chunk_hashes=["h1", "h2", "h3"], model="qwen3:14b")
    assert result.hits == {0: "Summary one.", 1: "Summary two."}
    assert result.misses_indices == [2]


def test_summary_cache_lookup_handles_empty_input(tmp_path: Path) -> None:
    database = lancedb.connect(str(tmp_path / "lancedb"))
    table = open_summary_cache_table(database)
    result = lookup_summaries(table, chunk_hashes=[], model="qwen3:14b")
    assert result.hits == {}
    assert result.misses_indices == []


def test_summary_cache_keys_are_model_scoped(tmp_path: Path) -> None:
    """A summary stored under model A must NOT surface under model B —
    swapping the local LLM should produce a fresh cache namespace."""
    database = lancedb.connect(str(tmp_path / "lancedb"))
    table = open_summary_cache_table(database)
    store_summaries(
        table,
        chunk_hashes=["h1"],
        summaries=["Summary under A."],
        model="model-a",
    )
    matched = lookup_summaries(table, chunk_hashes=["h1"], model="model-a")
    assert matched.hits == {0: "Summary under A."}
    different = lookup_summaries(table, chunk_hashes=["h1"], model="model-b")
    assert different.hits == {}
    assert different.misses_indices == [0]


def test_generate_chunk_context_returns_first_line(monkeypatch) -> None:
    """The Ollama response is post-processed: ``<think>`` blocks stripped,
    only the first non-empty line returned (some models emit a paragraph
    of commentary after the requested sentence)."""

    class FakeClient:
        def generate(self, **kwargs):
            assert kwargs["model"] == "qwen3:14b"
            assert kwargs["options"]["temperature"] == 0.0
            return {
                "response": (
                    "<think>let me plan</think>"
                    "This chunk discusses backward design.\n"
                    "Some extra commentary the indexer should not embed."
                )
            }

    monkeypatch.setattr(contextual_chunker, "get_ollama_client", lambda url, timeout: FakeClient())
    summary = generate_chunk_context(
        chunk_text="Backward design...",
        document_title="addie-model.md",
        document_summary="Five-phase ID framework.",
        config=_config(),
    )
    assert summary == "This chunk discusses backward design."


def test_generate_chunk_context_returns_empty_string_on_failure(monkeypatch) -> None:
    """An offline Ollama yields ``""`` so the pipeline can fall back to
    embedding the chunk without context (graceful degradation)."""

    class FailingClient:
        def generate(self, **kwargs):
            raise contextual_chunker.ollama.RequestError("connection refused")

    monkeypatch.setattr(
        contextual_chunker, "get_ollama_client", lambda url, timeout: FailingClient()
    )
    summary = generate_chunk_context(
        chunk_text="...",
        document_title="t.md",
        document_summary="",
        config=_config(),
    )
    assert summary == ""


def test_generate_chunk_context_handles_missing_document_summary(monkeypatch) -> None:
    """Empty document_summary must not crash the prompt builder."""
    captured_prompts: list[str] = []

    class CapturingClient:
        def generate(self, **kwargs):
            captured_prompts.append(kwargs["prompt"])
            return {"response": "OK."}

    monkeypatch.setattr(
        contextual_chunker, "get_ollama_client", lambda url, timeout: CapturingClient()
    )
    summary = generate_chunk_context(
        chunk_text="x",
        document_title="t.md",
        document_summary="",
        config=_config(),
    )
    assert summary == "OK."
    assert "(no summary available)" in captured_prompts[0]
