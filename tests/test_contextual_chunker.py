"""Tests for the contextual retrieval chunker.

The chunker has three integration points exercised here:

* ``augment_chunk_for_embedding`` — pure function, no I/O.
* ``generate_chunk_context`` — calls MLX; we monkeypatch the
  chat function so the test runs without a live model.
* ``lookup_summaries`` / ``store_summaries`` — round-trip through a
  real LanceDB table (project no-mocks rule for internal stores).
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import lancedb
import pytest

from lxd.ingest import contextual_chunker
from lxd.ingest.contextual_chunker import (
    augment_chunk_for_embedding,
    generate_chunk_context,
    lookup_summaries,
    open_summary_cache_table,
    store_summaries,
)
from lxd.settings.models import RuntimeConfig


def _config() -> RuntimeConfig:
    return cast(
        "RuntimeConfig",
        SimpleNamespace(
            chunking=SimpleNamespace(
                contextual_summary_enabled=True,
                contextual_summary_model="qwen3:14b",
                contextual_summary_temperature=0.0,
                contextual_summary_timeout_secs=60,
                contextual_summary_max_tokens=80,
            ),
            models=SimpleNamespace(llm_no_think=True),
        ),
    )


def test_augment_chunk_for_embedding_prepends_summary() -> None:
    augmented = augment_chunk_for_embedding(
        "Backward design starts from the desired learning outcome.",
        "This chunk introduces backward design within the ADDIE methodology.",
    )
    assert augmented.startswith(
        "This chunk introduces backward design within the ADDIE methodology."
    )
    assert augmented.endswith("Backward design starts from the desired learning outcome.")
    assert "\n\n" in augmented


def test_augment_chunk_for_embedding_noop_on_empty_summary() -> None:
    original = "Some text."
    assert augment_chunk_for_embedding(original, "") == original


def test_store_and_lookup_summaries_round_trip(tmp_path: Path) -> None:
    db = lancedb.connect(str(tmp_path / "lancedb"))
    table = open_summary_cache_table(db)

    stored = store_summaries(
        table,
        chunk_hashes=["hash-1", "hash-2"],
        summaries=["Summary 1.", "Summary 2."],
        model="model-a",
    )
    assert stored == 2

    # Look up by hash list
    result = lookup_summaries(table, chunk_hashes=["hash-1", "hash-missing"], model="model-a")
    assert result.hits == {0: "Summary 1."}
    assert result.misses_indices == [1]


def test_store_summaries_empty_list_is_noop(tmp_path: Path) -> None:
    db = lancedb.connect(str(tmp_path / "lancedb"))
    table = open_summary_cache_table(db)
    assert store_summaries(table, chunk_hashes=[], summaries=[], model="model-a") == 0


def test_lookup_summaries_empty_list_returns_empty_hits(tmp_path: Path) -> None:
    db = lancedb.connect(str(tmp_path / "lancedb"))
    table = open_summary_cache_table(db)
    result = lookup_summaries(table, chunk_hashes=[], model="model-a")
    assert result.hits == {}
    assert result.misses_indices == []


def test_lookup_summaries_isolates_by_model(tmp_path: Path) -> None:
    db = lancedb.connect(str(tmp_path / "lancedb"))
    table = open_summary_cache_table(db)
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


def test_generate_chunk_context_returns_first_line(monkeypatch: pytest.MonkeyPatch) -> None:
    """The MLX response is post-processed: ``<think>`` blocks stripped,
    only the first non-empty line returned."""

    def _complete(*args: Any, **kwargs: Any) -> str:
        return (
            "<think>let me plan</think>"
            "This chunk discusses backward design.\n"
            "Some extra commentary the indexer should not embed."
        )

    monkeypatch.setattr(contextual_chunker, "complete_chat", _complete)
    summary = generate_chunk_context(
        chunk_text="Backward design...",
        document_title="addie-model.md",
        document_summary="Five-phase ID framework.",
        config=_config(),
    )
    assert summary == "This chunk discusses backward design."


def test_generate_chunk_context_returns_empty_string_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An offline MLX yields ``""`` so the pipeline can fall back to
    embedding the chunk without context (graceful degradation)."""

    def _fail(*args: Any, **kwargs: Any) -> str:
        raise contextual_chunker.ChatError[0]("connection refused")

    monkeypatch.setattr(contextual_chunker, "complete_chat", _fail)
    summary = generate_chunk_context(
        chunk_text="...",
        document_title="t.md",
        document_summary="",
        config=_config(),
    )
    assert summary == ""


def test_generate_chunk_context_handles_missing_document_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty document_summary must not crash the prompt builder."""
    captured_prompts: list[str] = []

    def _capture(*args: Any, **kwargs: Any) -> str:
        captured_prompts.append(args[1] if len(args) > 1 else kwargs.get("prompt", ""))
        return "OK."

    monkeypatch.setattr(contextual_chunker, "complete_chat", _capture)
    summary = generate_chunk_context(
        chunk_text="x",
        document_title="t.md",
        document_summary="",
        config=_config(),
    )
    assert summary == "OK."
    assert "(no summary available)" in captured_prompts[0]
