"""Tests for the shared LLM client service."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lxd.ingest.llm_client import (
    build_cached_system_prompt,
    call_openai_async,
    call_with_fallback_async,
    prepare_batch_jsonl,
    reset_clients,
    run_concurrent_extraction,
)


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset client singletons before each test."""
    reset_clients()
    yield
    reset_clients()


# ---------------------------------------------------------------------------
# build_cached_system_prompt
# ---------------------------------------------------------------------------


def test_cached_prompt_without_vocabulary():
    """Base prompt returned unchanged when no vocabulary supplied."""
    result = build_cached_system_prompt("Base instructions.")
    assert result == "Base instructions."


def test_cached_prompt_with_entity_vocabulary():
    """Entity vocabulary appended as reference section."""
    result = build_cached_system_prompt("Base.", entity_vocabulary=["alpha", "beta", "gamma"])
    assert "--- REFERENCE: Known entity vocabulary (3 entities) ---" in result
    assert "  - alpha" in result
    assert "  - beta" in result
    assert "  - gamma" in result


def test_cached_prompt_with_predicate_vocabulary():
    """Predicate vocabulary appended as reference section."""
    result = build_cached_system_prompt("Base.", predicate_vocabulary=["relates_to", "teaches"])
    assert "--- REFERENCE: Valid predicate vocabulary (2 predicates) ---" in result
    assert "relates_to" in result
    assert "teaches" in result


def test_cached_prompt_with_both_vocabularies():
    """Both entity and predicate vocabularies appended."""
    result = build_cached_system_prompt(
        "Base.",
        entity_vocabulary=["entity_a"],
        predicate_vocabulary=["pred_x"],
    )
    assert "Known entity vocabulary" in result
    assert "Valid predicate vocabulary" in result


def test_cached_prompt_sorts_vocabulary():
    """Vocabulary is sorted for deterministic byte-identical prompts."""
    result = build_cached_system_prompt("Base.", entity_vocabulary=["zebra", "apple", "mango"])
    lines = result.split("\n")
    entity_lines = [line.strip() for line in lines if line.strip().startswith("- ")]
    assert entity_lines == ["- apple", "- mango", "- zebra"]


# ---------------------------------------------------------------------------
# call_openai_async
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_openai_async_returns_content():
    """Async OpenAI call returns message content."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"claims": []}'

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    with (
        patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
        patch("lxd.ingest.llm_client.get_async_openai_client", return_value=mock_client),
    ):
        result = await call_openai_async(
            system_prompt="System.",
            user_prompt="User.",
            model="gpt-4o-mini",
        )
        assert result == '{"claims": []}'


# ---------------------------------------------------------------------------
# call_with_fallback_async
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fallback_uses_ollama_when_openai_fails():
    """Fallback calls Ollama when OpenAI raises."""
    with (
        patch(
            "lxd.ingest.llm_client.call_openai_async",
            side_effect=RuntimeError("API down"),
        ),
        patch(
            "lxd.ingest.llm_client.call_ollama_sync_in_thread",
            return_value='{"claims": []}',
        ) as mock_ollama,
    ):
        result = await call_with_fallback_async(
            system_prompt="S",
            user_prompt="U",
            primary_backend="openai",
            openai_model="gpt-4o-mini",
            ollama_model="qwen3:14b",
        )
        assert result == '{"claims": []}'
        mock_ollama.assert_called_once()


@pytest.mark.asyncio
async def test_fallback_returns_empty_when_all_fail():
    """Returns empty string when both backends fail."""
    with (
        patch(
            "lxd.ingest.llm_client.call_openai_async",
            side_effect=RuntimeError("API down"),
        ),
        patch(
            "lxd.ingest.llm_client.call_ollama_sync_in_thread",
            side_effect=RuntimeError("Ollama down"),
        ),
    ):
        result = await call_with_fallback_async(
            system_prompt="S",
            user_prompt="U",
            primary_backend="openai",
            openai_model="gpt-4o-mini",
            ollama_model="qwen3:14b",
        )
        assert result == ""


@pytest.mark.asyncio
async def test_fallback_skips_ollama_when_fallback_is_none():
    """When fallback_backend='none', no Ollama call is made."""
    with (
        patch(
            "lxd.ingest.llm_client.call_openai_async",
            side_effect=RuntimeError("API down"),
        ),
        patch(
            "lxd.ingest.llm_client.call_ollama_sync_in_thread",
        ) as mock_ollama,
    ):
        result = await call_with_fallback_async(
            system_prompt="S",
            user_prompt="U",
            primary_backend="openai",
            openai_model="gpt-4o-mini",
            ollama_model="qwen3:14b",
            fallback_backend="none",
        )
        assert result == ""
        mock_ollama.assert_not_called()


# ---------------------------------------------------------------------------
# run_concurrent_extraction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_extraction_processes_all_items():
    """All items are processed and results returned."""
    items = [1, 2, 3, 4, 5]

    async def _extract(item: int) -> int:
        return item * 10

    results = await run_concurrent_extraction(items, _extract, max_concurrent=2, sub_batch_size=3)
    assert sorted(results) == [10, 20, 30, 40, 50]


@pytest.mark.asyncio
async def test_concurrent_extraction_calls_commit_fn_per_batch():
    """Commit function is called once per sub-batch."""
    items = list(range(10))
    committed: list[list[int]] = []

    async def _extract(item: int) -> int:
        return item

    def _commit(batch: list[int]) -> None:
        committed.append(batch)

    await run_concurrent_extraction(
        items, _extract, max_concurrent=5, sub_batch_size=4, commit_fn=_commit
    )
    assert len(committed) == 3  # 4 + 4 + 2
    assert sum(len(b) for b in committed) == 10


@pytest.mark.asyncio
async def test_concurrent_extraction_handles_exceptions_gracefully():
    """Failed items are logged but don't crash the batch."""
    items = [1, 2, 3]

    async def _extract(item: int) -> int:
        if item == 2:
            raise ValueError("Bad item")
        return item * 10

    results = await run_concurrent_extraction(items, _extract, max_concurrent=3, sub_batch_size=10)
    assert sorted(results) == [10, 30]


@pytest.mark.asyncio
async def test_concurrent_extraction_respects_semaphore():
    """No more than max_concurrent items are in flight at once."""
    max_in_flight = 0
    current_in_flight = 0
    lock = asyncio.Lock()

    items = list(range(20))

    async def _extract(item: int) -> int:
        nonlocal max_in_flight, current_in_flight
        async with lock:
            current_in_flight += 1
            if current_in_flight > max_in_flight:
                max_in_flight = current_in_flight
        await asyncio.sleep(0.01)
        async with lock:
            current_in_flight -= 1
        return item

    await run_concurrent_extraction(items, _extract, max_concurrent=3, sub_batch_size=20)
    assert max_in_flight <= 3


# ---------------------------------------------------------------------------
# prepare_batch_jsonl
# ---------------------------------------------------------------------------


def test_prepare_batch_jsonl_writes_correct_format(tmp_path: Path):
    """JSONL output has correct structure for OpenAI Batch API."""
    items: list[dict[str, Any]] = [
        {"custom_id": "chunk-001", "text": "Hello world"},
        {"custom_id": "chunk-002", "text": "Goodbye world"},
    ]

    def _build_messages(item: dict[str, Any]) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": "System prompt."},
            {"role": "user", "content": item["text"]},
        ]

    output = prepare_batch_jsonl(
        items,
        build_messages_fn=_build_messages,
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=1000,
        response_format={"type": "json_object"},
        output_path=tmp_path / "test.jsonl",
    )

    assert output.exists()
    lines = output.read_text().strip().split("\n")
    assert len(lines) == 2

    row = json.loads(lines[0])
    assert row["custom_id"] == "chunk-001"
    assert row["method"] == "POST"
    assert row["url"] == "/v1/chat/completions"
    assert row["body"]["model"] == "gpt-4o-mini"
    assert row["body"]["temperature"] == 0.0
    assert row["body"]["response_format"] == {"type": "json_object"}
    assert len(row["body"]["messages"]) == 2


def test_prepare_batch_jsonl_creates_parent_dirs(tmp_path: Path):
    """Output path parent directories are created if missing."""
    items: list[dict[str, Any]] = [{"custom_id": "x", "text": "t"}]
    output = prepare_batch_jsonl(
        items,
        build_messages_fn=lambda item: [{"role": "user", "content": item["text"]}],
        model="gpt-4o-mini",
        output_path=tmp_path / "nested" / "deep" / "batch.jsonl",
    )
    assert output.exists()
