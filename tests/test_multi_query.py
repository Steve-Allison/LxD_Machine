"""Tests for LLM-generated multi-query paraphrase expansion.

Mirrors the failure-safe contract established by ``test_hyde.py``: the
LLM call is monkeypatched, and every parse/dedupe edge case is pinned
without touching the network.
"""

from typing import Any

import pytest

from lxd.retrieval import multi_query as _multi_query_module
from lxd.retrieval.multi_query import generate_query_paraphrases
from lxd.settings.models import RetrievalConfig

pytestmark = [pytest.mark.unit]


def _config(**overrides: Any) -> RetrievalConfig:
    defaults: dict[str, Any] = {"dense_top_k": 20, "rerank_top_k": 20}
    defaults.update(overrides)
    return RetrievalConfig(**defaults)


def _install_fake_call(monkeypatch: pytest.MonkeyPatch, raw: str | Exception) -> None:
    async def _fake_call_openai_async(**_kwargs: Any) -> str:
        if isinstance(raw, Exception):
            raise raw
        return raw

    monkeypatch.setattr(_multi_query_module, "call_openai_async", _fake_call_openai_async)


def test_generate_query_paraphrases_returns_parsed_list(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = (
        '{"paraphrases": ["How do you design backward from outcomes?", '
        '"What is outcome-first curriculum design?"]}'
    )
    _install_fake_call(monkeypatch, raw)
    result = generate_query_paraphrases("What is backward design?", _config(multi_query_count=2))
    assert result == [
        "How do you design backward from outcomes?",
        "What is outcome-first curriculum design?",
    ]


def test_generate_query_paraphrases_dedupes_against_original_casefold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = '{"paraphrases": ["WHAT IS BACKWARD DESIGN?", "How do you plan from outcomes?"]}'
    _install_fake_call(monkeypatch, raw)
    result = generate_query_paraphrases("What is backward design?", _config(multi_query_count=2))
    assert result == ["How do you plan from outcomes?"]


def test_generate_query_paraphrases_drops_empty_strings(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = '{"paraphrases": ["", "   ", "How do you plan from outcomes?"]}'
    _install_fake_call(monkeypatch, raw)
    result = generate_query_paraphrases("What is backward design?", _config(multi_query_count=3))
    assert result == ["How do you plan from outcomes?"]


def test_generate_query_paraphrases_dedupes_within_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = '{"paraphrases": ["How do you plan from outcomes?", "how do you plan from outcomes?"]}'
    _install_fake_call(monkeypatch, raw)
    result = generate_query_paraphrases("What is backward design?", _config(multi_query_count=2))
    assert result == ["How do you plan from outcomes?"]


def test_generate_query_paraphrases_truncates_to_multi_query_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = '{"paraphrases": ["a", "b", "c", "d"]}'
    _install_fake_call(monkeypatch, raw)
    result = generate_query_paraphrases("original question", _config(multi_query_count=2))
    assert result == ["a", "b"]


def test_generate_query_paraphrases_returns_empty_on_malformed_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_call(monkeypatch, "not valid json")
    result = generate_query_paraphrases("What is backward design?", _config())
    assert result == []


def test_generate_query_paraphrases_returns_empty_when_key_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_call(monkeypatch, '{"other_key": []}')
    result = generate_query_paraphrases("What is backward design?", _config())
    assert result == []


def test_generate_query_paraphrases_returns_empty_on_llm_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_call(monkeypatch, RuntimeError("timeout"))
    result = generate_query_paraphrases("What is backward design?", _config())
    assert result == []


def test_generate_query_paraphrases_returns_empty_on_blank_question() -> None:
    result = generate_query_paraphrases("   ", _config())
    assert result == []


def test_generate_query_paraphrases_ignores_non_string_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = '{"paraphrases": [1, null, "How do you plan from outcomes?"]}'
    _install_fake_call(monkeypatch, raw)
    result = generate_query_paraphrases("What is backward design?", _config(multi_query_count=3))
    assert result == ["How do you plan from outcomes?"]


def test_generate_query_paraphrases_returns_empty_when_payload_not_a_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = '{"paraphrases": "not a list"}'
    _install_fake_call(monkeypatch, raw)
    result = generate_query_paraphrases("What is backward design?", _config())
    assert result == []
