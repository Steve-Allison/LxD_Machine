"""Tests for HyDE (Hypothetical Document Embeddings) query rewriting."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from lxd.retrieval import hyde
from lxd.retrieval.hyde import generate_hypothetical_answer
from lxd.settings.models import RuntimeConfig


def _config() -> RuntimeConfig:
    return cast(
        "RuntimeConfig",
        SimpleNamespace(
            retrieval=SimpleNamespace(
                hyde_enabled=True,
                hyde_model="qwen3:14b",
                hyde_temperature=0.0,
                hyde_timeout_secs=30,
                hyde_max_tokens=200,
            ),
            models=SimpleNamespace(llm_no_think=True),
            ollama=SimpleNamespace(url="http://localhost:11434"),
        ),
    )


def _install_ollama_client(monkeypatch: pytest.MonkeyPatch, client: object) -> None:
    def _factory(_url: str, _timeout: float) -> object:
        return client

    monkeypatch.setattr(hyde, "get_ollama_client", _factory)


def test_generate_hypothetical_answer_returns_model_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Happy path: the trimmed model response is returned verbatim."""

    class FakeClient:
        def generate(self, **kwargs: Any) -> dict[str, str]:
            assert kwargs["model"] == "qwen3:14b"
            assert kwargs["options"]["num_predict"] == 200
            assert "What is backward design?" in kwargs["prompt"]
            return {
                "response": (
                    "Backward design is a curriculum framework that begins "
                    "with the desired learning outcome and works backward to "
                    "design assessments and instruction."
                )
            }

    _install_ollama_client(monkeypatch, FakeClient())
    answer = generate_hypothetical_answer("What is backward design?", _config())
    assert "Backward design" in answer
    assert "outcome" in answer


def test_generate_hypothetical_answer_strips_think_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reasoning models may emit `<think>...</think>`. The HyDE output
    feeds the embedder — the think text would pollute the query
    embedding."""

    class ThinkingClient:
        def generate(self, **kwargs: Any) -> dict[str, str]:
            del kwargs
            return {
                "response": (
                    "<think>let me think about backward design</think>"
                    "Backward design begins with the outcome."
                )
            }

    _install_ollama_client(monkeypatch, ThinkingClient())
    answer = generate_hypothetical_answer("...", _config())
    assert answer == "Backward design begins with the outcome."


def test_generate_hypothetical_answer_returns_empty_on_offline_ollama(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Offline Ollama yields "" so the caller falls back to embedding
    the literal question — HyDE becomes a no-op rather than breaking
    retrieval."""

    class FailingClient:
        def generate(self, **kwargs: Any) -> dict[str, str]:
            del kwargs
            raise hyde.ollama.RequestError("connection refused")

    _install_ollama_client(monkeypatch, FailingClient())
    answer = generate_hypothetical_answer("What is X?", _config())
    assert answer == ""


def test_generate_hypothetical_answer_returns_empty_on_blank_question() -> None:
    """A whitespace-only question short-circuits without calling Ollama."""
    answer = generate_hypothetical_answer("   ", _config())
    assert answer == ""


def test_generate_hypothetical_answer_returns_empty_on_response_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BadClient:
        def generate(self, **kwargs: Any) -> dict[str, str]:
            del kwargs
            raise hyde.ollama.ResponseError("model unavailable")

    _install_ollama_client(monkeypatch, BadClient())
    answer = generate_hypothetical_answer("What is Y?", _config())
    assert answer == ""
