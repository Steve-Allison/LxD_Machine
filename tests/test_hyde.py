"""Tests for HyDE (Hypothetical Document Embeddings) query rewriting."""

from __future__ import annotations

from types import SimpleNamespace

from lxd.retrieval import hyde
from lxd.retrieval.hyde import generate_hypothetical_answer


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        retrieval=SimpleNamespace(
            hyde_enabled=True,
            hyde_model="qwen3:14b",
            hyde_temperature=0.0,
            hyde_timeout_secs=30,
            hyde_max_tokens=200,
        ),
        models=SimpleNamespace(llm_no_think=True),
        ollama=SimpleNamespace(url="http://localhost:11434"),
    )


def test_generate_hypothetical_answer_returns_model_response(monkeypatch) -> None:
    """Happy path: the trimmed model response is returned verbatim."""

    class FakeClient:
        def generate(self, **kwargs):
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

    monkeypatch.setattr(hyde, "get_ollama_client", lambda url, timeout: FakeClient())
    answer = generate_hypothetical_answer("What is backward design?", _config())
    assert "Backward design" in answer
    assert "outcome" in answer


def test_generate_hypothetical_answer_strips_think_blocks(monkeypatch) -> None:
    """Reasoning models may emit `<think>...</think>`. The HyDE output
    feeds the embedder — the think text would pollute the query
    embedding."""

    class ThinkingClient:
        def generate(self, **kwargs):
            return {
                "response": (
                    "<think>let me think about backward design</think>"
                    "Backward design begins with the outcome."
                )
            }

    monkeypatch.setattr(hyde, "get_ollama_client", lambda url, timeout: ThinkingClient())
    answer = generate_hypothetical_answer("...", _config())
    assert answer == "Backward design begins with the outcome."


def test_generate_hypothetical_answer_returns_empty_on_offline_ollama(monkeypatch) -> None:
    """Offline Ollama yields "" so the caller falls back to embedding
    the literal question — HyDE becomes a no-op rather than breaking
    retrieval."""

    class FailingClient:
        def generate(self, **kwargs):
            raise hyde.ollama.RequestError("connection refused")

    monkeypatch.setattr(hyde, "get_ollama_client", lambda url, timeout: FailingClient())
    answer = generate_hypothetical_answer("What is X?", _config())
    assert answer == ""


def test_generate_hypothetical_answer_returns_empty_on_blank_question() -> None:
    """A whitespace-only question short-circuits without calling Ollama."""
    answer = generate_hypothetical_answer("   ", _config())
    assert answer == ""


def test_generate_hypothetical_answer_returns_empty_on_response_error(monkeypatch) -> None:
    class BadClient:
        def generate(self, **kwargs):
            raise hyde.ollama.ResponseError("model unavailable")

    monkeypatch.setattr(hyde, "get_ollama_client", lambda url, timeout: BadClient())
    answer = generate_hypothetical_answer("What is Y?", _config())
    assert answer == ""
