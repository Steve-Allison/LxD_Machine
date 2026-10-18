"""Tests for HyDE (Hypothetical Document Embeddings) query rewriting."""

from types import SimpleNamespace
from typing import cast

import pytest

from lxd.mlx.chat import ChatError
from lxd.retrieval import hyde
from lxd.retrieval.hyde import generate_hypothetical_answer
from lxd.settings.models import RuntimeConfig


def _config() -> RuntimeConfig:
    return cast(
        "RuntimeConfig",
        SimpleNamespace(
            retrieval=SimpleNamespace(
                hyde_enabled=True,
                hyde_model="mlx-community/Qwen2.5-7B-Instruct-4bit",
                hyde_temperature=0.0,
                hyde_timeout_secs=30,
                hyde_max_tokens=200,
            ),
            models=SimpleNamespace(llm_no_think=True),
        ),
    )


def test_generate_hypothetical_answer_returns_model_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Happy path: the trimmed model response is returned verbatim."""

    def _mock_complete_chat(config: RuntimeConfig, prompt: str, **kwargs: object) -> str:
        del config
        assert kwargs.get("model") == "mlx-community/Qwen2.5-7B-Instruct-4bit"
        assert kwargs.get("max_tokens") == 200
        assert "What is backward design?" in prompt
        return (
            "Backward design is a curriculum framework that begins "
            "with the desired learning outcome and works backward to "
            "design assessments and instruction."
        )

    monkeypatch.setattr(hyde, "complete_chat", _mock_complete_chat)
    answer = generate_hypothetical_answer("What is backward design?", _config())
    assert "Backward design" in answer
    assert "outcome" in answer


def test_generate_hypothetical_answer_strips_think_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reasoning models may emit `<think>...</think>`. The HyDE output
    feeds the embedder — the think text would pollute the query
    embedding."""

    def _mock_complete_chat(config: RuntimeConfig, prompt: str, **kwargs: object) -> str:
        del config, prompt, kwargs
        return (
            "<think>let me think about backward design</think>"
            "Backward design begins with the outcome."
        )

    monkeypatch.setattr(hyde, "complete_chat", _mock_complete_chat)
    answer = generate_hypothetical_answer("...", _config())
    assert answer == "Backward design begins with the outcome."


def test_generate_hypothetical_answer_returns_empty_on_offline_mlx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Offline MLX yields "" so the caller falls back to embedding
    the literal question — HyDE becomes a no-op rather than breaking
    retrieval."""

    def _mock_complete_chat(config: RuntimeConfig, prompt: str, **kwargs: object) -> str:
        del config, prompt, kwargs
        raise RuntimeError("connection refused")

    monkeypatch.setattr(hyde, "complete_chat", _mock_complete_chat)
    answer = generate_hypothetical_answer("What is X?", _config())
    assert answer == ""


def test_generate_hypothetical_answer_returns_empty_on_blank_question() -> None:
    """A whitespace-only question short-circuits without calling MLX."""
    answer = generate_hypothetical_answer("   ", _config())
    assert answer == ""


def test_generate_hypothetical_answer_returns_empty_on_chat_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _mock_complete_chat(config: RuntimeConfig, prompt: str, **kwargs: object) -> str:
        del config, prompt, kwargs
        raise ChatError[0]("model unavailable")

    monkeypatch.setattr(hyde, "complete_chat", _mock_complete_chat)
    answer = generate_hypothetical_answer("What is Y?", _config())
    assert answer == ""
