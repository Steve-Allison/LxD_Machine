from types import SimpleNamespace
from typing import Any, cast

import pytest

from lxd.domain.status import QueryAnswerStatus
from lxd.settings.models import RuntimeConfig
from lxd.synthesis import answering
from lxd.synthesis.answering import (
    AnswerEnvelope,
    StreamingTextDelta,
    stream_synthesize_answer,
)


def _config() -> RuntimeConfig:
    """SimpleNamespace stub typed as RuntimeConfig."""
    return cast(
        "RuntimeConfig",
        SimpleNamespace(
            models=SimpleNamespace(llm="test-llm", llm_no_think=False),
            synthesis=SimpleNamespace(temperature=0.1, max_tokens=100, timeout_secs=15),
        ),
    )


def test_synthesize_answer_returns_explicit_unavailable_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail(*args: Any, **kwargs: Any) -> str:
        raise answering.ChatError[0]("offline")

    monkeypatch.setattr(answering, "complete_chat", _fail)

    result = answering.synthesize_answer(
        "question",
        [answering.EvidenceChunk(citation_label="A", text="evidence", score=1.0)],
        _config(),
    )

    assert result.answer_status == QueryAnswerStatus.SYNTHESIS_UNAVAILABLE
    assert result.citations == ["A"]
    assert result.warnings == ["Synthesis model unavailable: offline"]


def test_stream_synthesize_answer_yields_deltas_then_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Happy path: stream emits deltas in order, then the final envelope
    with the joined-and-stripped answer text."""
    fragments = ["The ", "ADDIE ", "model has ", "five phases."]
    monkeypatch.setattr(answering, "stream_chat", lambda *args, **kwargs: iter(fragments))

    events = list(
        stream_synthesize_answer(
            "What is ADDIE?",
            [answering.EvidenceChunk(citation_label="A", text="ev", score=1.0)],
            _config(),
        )
    )
    deltas = [event for event in events if isinstance(event, StreamingTextDelta)]
    envelopes = [event for event in events if isinstance(event, AnswerEnvelope)]
    assert [delta.text for delta in deltas] == fragments
    assert len(envelopes) == 1
    assert envelopes[0].answer_status == QueryAnswerStatus.ANSWERED
    assert envelopes[0].answer_text == "The ADDIE model has five phases."
    assert envelopes[0].metadata == {"streamed": True}


def test_stream_synthesize_answer_strips_think_blocks_from_final(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reasoning models may emit `<think>...</think>` scratchpad."""
    fragments = ["<think>let me", " plan</think>", "The answer is X."]
    monkeypatch.setattr(answering, "stream_chat", lambda *args, **kwargs: iter(fragments))

    events = list(
        stream_synthesize_answer(
            "q",
            [answering.EvidenceChunk(citation_label="A", text="ev", score=1.0)],
            _config(),
        )
    )
    envelopes = [event for event in events if isinstance(event, AnswerEnvelope)]
    assert envelopes[0].answer_text == "The answer is X."


def test_stream_synthesize_answer_emits_unavailable_when_initial_call_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If initial call raises, the iterator yields one SYNTHESIS_UNAVAILABLE envelope."""

    def _fail(*args: Any, **kwargs: Any) -> Any:
        raise answering.ChatError[0]("offline")

    monkeypatch.setattr(answering, "stream_chat", _fail)

    events = list(
        stream_synthesize_answer(
            "q",
            [answering.EvidenceChunk(citation_label="A", text="ev", score=1.0)],
            _config(),
        )
    )
    assert len(events) == 1
    assert isinstance(events[0], AnswerEnvelope)
    assert events[0].answer_status == QueryAnswerStatus.SYNTHESIS_UNAVAILABLE
    assert events[0].warnings == ["Synthesis model unavailable: offline"]


def test_stream_synthesize_answer_emits_unavailable_when_stream_breaks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mid-stream errors yield a SYNTHESIS_UNAVAILABLE envelope after
    whatever deltas were already produced."""

    def _broken_iter(*args: Any, **kwargs: Any) -> Any:
        yield "prefix "
        raise answering.ChatError[0]("server reset")

    monkeypatch.setattr(answering, "stream_chat", _broken_iter)

    events = list(
        stream_synthesize_answer(
            "q",
            [answering.EvidenceChunk(citation_label="A", text="ev", score=1.0)],
            _config(),
        )
    )
    text_deltas = [event for event in events if isinstance(event, StreamingTextDelta)]
    envelopes = [event for event in events if isinstance(event, AnswerEnvelope)]
    assert [delta.text for delta in text_deltas] == ["prefix "]
    assert len(envelopes) == 1
    assert envelopes[0].answer_status == QueryAnswerStatus.SYNTHESIS_UNAVAILABLE
    assert "Synthesis stream interrupted" in envelopes[0].warnings[0]


def test_stream_synthesize_answer_emits_unavailable_when_response_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty stream → SYNTHESIS_UNAVAILABLE envelope."""
    monkeypatch.setattr(answering, "stream_chat", lambda *args, **kwargs: iter([]))

    events = list(
        stream_synthesize_answer(
            "q",
            [answering.EvidenceChunk(citation_label="A", text="ev", score=1.0)],
            _config(),
        )
    )
    assert events == [
        answering.synthesis_unavailable_answer(["A"], "Synthesis model returned an empty response.")
    ]
