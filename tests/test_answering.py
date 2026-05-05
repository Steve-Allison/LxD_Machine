from __future__ import annotations

from types import SimpleNamespace

from lxd.domain.status import QueryAnswerStatus
from lxd.synthesis import answering
from lxd.synthesis.answering import (
    AnswerEnvelope,
    StreamingTextDelta,
    stream_synthesize_answer,
)


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        models=SimpleNamespace(llm="test-llm", llm_no_think=False),
        synthesis=SimpleNamespace(temperature=0.1, max_tokens=100, timeout_secs=15),
        ollama=SimpleNamespace(url="http://localhost:11434"),
    )


def test_synthesize_answer_returns_explicit_unavailable_status(monkeypatch) -> None:
    class FailingClient:
        def generate(self, **kwargs):
            raise answering.ollama.RequestError("offline")

    monkeypatch.setattr(answering, "_client", lambda config: FailingClient())

    result = answering.synthesize_answer(
        "question",
        [answering.EvidenceChunk(citation_label="A", text="evidence", score=1.0)],
        _config(),
    )

    assert result.answer_status == QueryAnswerStatus.SYNTHESIS_UNAVAILABLE
    assert result.citations == ["A"]
    assert result.warnings == ["Synthesis model unavailable: offline"]


def test_stream_synthesize_answer_yields_deltas_then_envelope(monkeypatch) -> None:
    """Happy path: stream emits deltas in order, then the final envelope
    with the joined-and-stripped answer text."""
    fragments = ["The ", "ADDIE ", "model has ", "five phases."]

    class StreamingClient:
        def generate(self, **kwargs):
            assert kwargs["stream"] is True
            return iter(SimpleNamespace(response=fragment) for fragment in fragments)

    monkeypatch.setattr(answering, "_client", lambda config: StreamingClient())

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


def test_stream_synthesize_answer_strips_think_blocks_from_final(monkeypatch) -> None:
    """Reasoning models may emit `<think>...</think>` scratchpad. The
    streaming path lets clients see the raw deltas (so progress UI is
    honest about what the model is doing) but the terminal envelope
    carries clean prose with think blocks stripped."""
    fragments = ["<think>let me", " plan</think>", "The answer is X."]

    class StreamingClient:
        def generate(self, **kwargs):
            return iter(SimpleNamespace(response=fragment) for fragment in fragments)

    monkeypatch.setattr(answering, "_client", lambda config: StreamingClient())

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
    monkeypatch,
) -> None:
    """If `generate(stream=True)` raises, the iterator yields one
    SYNTHESIS_UNAVAILABLE envelope and stops — no half-state."""

    class FailingClient:
        def generate(self, **kwargs):
            raise answering.ollama.RequestError("offline")

    monkeypatch.setattr(answering, "_client", lambda config: FailingClient())

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


def test_stream_synthesize_answer_emits_unavailable_when_stream_breaks(monkeypatch) -> None:
    """Mid-stream errors yield a SYNTHESIS_UNAVAILABLE envelope after
    whatever deltas were already produced."""

    def _broken_iter():
        yield SimpleNamespace(response="prefix ")
        raise answering.ollama.ResponseError("server reset")

    class StreamingClient:
        def generate(self, **kwargs):
            return _broken_iter()

    monkeypatch.setattr(answering, "_client", lambda config: StreamingClient())

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


def test_stream_synthesize_answer_emits_unavailable_when_response_is_empty(monkeypatch) -> None:
    """Empty stream → SYNTHESIS_UNAVAILABLE envelope (matches the non-streaming behaviour)."""

    class StreamingClient:
        def generate(self, **kwargs):
            return iter([])

    monkeypatch.setattr(answering, "_client", lambda config: StreamingClient())

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
