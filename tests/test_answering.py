from __future__ import annotations

from types import SimpleNamespace

from lxd.domain.status import QueryAnswerStatus
from lxd.synthesis import answering


def test_synthesize_answer_returns_explicit_unavailable_status(monkeypatch) -> None:
    class FailingClient:
        def generate(self, **kwargs):
            raise answering.ollama.RequestError("offline")

    monkeypatch.setattr(answering, "_client", lambda config: FailingClient())

    config = SimpleNamespace(
        models=SimpleNamespace(llm="test-llm", llm_no_think=False),
        synthesis=SimpleNamespace(temperature=0.1, max_tokens=100, timeout_secs=15),
        ollama=SimpleNamespace(url="http://localhost:11434"),
    )

    result = answering.synthesize_answer(
        "question",
        [answering.EvidenceChunk(citation_label="A", text="evidence", score=1.0)],
        config,
    )

    assert result.answer_status == QueryAnswerStatus.SYNTHESIS_UNAVAILABLE
    assert result.citations == ["A"]
    assert result.warnings == ["Synthesis model unavailable: offline"]
