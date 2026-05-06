"""End-to-end synthesis test against a real local Ollama (B-TEST-1).

Marked ``live`` so the default ``pixi run test`` does not run it. Run
explicitly with ``pixi run test-live`` (or ``pytest -m live``) when an
Ollama server matching ``config.yaml`` is reachable on localhost.

Why this test exists: every other test in the suite mocks the LLM
boundary. None of them would catch a regression in
:func:`lxd.synthesis.answering.synthesize_answer` that depended on the
actual model ignoring the preamble, or rendering the prompt in a
broken shape, or returning a no-text response.

What this test verifies:

* The synthesis prompt is wired correctly to the configured local
  Ollama instance.
* The answer envelope reaches ``ANSWERED`` status (not
  ``synthesis_unavailable``).
* The answer text references the entity name from the seeded
  evidence chunk — the simplest possible "did the model read the
  evidence?" assertion.
"""

from __future__ import annotations

import socket
from urllib.parse import urlparse

import pytest

from lxd.domain.status import QueryAnswerStatus
from lxd.settings.loader import load_runtime_config, resolve_repo_root
from lxd.synthesis.answering import EvidenceChunk, synthesize_answer

pytestmark = [pytest.mark.live, pytest.mark.integration]


def _ollama_reachable(url: str) -> bool:
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 11434
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


def test_synthesize_against_real_local_ollama() -> None:
    repo_root = resolve_repo_root()
    config, _ = load_runtime_config(repo_root)

    if not _ollama_reachable(str(config.ollama.url)):
        pytest.skip(f"Ollama not reachable at {config.ollama.url}; live test skipped.")

    evidence = [
        EvidenceChunk(
            citation_label="test/learning_objectives.md#1",
            text=(
                "Bloom's taxonomy organises cognitive learning objectives into six "
                "levels: remember, understand, apply, analyse, evaluate, and create. "
                "It is the foundational framework instructional designers use to "
                "specify what a learner should be able to do after instruction."
            ),
            score=1.0,
            cited_sources=(),
        )
    ]

    envelope = synthesize_answer(
        question="What are the six levels of Bloom's taxonomy?",
        evidence=evidence,
        config=config,
    )

    assert envelope.answer_status == QueryAnswerStatus.ANSWERED, (
        f"Expected ANSWERED status; saw {envelope.answer_status} with "
        f"text={envelope.answer_text!r} warnings={envelope.warnings}"
    )
    answer_lower = envelope.answer_text.lower()
    assert "bloom" in answer_lower or "remember" in answer_lower or "create" in answer_lower, (
        f"Expected the answer to reference Bloom's taxonomy or one of its levels; "
        f"saw: {envelope.answer_text!r}"
    )
    assert envelope.citations == ["test/learning_objectives.md#1"], (
        f"Citation list should mirror the seeded evidence; saw {envelope.citations}"
    )
