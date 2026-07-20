"""Unit tests for the design/critique agents (Phase 3b/c SOTA roadmap).

All LLM calls and corpus retrieval are monkeypatched — these tests never
touch a network, Ollama, or a real SQLite/LanceDB store. They exercise the
step-budget circuit, the retrieval circuit breaker, per-step LLM-failure
degradation, and the artefact/citation shape.
"""

import json
from types import SimpleNamespace
from typing import Any, cast

import pytest

from lxd.agents import critique as critique_module
from lxd.agents import design as design_module
from lxd.agents.artefacts import DesignArtefactBundle
from lxd.domain.brief import LearnerBrief
from lxd.retrieval.query_pipeline import RankedChunk, SearchOutcome
from lxd.settings.models import DesignAgentConfig, RuntimeConfig

_LLM_RESPONSES: dict[str, dict[str, Any]] = {
    "objectives": {
        "objectives": ["Explain the ADDIE phases.", "Apply ADDIE to a new course."],
    },
    "modality_plan": {
        "modality_plan": "Use blended ILT because the audience needs hands-on practice.",
    },
    "outline": {"outline": ["Module 1: Overview", "Module 2: Practice"]},
    "assessment": {
        "assessment": ["Quiz mapping to objective 1", "Scenario mapping to objective 2"],
    },
    "critique": {
        "overall_score": 0.8,
        "dimension_scores": {"objective_alignment": 0.9, "evidence_grounding": 0.7},
        "feedback": ["Tighten the outline to match Bloom's target."],
    },
}


def _chunk(chunk_id: str, text: str = "evidence text") -> RankedChunk:
    return RankedChunk(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        citation_label=f"[{chunk_id}]",
        source_rel_path=f"{chunk_id}.md",
        source_filename=f"{chunk_id}.md",
        source_type="markdown",
        source_domain="guides",
        source_hash=f"hash-{chunk_id}",
        chunk_index=0,
        chunk_occurrence=0,
        token_count=10,
        text=text,
        score_hint=chunk_id,
        metadata_json="{}",
        score=1.0,
    )


def _outcome(ranked: list[RankedChunk], warnings: list[str] | None = None) -> SearchOutcome:
    return SearchOutcome(
        ranked=ranked,
        warnings=warnings or [],
        reranking_applied=False,
        expansion_applied=False,
        matched_entity_ids=[],
        expansion_terms=[],
        config_drift_warnings=[],
    )


def _config() -> RuntimeConfig:
    return cast("RuntimeConfig", SimpleNamespace(design_agent=DesignAgentConfig()))


async def _fake_call_with_fallback_async(*, system_prompt: str, user_prompt: str, **kwargs: Any) -> str:
    """Route by the literal JSON-shape marker each system prompt asks for.

    Using the exact ``{"key":`` substring (rather than a loose keyword like
    "objectives") avoids false matches — several prompts reference other
    steps' outputs by name (e.g. the outline prompt says "in service of
    the OBJECTIVES already drafted").
    """
    del user_prompt, kwargs
    if '{"overall_score":' in system_prompt:
        return json.dumps(_LLM_RESPONSES["critique"])
    if '{"objectives":' in system_prompt:
        return json.dumps(_LLM_RESPONSES["objectives"])
    if '{"modality_plan":' in system_prompt:
        return json.dumps(_LLM_RESPONSES["modality_plan"])
    if '{"outline":' in system_prompt:
        return json.dumps(_LLM_RESPONSES["outline"])
    if '{"assessment":' in system_prompt:
        return json.dumps(_LLM_RESPONSES["assessment"])
    raise AssertionError(f"unexpected system prompt: {system_prompt[:160]}")


@pytest.fixture(autouse=True)
def _patch_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(design_module, "call_with_fallback_async", _fake_call_with_fallback_async)
    monkeypatch.setattr(critique_module, "call_with_fallback_async", _fake_call_with_fallback_async)


def _patch_search_constant(monkeypatch: pytest.MonkeyPatch, outcome: SearchOutcome) -> None:
    def _search_chunks(
        question: str, config: object, domain: object = None, limit: object = None, route_breadth: object = None
    ) -> SearchOutcome:
        del question, config, domain, limit, route_breadth
        return outcome

    monkeypatch.setattr(design_module, "search_chunks", _search_chunks)
    monkeypatch.setattr(critique_module, "search_chunks", _search_chunks)


# ---------------------------------------------------------------------------
# design_learning
# ---------------------------------------------------------------------------


def test_design_learning_full_run_produces_grounded_bundle(monkeypatch: pytest.MonkeyPatch) -> None:
    ranked = [_chunk("addie-1"), _chunk("addie-2")]
    _patch_search_constant(monkeypatch, _outcome(ranked))

    bundle = design_module.design_learning("ADDIE model", LearnerBrief(), _config())

    assert bundle.steps_completed == 6
    assert bundle.objectives.items == _LLM_RESPONSES["objectives"]["objectives"]
    assert bundle.objectives.citations == ["[addie-1]", "[addie-2]"]
    assert bundle.modality_plan.text == _LLM_RESPONSES["modality_plan"]["modality_plan"]
    assert bundle.outline.items == _LLM_RESPONSES["outline"]["outline"]
    assert bundle.assessment.items == _LLM_RESPONSES["assessment"]["assessment"]
    # Critique+revise pass folds its score/feedback into warnings rather
    # than raising or silently discarding them.
    assert any("critique_overall_score=0.80" in w for w in bundle.warnings)
    assert any("Tighten the outline" in w for w in bundle.warnings)


def test_design_learning_retrieval_circuit_breaker_stops_before_drafting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_search_constant(monkeypatch, _outcome([]))

    def _fail_llm(**kwargs: Any) -> str:
        raise AssertionError("LLM must not be called once the retrieval circuit trips")

    monkeypatch.setattr(design_module, "call_with_fallback_async", _fail_llm)

    bundle = design_module.design_learning("obscure topic", LearnerBrief(), _config())

    assert bundle.steps_completed == 1
    assert bundle.objectives.items == []
    assert bundle.modality_plan.text == ""
    assert bundle.outline.items == []
    assert bundle.assessment.items == []
    assert any("no pedagogy evidence" in w.lower() for w in bundle.warnings)


def test_design_learning_respects_max_steps_override(monkeypatch: pytest.MonkeyPatch) -> None:
    ranked = [_chunk("addie-1")]
    _patch_search_constant(monkeypatch, _outcome(ranked))

    bundle = design_module.design_learning("ADDIE model", LearnerBrief(), _config(), max_steps=2)

    assert bundle.steps_completed == 2
    assert bundle.objectives.items == _LLM_RESPONSES["objectives"]["objectives"]
    assert bundle.modality_plan.text == ""
    assert bundle.outline.items == []
    assert bundle.assessment.items == []
    assert any("max_steps reached" in w for w in bundle.warnings)


def test_design_learning_llm_failure_on_one_step_degrades_gracefully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ranked = [_chunk("addie-1")]
    _patch_search_constant(monkeypatch, _outcome(ranked))

    async def _flaky(*, system_prompt: str, user_prompt: str, **kwargs: Any) -> str:
        if '{"modality_plan":' in system_prompt:
            raise RuntimeError("simulated LLM outage")
        return await _fake_call_with_fallback_async(
            system_prompt=system_prompt, user_prompt=user_prompt, **kwargs
        )

    monkeypatch.setattr(design_module, "call_with_fallback_async", _flaky)

    bundle = design_module.design_learning("ADDIE model", LearnerBrief(), _config())

    assert bundle.objectives.items, "unaffected earlier step still drafted"
    assert bundle.modality_plan.text == ""
    assert any("modality_plan drafting failed" in w for w in bundle.warnings)
    assert bundle.outline.items, "later steps still ran after one step's failure"
    assert bundle.assessment.items


def test_design_learning_folds_brief_into_retrieval_query(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_queries: list[str] = []

    def _search_chunks(
        question: str, config: object, domain: object = None, limit: object = None, route_breadth: object = None
    ) -> SearchOutcome:
        del config, domain, limit, route_breadth
        seen_queries.append(question)
        return _outcome([_chunk("c1")])

    monkeypatch.setattr(design_module, "search_chunks", _search_chunks)
    monkeypatch.setattr(critique_module, "search_chunks", _search_chunks)

    brief = LearnerBrief(audience="new admins", modality="ILT", bloom_target="apply")
    design_module.design_learning("ADDIE model", brief, _config())

    assert any("ILT" in q for q in seen_queries)
    assert any("apply" in q for q in seen_queries)


def test_design_learning_zero_max_steps_returns_empty_bundle_without_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _explode(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("neither retrieval nor LLM should run when max_steps=0")

    monkeypatch.setattr(design_module, "search_chunks", _explode)
    monkeypatch.setattr(design_module, "call_with_fallback_async", _explode)

    bundle = design_module.design_learning("ADDIE model", LearnerBrief(), _config(), max_steps=0)

    assert bundle.steps_completed == 0
    assert bundle.objectives.items == []
    assert bundle.warnings


# ---------------------------------------------------------------------------
# critique_design
# ---------------------------------------------------------------------------


def test_critique_design_parses_scores_and_feedback(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_search_constant(monkeypatch, _outcome([_chunk("c1")]))

    bundle = DesignArtefactBundle(topic="ADDIE model")
    result = critique_module.critique_design(bundle, "ADDIE model", _config())

    assert result.overall_score == 0.8
    assert result.dimension_scores["objective_alignment"] == 0.9
    assert result.feedback == _LLM_RESPONSES["critique"]["feedback"]
    assert result.citations == ["[c1]"]
    assert result.warnings == []


def test_critique_design_accepts_json_string_input(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_search_constant(monkeypatch, _outcome([_chunk("c1")]))

    payload = json.dumps({"topic": "ADDIE model"})
    result = critique_module.critique_design(payload, "", _config())

    assert result.overall_score == 0.8


def test_critique_design_handles_empty_artefact() -> None:
    result = critique_module.critique_design("", "", _config())
    assert result.overall_score == 0.0
    assert result.warnings


def test_critique_design_degrades_when_llm_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_search_constant(monkeypatch, _outcome([_chunk("c1")]))

    async def _boom(**kwargs: Any) -> str:
        del kwargs
        raise RuntimeError("offline")

    monkeypatch.setattr(critique_module, "call_with_fallback_async", _boom)

    result = critique_module.critique_design("free text artefact", "", _config())
    assert result.overall_score == 0.0
    assert any("Critique LLM call failed" in w for w in result.warnings)


def test_critique_design_flags_empty_retrieval(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_search_constant(monkeypatch, _outcome([]))

    result = critique_module.critique_design("free text artefact", "some topic", _config())

    assert result.citations == []
    assert any("no corpus evidence" in w.lower() for w in result.warnings)
