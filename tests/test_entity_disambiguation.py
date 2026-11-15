"""Tests for Stage 1 escalation and Stage 2 proper-noun disambiguation."""

from collections.abc import Sequence

from lxd.ingest.mentions import detect_mentions
from lxd.ontology.ambiguity import (
    escalated_surface_forms_with_candidates,
    requires_stage2_disambiguation,
)
from lxd.ontology.disambiguator import (
    DisambiguationCandidate,
    MentionDisambiguator,
    build_disambiguation_regex,
    context_window,
)
from lxd.ontology.matcher import LexiconRule, build_automaton


def _records() -> list[LexiconRule]:
    return [
        LexiconRule(
            surface_lower="id",
            curie="coe:C01010",
            surface="ID",
            pref_label="Adobe InDesign",
            collection_name="offerings",
            match_policy="case_exact",
            case_sensitive=True,
        ),
        LexiconRule(
            surface_lower="id",
            curie="coe:C01020",
            surface="ID",
            pref_label="Identity Service",
            collection_name="offerings",
            match_policy="case_exact",
            case_sensitive=True,
        ),
        LexiconRule(
            surface_lower="adobe",
            curie="coe:C00001",
            surface="Adobe",
            pref_label="Adobe",
            collection_name="organisations",
            match_policy="unambiguous",
        ),
        LexiconRule(
            surface_lower="target",
            curie="coe:C01008",
            surface="Target",
            pref_label="Adobe Target",
            collection_name="offerings",
            match_policy="context_anchored",
            case_sensitive=True,
            anchor_window_tokens=12,
            context_anchors=("Adobe",),
            requires_llm_disambiguation=True,
        ),
    ]


class _StubDisambiguator:
    def __init__(self, decision: str | None) -> None:
        self.decision = decision
        self.calls: list[tuple[str, list[str]]] = []

    def disambiguate(
        self,
        *,
        text: str,
        start: int,
        end: int,
        surface: str,
        candidates: Sequence[DisambiguationCandidate],
    ) -> str | None:
        assert text[start:end] == surface
        curies = [c.curie for c in candidates]
        self.calls.append((surface, curies))
        return self.decision


def test_escalated_map_excludes_unambiguous_terms() -> None:
    escalated = escalated_surface_forms_with_candidates(_records())

    assert "adobe" not in escalated
    assert escalated["id"] == ["coe:C01010", "coe:C01020"]
    assert escalated["target"] == ["coe:C01008"]


def test_requires_stage2_disambiguation_checks_flag_and_multi_curie() -> None:
    unambiguous = [
        LexiconRule(surface_lower="adobe", curie="coe:C00001", surface="Adobe"),
    ]
    flagged = [
        LexiconRule(
            surface_lower="target",
            curie="coe:C01008",
            surface="Target",
            requires_llm_disambiguation=True,
        ),
    ]
    assert not requires_stage2_disambiguation(unambiguous)
    assert requires_stage2_disambiguation(flagged)


def test_detect_mentions_calls_disambiguator_only_on_escalated_spans() -> None:
    automaton = build_automaton(_records())
    stub = _StubDisambiguator(decision="coe:C01010")
    assert isinstance(stub, MentionDisambiguator)

    text = "ID integrates with Adobe workflows."
    mentions = detect_mentions(text, automaton, disambiguator=stub)

    assert len(stub.calls) == 1
    assert stub.calls[0] == ("ID", ["coe:C01010", "coe:C01020"])
    by_surface = {m.surface_form: m.entity_id for m in mentions}
    assert by_surface == {"ID": "coe:C01010", "Adobe": "coe:C00001"}


def test_detect_mentions_abstains_when_disambiguator_returns_nil_or_none() -> None:
    automaton = build_automaton(_records())

    for abstention in ("NIL", None):
        stub = _StubDisambiguator(decision=abstention)
        mentions = detect_mentions("ID matters here alongside Adobe.", automaton, disambiguator=stub)
        assert [m.entity_id for m in mentions] == ["coe:C00001"]


def test_detect_mentions_without_disambiguator_skips_multi_curie_collision() -> None:
    automaton = build_automaton(_records())
    mentions = detect_mentions("ID and Adobe Target", automaton, disambiguator=None)
    # Multi-CURIE 'ID' abstains without Stage 2; longest span 'Adobe Target' isn't in _records(),
    # so 'Adobe' (unambiguous) and context-anchored single-CURIE 'Target' resolve cleanly.
    assert [m.entity_id for m in mentions] == ["coe:C00001", "coe:C01008"]


def test_build_disambiguation_regex_escapes_curies_and_appends_nil() -> None:
    assert build_disambiguation_regex(["coe:C01008", "coe:C01009"]) == r"^(coe:C01008|coe:C01009|NIL)$"


def test_context_window_clips_and_marks_exact_span() -> None:
    text = "abcdefghij"
    assert context_window(text, start=0, end=2, radius=20) == "<<ab>>cdefghij"
    assert context_window(text, start=4, end=6, radius=2) == "cd<<ef>>gh"
