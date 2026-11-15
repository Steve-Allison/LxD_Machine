"""Validate two-stage proper-noun grounding against all 200 Central_Configs golden cases."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import llguidance.tiktoken
import pytest
import tiktoken
import yaml

from lxd.ingest.mentions import detect_mentions, extract_stage1_spans
from lxd.ontology.disambiguator import (
    DisambiguationCandidate,
    MentionDisambiguator,
    build_disambiguation_prompt,
    build_disambiguation_regex,
    build_entity_catalog,
    compile_llguidance_matcher,
)
from lxd.ontology.loader import load_ontology
from lxd.ontology.matcher import build_automaton

CENTRAL_CONFIGS_ROOT = Path(__file__).resolve().parents[2] / "Central_Configs"
GOLDEN_YAML_PATH = CENTRAL_CONFIGS_ROOT / "tests" / "golden_grounding.yaml"

pytestmark = pytest.mark.skipif(
    not (CENTRAL_CONFIGS_ROOT / "dist" / "build" / "central.duckdb").is_file()
    or not GOLDEN_YAML_PATH.is_file(),
    reason="Central_Configs v5.0.0 authority store or golden_grounding.yaml not present locally",
)


class _ProtocolAnchorDisambiguator:
    """Offline ``MentionDisambiguator`` protocol double for deterministic CI tests.

    Confirms Stage 1 context-anchored candidates when present in the candidate
    list, and supports explicit ``NIL`` overrides for negative-control assertions.
    """

    def __init__(self, *, force_nil_surfaces: frozenset[str] = frozenset()) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []
        self.force_nil_surfaces = force_nil_surfaces

    def disambiguate(
        self,
        *,
        text: str,
        start: int,
        end: int,
        surface: str,
        candidates: Sequence[DisambiguationCandidate],
    ) -> str | None:
        _ = (text, start, end)
        curies = tuple(c.curie for c in candidates)
        self.calls.append((surface, curies))
        if surface in self.force_nil_surfaces or not curies:
            return "NIL"
        return curies[0]


@pytest.fixture(scope="module")
def central_grounding_bundle() -> tuple[Any, dict[str, DisambiguationCandidate], list[dict[str, Any]]]:
    loaded = load_ontology(CENTRAL_CONFIGS_ROOT)
    automaton = build_automaton(loaded.matcher_records)
    catalog = build_entity_catalog(loaded.entity_definitions)
    with GOLDEN_YAML_PATH.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    cases = list(raw["cases"])
    return automaton, catalog, cases


def test_golden_grounding_200_cases_stage1_and_stage2(
    central_grounding_bundle: tuple[Any, dict[str, DisambiguationCandidate], list[dict[str, Any]]],
) -> None:
    automaton, catalog, cases = central_grounding_bundle
    assert len(cases) == 200

    disambiguator = _ProtocolAnchorDisambiguator()
    assert isinstance(disambiguator, MentionDisambiguator)

    deterministic_case_count = 0
    escalated_case_ids: list[str] = []

    for case in cases:
        case_id = str(case["id"])
        text = str(case["text"])
        expected_curies = set(case["expected_curies"])

        spans = extract_stage1_spans(text, automaton)
        case_escalated = any(span.requires_disambiguation for span in spans)
        if case_escalated:
            escalated_case_ids.append(case_id)
        else:
            deterministic_case_count += 1
            stage1_curies = {span.candidate_curies[0] for span in spans}
            assert stage1_curies == expected_curies, (
                f"{case_id} ({case['category']}): Stage 1 got {sorted(stage1_curies)}, "
                f"expected {sorted(expected_curies)}"
            )

        mentions = detect_mentions(
            text,
            automaton,
            disambiguator=disambiguator,
            entity_catalog=catalog,
        )
        for mention in mentions:
            assert text[mention.start_char : mention.end_char] == mention.surface_form
        resolved_curies = {mention.entity_id for mention in mentions}
        assert resolved_curies == expected_curies, (
            f"{case_id} ({case['category']}): Stage 1+2 got {sorted(resolved_curies)}, "
            f"expected {sorted(expected_curies)}"
        )

    assert deterministic_case_count == 190
    assert len(escalated_case_ids) == 10
    assert escalated_case_ids == [
        "GG-091",
        "GG-092",
        "GG-100",
        "GG-101",
        "GG-113",
        "GG-114",
        "GG-115",
        "GG-118",
        "GG-120",
        "GG-128",
    ]
    assert len(disambiguator.calls) == 10


def test_stage2_nil_decision_suppresses_escalated_homograph(
    central_grounding_bundle: tuple[Any, dict[str, DisambiguationCandidate], list[dict[str, Any]]],
) -> None:
    automaton, catalog, cases = central_grounding_bundle
    gg_091 = next(case for case in cases if case["id"] == "GG-091")
    disambiguator = _ProtocolAnchorDisambiguator(force_nil_surfaces=frozenset({"Target"}))

    mentions = detect_mentions(
        str(gg_091["text"]),
        automaton,
        disambiguator=disambiguator,
        entity_catalog=catalog,
    )
    resolved_curies = {m.entity_id for m in mentions}
    assert resolved_curies == {"coe:C00001"}


def test_llguidance_regex_grammar_accepts_only_candidates_or_nil() -> None:
    candidates = ["coe:C01008", "coe:C01009"]
    regex = build_disambiguation_regex(candidates)
    assert regex == r"^(coe:C01008|coe:C01009|NIL)$"

    enc = tiktoken.get_encoding("cl100k_base")
    ll_tok = llguidance.tiktoken.lltokenizer_from_encoding(enc)
    matcher = compile_llguidance_matcher(ll_tok, candidates)

    for valid in ("coe:C01008", "coe:C01009", "NIL"):
        matcher.reset()
        toks = enc.encode(valid)
        assert matcher.validate_tokens(toks) == len(toks)
        assert matcher.consume_tokens(toks)
        assert matcher.is_accepting()

    for invalid in ("Adobe Target", "coe:C99999", "nil", "coe:C01008 "):
        matcher.reset()
        toks = enc.encode(invalid)
        accepted = matcher.consume_tokens(toks) and matcher.is_accepting()
        assert not accepted, f"Grammar must reject {invalid!r}"

    prompt = build_disambiguation_prompt(
        text="Configure Adobe Target activities in Experience Cloud.",
        start=16,
        end=22,
        surface="Target",
        candidates=[
            DisambiguationCandidate(
                curie="coe:C01008",
                pref_label="Adobe Target",
                collection_name="offerings",
                definition="Personalisation and experimentation application.",
            )
        ],
    )
    assert "<<Target>>" in prompt
    assert "coe:C01008" in prompt
    assert "NIL" in prompt
