"""Two-stage non-mutating proper-noun mention grounding in raw source text."""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import ahocorasick

from lxd.ontology.ambiguity import requires_stage2_disambiguation
from lxd.ontology.disambiguator import DisambiguationCandidate, MentionDisambiguator
from lxd.ontology.matcher import LexiconRule, offset_preserving_lower

_TOKEN_RE = re.compile(r"[A-Za-z0-9_.+-]+")
_NEGATIVE_WINDOW_TOKENS = 25


@dataclass(frozen=True, slots=True)
class Mention:
    """Detected proper-noun mention span aligned to unmodified ``text``."""

    entity_id: str
    term_source: str
    surface_form: str
    start_char: int
    end_char: int


@dataclass(frozen=True, slots=True)
class Stage1Span:
    """Accepted non-overlapping Stage 1 proper-noun candidate span."""

    start_char: int
    end_char: int
    surface_form: str
    rules: tuple[LexiconRule, ...]
    candidate_curies: tuple[str, ...]
    requires_disambiguation: bool


def extract_stage1_spans(text: str, automaton: Any) -> list[Stage1Span]:
    """Run deterministic Stage 1 proper-noun filtering on unmodified ``text``.

    Enforces all five ``Central_Configs`` v5.0.0 Stage 1 rules:
      1. Alphanumeric boundary check, dotted package suffix rejection, and
         hyphenated prefix rejection.
      2. Exact case gate when ``case_sensitive`` or ``match_policy == 'case_exact'``.
      3. ``negative_phrases`` veto across ``±25`` tokens inclusive of span.
      4. ``context_anchored`` gate across ``±anchor_window_tokens`` strictly
         excluding the matched span itself.
      5. Leftmost-longest non-overlapping span selection.
    """
    if not text or getattr(automaton, "kind", ahocorasick.AHOCORASICK) != ahocorasick.AHOCORASICK:
        return []

    tokens = [(match.start(), match.end()) for match in _TOKEN_RE.finditer(text)]
    by_span: dict[tuple[int, int], list[LexiconRule]] = {}
    lowered = offset_preserving_lower(text)
    text_len = len(text)

    for end_inclusive, rule_group in automaton.iter(lowered):
        if not rule_group:
            continue
        first_rule: LexiconRule = rule_group[0]
        end = end_inclusive + 1
        start = end - len(first_rule.surface)
        if start < 0 or end > text_len:
            continue

        # Rule 1: Boundary check + dotted suffix + hyphenated prefix rejections.
        if start > 0 and text[start - 1].isalnum():
            continue
        if end < text_len and text[end].isalnum():
            continue
        if end + 1 < text_len and text[end] == "." and text[end + 1].isalnum():
            continue
        if start >= 2 and text[start - 1] == "-" and text[start - 2].isalnum():
            continue

        first_tok, last_tok = _overlapping_token_bounds(tokens, start=start, end=end)
        raw_surface = text[start:end]

        for rule in rule_group:
            # Rule 2: Case gate.
            if (
                rule.case_sensitive or rule.match_policy == "case_exact"
            ) and raw_surface != rule.surface:
                continue

            # Rule 3: negative_phrases veto across ±25 tokens INCLUSIVE of span.
            if rule.negative_phrases and tokens and first_tok <= last_tok:
                win_first = max(0, first_tok - _NEGATIVE_WINDOW_TOKENS)
                win_last = min(len(tokens) - 1, last_tok + _NEGATIVE_WINDOW_TOKENS)
                win_slice = text[tokens[win_first][0] : tokens[win_last][1]]
                if any(_contains_boundary_phrase(win_slice, neg) for neg in rule.negative_phrases):
                    continue

            # Rule 4: context_anchored gate across ±anchor_window_tokens EXCLUDING span.
            if rule.match_policy == "context_anchored" or rule.context_anchors:
                if not rule.context_anchors or not tokens or first_tok > last_tok:
                    continue
                window = max(1, rule.anchor_window_tokens)
                left_tokens = tokens[max(0, first_tok - window) : first_tok]
                right_tokens = tokens[last_tok + 1 : min(len(tokens), last_tok + 1 + window)]
                left_ctx = text[left_tokens[0][0] : left_tokens[-1][1]] if left_tokens else ""
                right_ctx = text[right_tokens[0][0] : right_tokens[-1][1]] if right_tokens else ""
                ctx = f"{left_ctx} {right_ctx}"
                if not any(_contains_boundary_phrase(ctx, anchor) for anchor in rule.context_anchors):
                    continue

            by_span.setdefault((start, end), []).append(rule)

    # Rule 5: Leftmost-longest non-overlapping span selection.
    accepted: list[Stage1Span] = []
    last_end = -1
    for start, end in sorted(by_span, key=lambda se: (se[0], -(se[1] - se[0]))):
        if start < last_end:
            continue
        surviving_rules = tuple(by_span[(start, end)])
        unique_curies = tuple(dict.fromkeys(rule.curie for rule in surviving_rules))
        accepted.append(
            Stage1Span(
                start_char=start,
                end_char=end,
                surface_form=text[start:end],
                rules=surviving_rules,
                candidate_curies=unique_curies,
                requires_disambiguation=requires_stage2_disambiguation(surviving_rules),
            )
        )
        last_end = end
    return accepted


def detect_mentions(
    text: str,
    automaton: Any,
    *,
    disambiguator: MentionDisambiguator | None = None,
    entity_catalog: Mapping[str, DisambiguationCandidate] | None = None,
) -> list[Mention]:
    """Detect proper-noun entity mentions in unmodified ``text``.

    Args:
        text: Unmodified source text. Returned ``start_char`` and ``end_char``
            index directly into ``text`` and ``surface_form == text[start:end]``.
        automaton: ``pyahocorasick.Automaton`` built strictly over
            ``match_lexicon`` ``LexiconRule`` entries.
        disambiguator: Optional Stage 2 ``MentionDisambiguator`` invoked iff
            ``any(r.requires_llm_disambiguation)`` or ``len(unique_curies) > 1``.
            Returning ``None`` or ``"NIL"`` abstains and drops the mention.
        entity_catalog: Optional CURIE -> ``DisambiguationCandidate`` metadata
            used to enrich Stage 2 candidate descriptions.

    Returns:
        Sorted, non-overlapping ``Mention`` records.
    """
    stage1_spans = extract_stage1_spans(text, automaton)
    mentions: list[Mention] = []

    for span in stage1_spans:
        chosen_curie: str | None
        if span.requires_disambiguation:
            if disambiguator is not None:
                candidates = [
                    _candidate_for_curie(curie, span.rules, entity_catalog)
                    for curie in span.candidate_curies
                ]
                decision = disambiguator.disambiguate(
                    text=text,
                    start=span.start_char,
                    end=span.end_char,
                    surface=span.surface_form,
                    candidates=candidates,
                )
                if decision is None or decision == "NIL" or decision not in span.candidate_curies:
                    continue
                chosen_curie = decision
            elif len(span.candidate_curies) == 1:
                chosen_curie = span.candidate_curies[0]
            else:
                continue
        else:
            chosen_curie = span.candidate_curies[0]

        winning_rule = next((r for r in span.rules if r.curie == chosen_curie), span.rules[0])
        mentions.append(
            Mention(
                entity_id=chosen_curie,
                term_source=winning_rule.match_policy,
                surface_form=span.surface_form,
                start_char=span.start_char,
                end_char=span.end_char,
            )
        )
    return mentions


def _overlapping_token_bounds(
    tokens: list[tuple[int, int]],
    *,
    start: int,
    end: int,
) -> tuple[int, int]:
    first_idx = -1
    last_idx = -2
    for idx, (tok_start, tok_end) in enumerate(tokens):
        if tok_end <= start:
            continue
        if tok_start >= end:
            break
        if first_idx == -1:
            first_idx = idx
        last_idx = idx
    return (first_idx, last_idx) if first_idx != -1 else (0, -1)


def _contains_boundary_phrase(haystack: str, phrase: str) -> bool:
    """Return True iff ``phrase`` occurs case-insensitively at alphanumeric boundaries."""
    if not haystack or not phrase:
        return False
    hay_lower = offset_preserving_lower(haystack)
    needle = offset_preserving_lower(phrase)
    needle_len = len(needle)
    hay_len = len(haystack)
    search_from = 0
    while True:
        pos = hay_lower.find(needle, search_from)
        if pos == -1:
            return False
        end = pos + needle_len
        left_ok = pos == 0 or not haystack[pos - 1].isalnum()
        right_ok = end == hay_len or not haystack[end].isalnum()
        if left_ok and right_ok:
            return True
        search_from = pos + 1


def _candidate_for_curie(
    curie: str,
    rules: tuple[LexiconRule, ...],
    entity_catalog: Mapping[str, DisambiguationCandidate] | None,
) -> DisambiguationCandidate:
    if entity_catalog and curie in entity_catalog:
        return entity_catalog[curie]
    for rule in rules:
        if rule.curie == curie:
            return DisambiguationCandidate(
                curie=curie,
                pref_label=rule.pref_label,
                collection_name=rule.collection_name,
            )
    return DisambiguationCandidate(curie=curie)
