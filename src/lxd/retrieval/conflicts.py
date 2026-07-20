"""Heuristic contradiction detection over extracted claims (V1 — no LLM call).

Graph context should never silently pick a winner between two claims that
disagree — see the ``### Conflicting Claims`` section this module feeds
in :mod:`lxd.retrieval.graph_routing`. A full LLM-adjudicated contradiction
detector is future work; this V1 uses two cheap textual heuristics so the
signal is available at query time with zero extra API spend.
"""

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Final

from lxd.stores.models import ClaimRecord

_ASSERTION_LIKE_TYPES: Final = frozenset({"assertion", "definition"})

# Cues that mark a claim as negating/contradicting a plain assertion
# about the same subject (e.g. "X improves retention" vs "there is no
# evidence X improves retention").
_NEGATION_CUES: Final = (
    "not ",
    "never ",
    "no evidence",
    "contrary",
    "however research shows",
    "debunk",
    "myth",
)

# Opposing-polarity vocabulary for "comparison"-type claims (e.g. "X is
# more effective than Y" vs "X is less effective than Y").
_POSITIVE_POLARITY_WORDS: Final = frozenset(
    {
        "better",
        "more effective",
        "superior",
        "increases",
        "improves",
        "faster",
        "higher",
        "outperforms",
        "stronger",
    }
)
_NEGATIVE_POLARITY_WORDS: Final = frozenset(
    {
        "worse",
        "less effective",
        "inferior",
        "decreases",
        "reduces",
        "slower",
        "lower",
        "underperforms",
        "weaker",
    }
)

_GroupKey = frozenset[str] | tuple[str, str]


@dataclass(frozen=True, slots=True)
class ClaimConflict:
    """A pair of claims that heuristically appear to disagree."""

    claim_a_id: str
    claim_b_id: str
    claim_a_text: str
    claim_b_text: str
    subject_entity_id: str | None
    object_entity_id: str | None
    reason: str


def detect_claim_conflicts(
    claims: list[ClaimRecord], *, max_conflicts: int
) -> list[ClaimConflict]:
    """Flag pairs of claims that look like they disagree, without an LLM call.

    Claims are grouped so only claims *about the same thing* are ever
    compared:

    - Both ``subject_entity_id`` and ``object_entity_id`` present → grouped
      by the unordered pair (a ``frozenset``), so "A causes B" and
      "B causes A" extractions still land in the same group.
    - Only ``subject_entity_id`` present → grouped by
      ``(subject_entity_id, claim_type)``, since without an object entity
      claim type is the only reliable signal that two claims describe the
      same kind of statement about the subject.
    - Neither present → the claim cannot be grouped meaningfully and is
      skipped.

    Within a group, every pair is checked for two heuristics:

    1. Both claims are assertion/definition-type and exactly one carries a
       negation cue (``"not "``, ``"never "``, ``"debunk"``, ...) — the
       classic "X is Y" vs "X is not Y" shape.
    2. Both claims are comparison-type and use opposing polarity language
       (``"better"`` vs ``"worse"``, ``"increases"`` vs ``"decreases"``,
       ...).

    Results are capped at ``max_conflicts``, returned in the order groups
    and pairs are encountered — callers that want confidence-priority
    should pre-sort ``claims`` before calling.
    """
    if max_conflicts <= 0 or len(claims) < 2:
        return []

    groups: dict[_GroupKey, list[ClaimRecord]] = defaultdict(list)
    for claim in claims:
        key = _group_key(claim)
        if key is not None:
            groups[key].append(claim)

    conflicts: list[ClaimConflict] = []
    for group_claims in groups.values():
        if len(group_claims) < 2:
            continue
        for claim_a, claim_b in combinations(group_claims, 2):
            reason = _conflict_reason(claim_a, claim_b)
            if reason is None:
                continue
            conflicts.append(
                ClaimConflict(
                    claim_a_id=claim_a.claim_id,
                    claim_b_id=claim_b.claim_id,
                    claim_a_text=claim_a.claim_text,
                    claim_b_text=claim_b.claim_text,
                    subject_entity_id=claim_a.subject_entity_id,
                    object_entity_id=claim_a.object_entity_id,
                    reason=reason,
                )
            )
            if len(conflicts) >= max_conflicts:
                return conflicts
    return conflicts


def _group_key(claim: ClaimRecord) -> _GroupKey | None:
    if claim.subject_entity_id and claim.object_entity_id:
        return frozenset({claim.subject_entity_id, claim.object_entity_id})
    if claim.subject_entity_id:
        return (claim.subject_entity_id, claim.claim_type)
    return None


def _conflict_reason(claim_a: ClaimRecord, claim_b: ClaimRecord) -> str | None:
    if claim_a.claim_type in _ASSERTION_LIKE_TYPES and claim_b.claim_type in _ASSERTION_LIKE_TYPES:
        a_negated = _has_negation_cue(claim_a.claim_text)
        b_negated = _has_negation_cue(claim_b.claim_text)
        if a_negated != b_negated:
            return (
                "One claim carries a negation/contradiction cue (e.g. 'not', "
                "'never', 'debunk') about the same subject the other states "
                "plainly — the two likely disagree."
            )
    if (
        claim_a.claim_type == "comparison"
        and claim_b.claim_type == "comparison"
        and _opposing_polarity(claim_a.claim_text, claim_b.claim_text)
    ):
        return (
            "Both claims compare the same subject(s) but use opposing "
            "polarity language (e.g. 'better' vs 'worse')."
        )
    return None


def _has_negation_cue(text: str) -> bool:
    lowered = text.lower()
    return any(cue in lowered for cue in _NEGATION_CUES)


def _has_any_word(text: str, words: frozenset[str]) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in words)


def _opposing_polarity(text_a: str, text_b: str) -> bool:
    a_positive = _has_any_word(text_a, _POSITIVE_POLARITY_WORDS)
    a_negative = _has_any_word(text_a, _NEGATIVE_POLARITY_WORDS)
    b_positive = _has_any_word(text_b, _POSITIVE_POLARITY_WORDS)
    b_negative = _has_any_word(text_b, _NEGATIVE_POLARITY_WORDS)
    return (a_positive and b_negative) or (a_negative and b_positive)
