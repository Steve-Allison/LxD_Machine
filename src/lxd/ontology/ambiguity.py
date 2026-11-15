"""Stage 2 disambiguation escalation predicates over ``match_lexicon`` rules."""

from collections import defaultdict
from collections.abc import Iterable, Sequence

from lxd.ontology.matcher import LexiconRule, offset_preserving_lower


def requires_stage2_disambiguation(rules: Sequence[LexiconRule]) -> bool:
    """Return True iff surviving span rules must escalate to Stage 2 disambiguation.

    Escalates iff any surviving rule sets ``requires_llm_disambiguation`` or more
    than one distinct ``coe:Cxxxxx`` CURIE survives Stage 1 filtering.
    """
    if not rules:
        return False
    if any(rule.requires_llm_disambiguation for rule in rules):
        return True
    unique_curies = {rule.curie for rule in rules}
    return len(unique_curies) > 1


def escalated_surface_forms_with_candidates(
    records: Iterable[LexiconRule],
) -> dict[str, list[str]]:
    """Return ``{surface_lower: sorted [curie, ...]}`` for Stage 2 surface forms.

    Includes every lowercase surface form whose ``LexiconRule`` group has
    ``len(unique_curies) > 1`` or ``any(r.requires_llm_disambiguation)``.
    """
    by_surface: dict[str, list[LexiconRule]] = defaultdict(list)
    for rule in records:
        key = offset_preserving_lower(rule.surface)
        if key:
            by_surface[key].append(rule)

    escalated: dict[str, list[str]] = {}
    for surface_lower, group in sorted(by_surface.items()):
        if requires_stage2_disambiguation(group):
            escalated[surface_lower] = sorted({rule.curie for rule in group})
    return escalated
