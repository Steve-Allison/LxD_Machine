"""Compute the set of ambiguous surface forms from matcher term records.

A surface form is *ambiguous* when the ontology maps it to more than one
canonical entity id. The classic example is the acronym "ID" — it could
mean ``instructional_design`` or ``identifier`` depending on context.
Aho-Corasick mention detection cannot resolve that on its own;
:func:`ambiguous_surface_forms_with_candidates` produces the lookup
table that the embedding-based disambiguator (`B-KG-2`) consults at
mention time.

Surface forms with a single canonical id are intentionally left out of
the result map: the disambiguation lane is opt-in per surface form, so
unambiguous matches incur zero overhead.
"""

from collections import defaultdict
from collections.abc import Iterable

from lxd.ontology.matcher import MatcherTermRecord


def ambiguous_surface_forms_with_candidates(
    records: Iterable[MatcherTermRecord],
) -> dict[str, list[str]]:
    """Return ``{normalized_term: [entity_id, ...]}`` for terms with >1 candidate.

    The candidate list is sorted by ``entity_id`` for deterministic
    iteration so disambiguator decisions are reproducible across runs.
    Terms with exactly one candidate are excluded — the caller does not
    need to disambiguate them, and including them would force a
    superfluous lookup on every mention.
    """
    by_term: dict[str, set[str]] = defaultdict(set)
    for record in records:
        by_term[record.normalized_term].add(record.entity_id)
    return {term: sorted(ids) for term, ids in by_term.items() if len(ids) > 1}
