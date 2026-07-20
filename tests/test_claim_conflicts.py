"""Tests for the heuristic claim-contradiction detector (Phase 1c)."""

from lxd.retrieval.conflicts import detect_claim_conflicts
from lxd.stores.models import ClaimRecord


def _claim(
    claim_id: str,
    *,
    text: str,
    claim_type: str = "assertion",
    subject: str | None = "entity-a",
    obj: str | None = None,
    confidence: float = 0.8,
) -> ClaimRecord:
    return ClaimRecord(
        claim_id=claim_id,
        chunk_id=f"chunk-{claim_id}",
        document_id="doc-1",
        source_rel_path="x.md",
        claim_text=text,
        subject_entity_id=subject,
        object_entity_id=obj,
        claim_type=claim_type,
        confidence=confidence,
        extraction_model="test",
        extracted_at="2026-05-05T00:00:00Z",
    )


def test_no_conflicts_when_fewer_than_two_claims() -> None:
    assert detect_claim_conflicts([], max_conflicts=5) == []
    assert detect_claim_conflicts([_claim("c1", text="X improves retention.")], max_conflicts=5) == []


def test_no_conflicts_when_max_conflicts_is_zero() -> None:
    claims = [
        _claim("c1", text="Spaced repetition improves retention."),
        _claim("c2", text="There is no evidence spaced repetition improves retention."),
    ]
    assert detect_claim_conflicts(claims, max_conflicts=0) == []


def test_detects_negation_conflict_within_same_subject_object_pair() -> None:
    claims = [
        _claim(
            "c1",
            text="Spaced repetition improves long-term retention.",
            subject="concept-spaced-repetition",
            obj="concept-retention",
        ),
        _claim(
            "c2",
            text="There is no evidence spaced repetition improves long-term retention.",
            subject="concept-spaced-repetition",
            obj="concept-retention",
        ),
    ]

    conflicts = detect_claim_conflicts(claims, max_conflicts=5)

    assert len(conflicts) == 1
    conflict = conflicts[0]
    assert {conflict.claim_a_id, conflict.claim_b_id} == {"c1", "c2"}
    assert "negation" in conflict.reason.lower() or "disagree" in conflict.reason.lower()


def test_detects_negation_conflict_grouped_by_subject_and_claim_type_when_no_object() -> None:
    claims = [
        _claim("c1", text="Cognitive load theory is well supported.", subject="concept-clt", obj=None),
        _claim(
            "c2",
            text="Cognitive load theory is not well supported by recent meta-analyses.",
            subject="concept-clt",
            obj=None,
        ),
    ]

    conflicts = detect_claim_conflicts(claims, max_conflicts=5)

    assert len(conflicts) == 1


def test_does_not_group_claims_with_different_subjects() -> None:
    claims = [
        _claim("c1", text="X improves retention.", subject="concept-x"),
        _claim("c2", text="There is no evidence Y improves retention.", subject="concept-y"),
    ]

    assert detect_claim_conflicts(claims, max_conflicts=5) == []


def test_does_not_group_claims_with_different_claim_types_when_no_object() -> None:
    claims = [
        _claim("c1", text="X is defined as Y.", claim_type="definition", subject="concept-x"),
        _claim("c2", text="There is no evidence for X.", claim_type="assertion", subject="concept-x"),
    ]

    assert detect_claim_conflicts(claims, max_conflicts=5) == []


def test_no_conflict_when_neither_claim_is_negated() -> None:
    claims = [
        _claim("c1", text="X improves retention.", subject="concept-x", obj="concept-y"),
        _claim("c2", text="X strengthens retention over time.", subject="concept-x", obj="concept-y"),
    ]

    assert detect_claim_conflicts(claims, max_conflicts=5) == []


def test_no_conflict_when_both_claims_are_negated() -> None:
    claims = [
        _claim("c1", text="There is no evidence X improves retention.", subject="concept-x", obj="concept-y"),
        _claim("c2", text="Contrary to popular belief, X never improves retention.", subject="concept-x", obj="concept-y"),
    ]

    assert detect_claim_conflicts(claims, max_conflicts=5) == []


def test_detects_opposing_polarity_in_comparison_claims() -> None:
    claims = [
        _claim(
            "c1",
            text="Method A is more effective than Method B for skill retention.",
            claim_type="comparison",
            subject="method-a",
            obj="method-b",
        ),
        _claim(
            "c2",
            text="Method A is less effective than Method B for skill retention.",
            claim_type="comparison",
            subject="method-a",
            obj="method-b",
        ),
    ]

    conflicts = detect_claim_conflicts(claims, max_conflicts=5)

    assert len(conflicts) == 1
    assert "polarity" in conflicts[0].reason.lower()


def test_no_conflict_when_comparison_claims_share_polarity() -> None:
    claims = [
        _claim(
            "c1",
            text="Method A is more effective than Method B.",
            claim_type="comparison",
            subject="method-a",
            obj="method-b",
        ),
        _claim(
            "c2",
            text="Method A is superior to Method B in most contexts.",
            claim_type="comparison",
            subject="method-a",
            obj="method-b",
        ),
    ]

    assert detect_claim_conflicts(claims, max_conflicts=5) == []


def test_skips_claims_with_no_subject_entity() -> None:
    claims = [
        _claim("c1", text="X improves retention.", subject=None, obj=None),
        _claim("c2", text="There is no evidence X improves retention.", subject=None, obj=None),
    ]

    assert detect_claim_conflicts(claims, max_conflicts=5) == []


def test_max_conflicts_caps_results() -> None:
    claims = [
        _claim(
            f"pos-{i}",
            text=f"Concept {i} improves outcomes.",
            subject="shared-subject",
            obj="shared-object",
        )
        for i in range(3)
    ] + [
        _claim(
            f"neg-{i}",
            text=f"There is no evidence concept {i} improves outcomes.",
            subject="shared-subject",
            obj="shared-object",
        )
        for i in range(3)
    ]

    conflicts = detect_claim_conflicts(claims, max_conflicts=2)

    assert len(conflicts) == 2


def test_subject_object_pair_is_order_independent() -> None:
    """Claims with subject/object swapped still land in the same group."""
    claims = [
        _claim("c1", text="A causes B to increase.", subject="entity-a", obj="entity-b"),
        _claim(
            "c2",
            text="There is no evidence A causes B to increase.",
            subject="entity-b",
            obj="entity-a",
        ),
    ]

    conflicts = detect_claim_conflicts(claims, max_conflicts=5)

    assert len(conflicts) == 1
