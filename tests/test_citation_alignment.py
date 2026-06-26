"""Tests for per-sentence citation alignment.

The alignment pipeline is the load-bearing contract for the
hallucination-risk signal: sentences without citations are surfaced as
unattributed claims. These tests pin the boundaries — invalid markers
dropped, valid markers retained in order, sentence boundaries
preserved, markers stripped from display text.
"""

import pytest
from pydantic import ValidationError

from lxd.synthesis import citation_alignment as _citation_alignment
from lxd.synthesis.citation_alignment import SentenceCitation, align_citations

_extract_valid_labels = _citation_alignment._extract_valid_labels  # pyright: ignore[reportPrivateUsage]
_strip_markers = _citation_alignment._strip_markers  # pyright: ignore[reportPrivateUsage]

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# align_citations end-to-end
# ---------------------------------------------------------------------------


def test_align_citations_pairs_each_sentence_with_its_markers() -> None:
    answer = (
        "Bloom's taxonomy organises learning [a.md#0]. ARCS has four components [b.md#1] [c.md#2]."
    )
    result = align_citations(
        answer_text=answer,
        valid_labels=["a.md#0", "b.md#1", "c.md#2"],
    )
    assert len(result) == 2
    assert result[0].citation_labels == ["a.md#0"]
    assert result[1].citation_labels == ["b.md#1", "c.md#2"]


def test_align_citations_marks_unattributed_sentence_with_empty_labels() -> None:
    answer = "Mayer's principles support multimedia learning [m.md#0]. Many people use them."
    result = align_citations(answer_text=answer, valid_labels=["m.md#0"])
    assert len(result) == 2
    assert result[0].citation_labels == ["m.md#0"]
    # Second sentence has no marker → unattributed.
    assert result[1].citation_labels == []


def test_align_citations_drops_unknown_markers_but_strips_them_from_text() -> None:
    answer = "ADDIE is a model [real.md#0] [fabricated.md#99]."
    result = align_citations(answer_text=answer, valid_labels=["real.md#0"])
    assert len(result) == 1
    # Only the valid marker is reported.
    assert result[0].citation_labels == ["real.md#0"]
    # Both markers are stripped from display text.
    assert "[real.md#0]" not in result[0].text
    assert "[fabricated.md#99]" not in result[0].text


def test_align_citations_deduplicates_markers_within_a_sentence() -> None:
    answer = "Repeated citation [a.md#0] said twice [a.md#0]."
    result = align_citations(answer_text=answer, valid_labels=["a.md#0"])
    assert result[0].citation_labels == ["a.md#0"]


def test_align_citations_preserves_marker_order() -> None:
    answer = "Three citations in order [c] [a] [b]."
    result = align_citations(answer_text=answer, valid_labels=["a", "b", "c"])
    assert result[0].citation_labels == ["c", "a", "b"]


def test_align_citations_empty_text_returns_empty_list() -> None:
    assert align_citations(answer_text="", valid_labels=["a"]) == []
    assert align_citations(answer_text="   \n  ", valid_labels=["a"]) == []


def test_align_citations_handles_question_and_exclamation_sentences() -> None:
    answer = "Did you know? Yes, ARCS works [a.md#0]! It does."
    result = align_citations(answer_text=answer, valid_labels=["a.md#0"])
    assert len(result) == 3
    # The middle sentence carries the marker.
    assert result[1].citation_labels == ["a.md#0"]
    assert result[0].citation_labels == []
    assert result[2].citation_labels == []


def test_align_citations_keeps_terminal_punctuation_after_marker_strip() -> None:
    answer = "Bloom's first level is remember [a]."
    result = align_citations(answer_text=answer, valid_labels=["a"])
    # The trailing period must survive the marker strip.
    assert result[0].text.endswith(".")


# ---------------------------------------------------------------------------
# _strip_markers — display text cleanliness
# ---------------------------------------------------------------------------


def test_strip_markers_collapses_orphan_whitespace() -> None:
    assert _strip_markers("X [a]  Y") == "X Y"


def test_strip_markers_tightens_space_before_terminal_punctuation() -> None:
    assert _strip_markers("X [a] .") == "X."


def test_strip_markers_with_no_markers_passes_through() -> None:
    assert _strip_markers("Plain sentence.") == "Plain sentence."


# ---------------------------------------------------------------------------
# _extract_valid_labels invariants
# ---------------------------------------------------------------------------


def test_extract_valid_labels_returns_empty_when_no_markers() -> None:
    assert _extract_valid_labels("Plain prose with no markers.", {"a"}) == []


def test_extract_valid_labels_returns_empty_when_all_markers_invalid() -> None:
    assert _extract_valid_labels("Some text [fake] [also_fake].", {"real"}) == []


def test_extract_valid_labels_trims_label_whitespace() -> None:
    # The model occasionally puts spaces inside brackets.
    assert _extract_valid_labels("Text [  a  ].", {"a"}) == ["a"]


# ---------------------------------------------------------------------------
# Pydantic invariants
# ---------------------------------------------------------------------------


def test_sentence_citation_is_frozen() -> None:
    sc = SentenceCitation(text="x", citation_labels=["a"])
    with pytest.raises(ValidationError):
        sc.text = "y"  # type: ignore[misc]


def test_sentence_citation_rejects_extra_keys() -> None:
    with pytest.raises(ValidationError):
        SentenceCitation.model_validate({"text": "x", "citation_labels": [], "extra": 1})
