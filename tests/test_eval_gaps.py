"""Unit tests for retrieval-eval gap ticket building, writing, and listing."""

from pathlib import Path

import pytest

from lxd.eval.gaps import (
    GapTicket,
    build_gap_tickets,
    list_gap_tickets,
    make_ticket_id,
    write_gap_tickets,
)
from lxd.retrieval.eval import EvalCaseResult, EvalSummary

pytestmark = [pytest.mark.unit]


def _summary(cases: list[EvalCaseResult]) -> EvalSummary:
    question_count = len(cases)
    mean_recall = sum(c.recall_at_10 for c in cases) / question_count if cases else 0.0
    mean_mrr = sum(c.mrr_at_10 for c in cases) / question_count if cases else 0.0
    return EvalSummary(
        question_count=question_count,
        mean_recall_at_10=mean_recall,
        mean_mrr_at_10=mean_mrr,
        cases=cases,
    )


def _case(
    question: str = "What is cognitive load theory?",
    recall_at_10: float = 0.0,
    mrr_at_10: float = 0.0,
    expected: list[str] | None = None,
    ranked: list[str] | None = None,
    warnings: list[str] | None = None,
) -> EvalCaseResult:
    return EvalCaseResult(
        question=question,
        recall_at_10=recall_at_10,
        mrr_at_10=mrr_at_10,
        expected=expected if expected is not None else ["cognitive_load.md"],
        ranked=ranked if ranked is not None else ["other_page.md"],
        warnings=warnings if warnings is not None else [],
    )


def test_make_ticket_id_is_deterministic_and_order_independent() -> None:
    id_a = make_ticket_id("q?", ["b.md", "a.md"])
    id_b = make_ticket_id("q?", ["a.md", "b.md"])
    id_c = make_ticket_id("different?", ["a.md", "b.md"])

    assert id_a == id_b
    assert id_a != id_c
    assert len(id_a) == 64  # blake3 hex digest length


def test_build_gap_tickets_classifies_empty_results() -> None:
    case = _case(recall_at_10=0.0, mrr_at_10=0.0, ranked=[])
    summary = _summary([case])

    tickets = build_gap_tickets(summary)

    assert len(tickets) == 1
    assert tickets[0].gap_kind == "empty_results"
    assert tickets[0].status == "open"


def test_build_gap_tickets_classifies_missed_source() -> None:
    case = _case(
        recall_at_10=0.0,
        mrr_at_10=0.0,
        expected=["cognitive_load.md"],
        ranked=["unrelated.md", "other.md"],
    )
    summary = _summary([case])

    tickets = build_gap_tickets(summary)

    assert len(tickets) == 1
    assert tickets[0].gap_kind == "missed_source"


def test_build_gap_tickets_classifies_weak_rank() -> None:
    case = _case(
        recall_at_10=0.5,
        mrr_at_10=0.2,
        expected=["a.md", "b.md"],
        ranked=["c.md", "a.md"],
    )
    summary = _summary([case])

    tickets = build_gap_tickets(summary)

    assert len(tickets) == 1
    assert tickets[0].gap_kind == "weak_rank"


def test_build_gap_tickets_classifies_eval_warning_on_perfect_score() -> None:
    case = _case(
        recall_at_10=1.0,
        mrr_at_10=1.0,
        expected=["a.md"],
        ranked=["a.md"],
        warnings=["reranker unavailable, falling back to dense scores"],
    )
    summary = _summary([case])

    tickets = build_gap_tickets(summary)

    assert len(tickets) == 1
    assert tickets[0].gap_kind == "eval_warning"


def test_build_gap_tickets_skips_clean_case() -> None:
    case = _case(recall_at_10=1.0, mrr_at_10=1.0, expected=["a.md"], ranked=["a.md"], warnings=[])
    summary = _summary([case])

    tickets = build_gap_tickets(summary)

    assert tickets == []


def test_build_gap_tickets_caps_ranked_top_at_ten() -> None:
    ranked = [f"page_{i}.md" for i in range(15)]
    case = _case(recall_at_10=0.0, mrr_at_10=0.0, expected=["missing.md"], ranked=ranked)
    summary = _summary([case])

    tickets = build_gap_tickets(summary)

    assert tickets[0].ranked_top == ranked[:10]


def test_write_and_list_gap_tickets_roundtrip(tmp_path: Path) -> None:
    directory = tmp_path / "gaps"
    cases = [
        _case(question="Q1?", recall_at_10=0.0, mrr_at_10=0.0),
        _case(question="Q2?", recall_at_10=1.0, mrr_at_10=1.0, expected=["a.md"], ranked=["a.md"]),
    ]
    summary = _summary(cases)
    tickets = build_gap_tickets(summary)
    assert len(tickets) == 1  # only Q1 produced a ticket

    written = write_gap_tickets(tickets, directory)

    assert len(written) == 1
    assert written[0].exists()
    assert written[0].parent == directory

    loaded = list_gap_tickets(directory)
    assert len(loaded) == 1
    assert loaded[0].question == "Q1?"
    assert loaded[0].status == "open"


def test_write_gap_tickets_creates_directory(tmp_path: Path) -> None:
    directory = tmp_path / "nested" / "gaps"
    assert not directory.exists()

    case = _case(recall_at_10=0.0, mrr_at_10=0.0)
    tickets = build_gap_tickets(_summary([case]))
    write_gap_tickets(tickets, directory)

    assert directory.exists()


def test_write_gap_tickets_upserts_open_ticket(tmp_path: Path) -> None:
    directory = tmp_path / "gaps"
    case = _case(question="Q?", recall_at_10=0.0, mrr_at_10=0.0)
    first_run = build_gap_tickets(_summary([case]))
    write_gap_tickets(first_run, directory)

    # Re-run eval; the same gap is still present, so the ticket is refreshed.
    second_run = build_gap_tickets(_summary([case]))
    written_again = write_gap_tickets(second_run, directory)

    assert len(written_again) == 1
    loaded = list_gap_tickets(directory)
    assert len(loaded) == 1
    assert loaded[0].ticket_id == first_run[0].ticket_id


def test_write_gap_tickets_does_not_reopen_closed_ticket(tmp_path: Path) -> None:
    directory = tmp_path / "gaps"
    case = _case(question="Q?", recall_at_10=0.0, mrr_at_10=0.0)
    tickets = build_gap_tickets(_summary([case]))
    written = write_gap_tickets(tickets, directory)

    # Human reviewer closes the ticket by hand.
    closed = tickets[0].model_copy(update={"status": "closed"})
    written[0].write_text(closed.model_dump_json(indent=2) + "\n", encoding="utf-8")

    # Re-running eval must not silently reopen it.
    written_again = write_gap_tickets(build_gap_tickets(_summary([case])), directory)

    assert written_again == []
    loaded = list_gap_tickets(directory, status="closed")
    assert len(loaded) == 1
    assert loaded[0].ticket_id == tickets[0].ticket_id


def test_list_gap_tickets_missing_directory_returns_empty(tmp_path: Path) -> None:
    assert list_gap_tickets(tmp_path / "does_not_exist") == []


def test_list_gap_tickets_filters_by_status(tmp_path: Path) -> None:
    directory = tmp_path / "gaps"
    open_ticket = GapTicket(
        ticket_id=make_ticket_id("open?", ["a.md"]),
        question="open?",
        expected_sources=["a.md"],
        ranked_top=[],
        recall_at_10=0.0,
        mrr_at_10=0.0,
        gap_kind="missed_source",
        notes="",
        created_at="2026-07-20T00:00:00+00:00",
        status="open",
    )
    closed_ticket = open_ticket.model_copy(
        update={
            "ticket_id": make_ticket_id("closed?", ["b.md"]),
            "question": "closed?",
            "status": "closed",
        }
    )
    write_gap_tickets([open_ticket, closed_ticket], directory)

    assert [t.question for t in list_gap_tickets(directory, status="open")] == ["open?"]
    assert [t.question for t in list_gap_tickets(directory, status="closed")] == ["closed?"]
    assert len(list_gap_tickets(directory)) == 2
