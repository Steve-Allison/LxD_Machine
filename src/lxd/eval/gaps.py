"""Turn retrieval-eval failures into actionable, human-reviewed gap tickets.

Every case in an :class:`lxd.retrieval.eval.EvalSummary` where retrieval fell
short of a perfect score becomes a :class:`GapTicket`: a small JSON file a
human reviewer can read, action (fix the wiki page, adjust the ontology,
re-word the eval case), and close. This module never edits the wiki or
ontology itself — see ``Plans/SOTA_PRODUCT_CONTRACT.md`` non-goals — it only
surfaces gaps for a human to triage.

Tickets are upserted by content-addressed ``ticket_id`` (a BLAKE3 hash of the
question and its expected sources), so re-running the eval after a fix
naturally re-creates the same ticket if the gap persists, and never
resurrects a ticket a human has already closed.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from lxd.domain.ids import blake3_hex
from lxd.domain.time import utc_now
from lxd.retrieval.eval import EvalCaseResult, EvalSummary

GapKind = Literal["missed_source", "weak_rank", "empty_results", "eval_warning"]
GapStatus = Literal["open", "closed"]


class GapTicket(BaseModel):
    """One actionable retrieval-eval gap, persisted as a single JSON file.

    ``status`` defaults to ``open``; a human reviewer flips it to ``closed``
    directly in the JSON file (or via future tooling) once the underlying
    gap has been addressed or judged not actionable. :func:`write_gap_tickets`
    respects that decision and will not silently reopen a closed ticket.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    ticket_id: str = Field(description="BLAKE3 hash of the question and expected sources.")
    question: str
    expected_sources: list[str] = Field(default_factory=list)
    ranked_top: list[str] = Field(
        default_factory=list, description="Top-10 ranked source paths retrieval actually returned."
    )
    recall_at_10: float
    mrr_at_10: float
    gap_kind: GapKind
    notes: str = Field(default="", description="Human-readable summary of the gap.")
    created_at: str
    status: GapStatus = "open"


def make_ticket_id(question: str, expected_sources: list[str]) -> str:
    """Derive a stable ticket ID from the question and its expected sources.

    Sorting the expected sources before hashing means the ID is stable
    regardless of the order they were supplied in the eval set.
    """
    return blake3_hex(question, *sorted(expected_sources))


def build_gap_tickets(summary: EvalSummary) -> list[GapTicket]:
    """Derive one :class:`GapTicket` per failing case in an eval summary.

    Classification, in priority order:
        - ``empty_results``: retrieval returned nothing at all — usually a
          pipeline or config problem rather than a corpus gap.
        - ``missed_source``: none of the expected sources appeared in the
          top 10 (``recall_at_10 == 0.0``) — the corpus likely lacks
          coverage, or the expected source path is stale.
        - ``weak_rank``: some but not all expected sources were found, or
          the best match ranked too low to be first (``mrr_at_10 == 0.0``
          despite partial recall) — a ranking/relevance problem rather than
          a missing-content problem.
        - ``eval_warning``: recall and MRR were both perfect, but the run
          logged warnings (e.g. config drift, reranker fallback) — included
          so degraded-but-passing runs stay visible to a reviewer.

    Cases with perfect recall, perfect MRR, and no warnings produce no
    ticket — there is nothing actionable to report.
    """
    tickets: list[GapTicket] = []
    created_at = utc_now()
    for case in summary.cases:
        gap_kind = _classify_gap(case)
        if gap_kind is None:
            continue
        tickets.append(
            GapTicket(
                ticket_id=make_ticket_id(case.question, case.expected),
                question=case.question,
                expected_sources=case.expected,
                ranked_top=case.ranked[:10],
                recall_at_10=case.recall_at_10,
                mrr_at_10=case.mrr_at_10,
                gap_kind=gap_kind,
                notes=_build_notes(case, gap_kind),
                created_at=created_at,
                status="open",
            )
        )
    return tickets


def write_gap_tickets(tickets: list[GapTicket], directory: Path) -> list[Path]:
    """Upsert open gap tickets as one JSON file each under ``directory``.

    A ticket a human has already closed (same ``ticket_id`` already on disk
    with ``status == "closed"``) is left untouched — this is a human-in-the-
    loop workflow, and re-running eval must never silently reopen a
    triaged ticket. Every other ticket is written (created or overwritten).

    Returns:
        Paths actually written, in the same order as ``tickets``.
    """
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for ticket in tickets:
        path = directory / f"{ticket.ticket_id}.json"
        if path.exists():
            existing = GapTicket.model_validate_json(path.read_text(encoding="utf-8"))
            if existing.status == "closed":
                continue
        path.write_text(ticket.model_dump_json(indent=2) + "\n", encoding="utf-8")
        written.append(path)
    return written


def list_gap_tickets(directory: Path, *, status: GapStatus | None = None) -> list[GapTicket]:
    """Load every gap ticket under ``directory``, optionally filtered by status.

    Returns an empty list if the directory does not exist. Results are
    sorted by ``question`` then ``ticket_id`` for deterministic ordering.
    """
    if not directory.exists():
        return []
    tickets = [
        GapTicket.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("*.json"))
    ]
    if status is not None:
        tickets = [ticket for ticket in tickets if ticket.status == status]
    return sorted(tickets, key=lambda t: (t.question, t.ticket_id))


def _classify_gap(case: EvalCaseResult) -> GapKind | None:
    """Classify a single :class:`lxd.retrieval.eval.EvalCaseResult` into a gap kind."""
    if not case.ranked:
        return "empty_results"
    if case.recall_at_10 <= 0.0:
        return "missed_source"
    if case.recall_at_10 < 1.0 or case.mrr_at_10 == 0.0:
        return "weak_rank"
    if case.warnings:
        return "eval_warning"
    return None


def _build_notes(case: EvalCaseResult, gap_kind: GapKind) -> str:
    """Render a short human-readable summary of why this case became a ticket."""
    recall = case.recall_at_10
    mrr = case.mrr_at_10
    expected = case.expected
    ranked = case.ranked
    warnings = case.warnings
    found = [source for source in expected if source in set(ranked[:10])]
    missing = [source for source in expected if source not in set(ranked[:10])]

    lines = [f"recall@10={recall:.2f} mrr@10={mrr:.2f} ({gap_kind})"]
    if expected:
        lines.append(f"found {len(found)}/{len(expected)} expected source(s) in top 10")
    if missing:
        lines.append(f"missing: {', '.join(missing)}")
    if warnings:
        lines.append(f"warnings: {'; '.join(warnings)}")
    return " | ".join(lines)
