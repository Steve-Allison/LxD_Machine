"""Tests for the per-run ingest budget tracker.

The tracker is intentionally simple: count LLM calls, raise when the
configured cap is hit. These tests cover the full surface — defaults,
threshold semantics (last-call-allowed vs first-call-refused),
unbounded-when-None, snapshot shape.
"""

import pytest

from lxd.ingest.budget import BudgetExceededError, IngestBudgetTracker
from lxd.settings.models import IngestBudget


def test_unbounded_budget_never_raises() -> None:
    """The default `IngestBudget()` has no cap — `check()` is a no-op
    regardless of how many calls have been recorded."""
    tracker = IngestBudgetTracker(IngestBudget())
    for _ in range(1000):
        tracker.check()
        tracker.record_llm_call()
    assert tracker.llm_calls == 1000


def test_check_allows_calls_up_to_the_cap() -> None:
    """A budget of N permits exactly N LLM calls; the (N+1)th `check()`
    raises before the call is made."""
    tracker = IngestBudgetTracker(IngestBudget(max_llm_calls_per_run=3))
    for _ in range(3):
        tracker.check()
        tracker.record_llm_call()
    assert tracker.llm_calls == 3
    with pytest.raises(BudgetExceededError, match="max_llm_calls_per_run=3"):
        tracker.check()


def test_check_with_zero_cap_refuses_first_call() -> None:
    """A cap of 0 means no calls are permitted — the very first
    `check()` raises."""
    tracker = IngestBudgetTracker(IngestBudget(max_llm_calls_per_run=0))
    with pytest.raises(BudgetExceededError):
        tracker.check()


def test_record_llm_call_supports_batched_increment() -> None:
    """Callers can record multiple calls at once — useful for batched
    LLM dispatch where the spend is one logical operation but many
    sub-requests."""
    tracker = IngestBudgetTracker(IngestBudget(max_llm_calls_per_run=10))
    tracker.record_llm_call(count=4)
    assert tracker.llm_calls == 4
    tracker.check()
    tracker.record_llm_call(count=6)
    with pytest.raises(BudgetExceededError):
        tracker.check()


def test_snapshot_returns_current_counters() -> None:
    """`snapshot()` returns a serialisable dict suitable for the run-
    completion notes (no class instances, no enums)."""
    tracker = IngestBudgetTracker(IngestBudget(max_llm_calls_per_run=5))
    tracker.record_llm_call()
    tracker.record_llm_call()
    snapshot = tracker.snapshot()
    assert snapshot == {"llm_calls": 2}


def test_budget_message_explains_the_remediation_options() -> None:
    """The error message is the user's first-line debugging tool —
    it must name the cap, name the config field, and suggest a fix."""
    tracker = IngestBudgetTracker(IngestBudget(max_llm_calls_per_run=2))
    tracker.record_llm_call(count=2)
    with pytest.raises(BudgetExceededError) as excinfo:
        tracker.check()
    message = str(excinfo.value)
    assert "max_llm_calls_per_run=2" in message
    assert "ingest_budget.max_llm_calls_per_run" in message
