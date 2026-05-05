"""Per-run ingest budget tracker.

A misconfigured ``--full`` ingest against a large corpus can burn LLM-API
spend before any error-class circuit breaker trips: relation extraction
makes one LLM call per qualifying chunk, regardless of result quality.
This tracker counts those calls and aborts the run when the configured
ceiling is reached, so over-large corpora and bad ``min_entity_mentions``
settings cannot keep spending unbounded.

Scope today:

* Counts LLM calls during ingest (relation extraction).
* Embedding-token tracking and per-call cost estimation are **not** wired
  through; cap embedding spend via the provider's account dashboard.

Threading model: the ingest pipeline is a single per-source loop today,
so this tracker is intentionally not thread-safe — adding a lock costs
real wall-clock for no benefit. If concurrent ingest is added later, the
counter must be guarded.
"""

from __future__ import annotations

from lxd.settings.models import IngestBudget


class BudgetExceededError(RuntimeError):
    """Raised when an ingest run hits its configured cost ceiling.

    The pipeline catches this at the per-source loop boundary and ends
    the run cleanly with status ``aborted_budget`` so an in-flight
    chunk's manifest row does not silently flip to ``searchable``.
    """


class IngestBudgetTracker:
    """Counts LLM calls and refuses further work when the cap is hit.

    Construct once per :func:`lxd.ingest.pipeline.run_ingest` invocation.
    Pre-call: invoke :meth:`check` before each LLM call. Post-call:
    invoke :meth:`record_llm_call` on success. Skipped chunks (no
    qualifying mentions, no valid predicates) bypass both methods, so
    the count reflects actual LLM spend rather than chunks visited.
    """

    __slots__ = ("budget", "llm_calls")

    def __init__(self, budget: IngestBudget) -> None:
        self.budget = budget
        self.llm_calls = 0

    def check(self) -> None:
        """Raise :class:`BudgetExceededError` if the next call would exceed the cap.

        Called before issuing an LLM request. The check is "before this
        call" rather than "after the previous call" so the error message
        reflects the actual point of refusal — a chunk that would have
        made the (cap+1)th call sees the abort, not a chunk after it.
        """
        cap = self.budget.max_llm_calls_per_run
        if cap is None:
            return
        if self.llm_calls >= cap:
            raise BudgetExceededError(
                f"Ingest budget exceeded: max_llm_calls_per_run={cap} reached. "
                "Increase the cap in `ingest_budget.max_llm_calls_per_run` or "
                "narrow the corpus to fit."
            )

    def record_llm_call(self, count: int = 1) -> None:
        """Record one or more completed LLM calls."""
        self.llm_calls += count

    def snapshot(self) -> dict[str, int]:
        """Return a serialisable copy of the current counters."""
        return {"llm_calls": self.llm_calls}
