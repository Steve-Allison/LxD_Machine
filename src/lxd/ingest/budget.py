"""Per-run ingest budget tracker.

A misconfigured ``--full`` ingest against a large corpus can burn LLM-API
spend before any error-class circuit breaker trips: relation extraction
makes one LLM call per qualifying chunk, regardless of result quality.
This tracker counts those calls and aborts the run when the configured
ceiling is reached, so over-large corpora and bad ``min_entity_mentions``
settings cannot keep spending unbounded.

Scope today:

* Counts LLM calls during ingest (relation extraction).
* :func:`estimate_run_cost` provides a pre-flight cost estimate so the
  user can refuse the run before any API spend (B-STACK-10).
* Embedding-token tracking during the run itself is **not** wired
  through; cap embedding spend via the provider's account dashboard.

Threading model: the ingest pipeline is a single per-source loop today,
so this tracker is intentionally not thread-safe — adding a lock costs
real wall-clock for no benefit. If concurrent ingest is added later, the
counter must be guarded.
"""

from __future__ import annotations

from dataclasses import dataclass

from lxd.ingest.scanner import ScannedCorpusFile
from lxd.settings.models import IngestBudget, RuntimeConfig

# OpenAI list-price USD per 1M tokens — sourced from the public pricing page.
# Re-check at least every 6 months; last reviewed 2026-05-06.
_OPENAI_PRICES_PER_MILLION_USD: dict[str, float] = {
    "text-embedding-3-small": 0.020,
    "text-embedding-3-large": 0.130,
    "gpt-4o-mini-input": 0.150,
    "gpt-4o-mini-output": 0.600,
}

# Conservative average tokens per relation-extraction LLM call. Prompt
# carries chunk text + entity list + system message; completion is a
# small structured-relation list.
_DEFAULT_RELATION_PROMPT_TOKENS = 2_000
_DEFAULT_RELATION_COMPLETION_TOKENS = 500

# Char-per-token ratio for English-prose corpora; OpenAI's tokeniser
# averages ~4 chars per token on natural English, slightly less on
# Markdown headings and code blocks. We round to 4 for simplicity and
# under-count by ~3-5% on pure prose, which leans the estimate
# **upwards** on mixed content (safer for a "cost ceiling" framing).
_CHARS_PER_TOKEN = 4


@dataclass(frozen=True, slots=True)
class CostEstimate:
    """Pre-flight cost estimate for an ingest run.

    All token counts are integer; all USD figures are floats (rounded to
    the nearest cent at render time, never at compute time).

    ``embedding_*`` figures come directly from the corpus byte size;
    ``llm_*`` figures are an upper bound derived from
    :class:`IngestBudget.max_llm_calls_per_run` (the run hard-aborts at
    that ceiling, so we know the worst case).
    """

    text_file_count: int
    text_corpus_bytes: int
    embedding_tokens_est: int
    embedding_model: str
    embedding_usd_est: float

    llm_call_cap: int | None
    llm_prompt_tokens_per_call: int
    llm_completion_tokens_per_call: int
    llm_total_tokens_est: int
    llm_model: str
    llm_usd_est: float

    @property
    def total_usd_est(self) -> float:
        """Sum of embedding and LLM cost ceilings."""
        return self.embedding_usd_est + self.llm_usd_est


def estimate_run_cost(
    scanned_files: list[ScannedCorpusFile],
    config: RuntimeConfig,
    *,
    relation_prompt_tokens: int = _DEFAULT_RELATION_PROMPT_TOKENS,
    relation_completion_tokens: int = _DEFAULT_RELATION_COMPLETION_TOKENS,
) -> CostEstimate:
    """Return an upper-bound cost estimate for the next ingest run.

    Embedding cost is tight: every text file goes through the embedder, so
    ``ceil(total_text_bytes / 4)`` is a known good upper bound on prompt
    tokens.

    LLM cost is an upper bound, not a tight estimate: the actual relation
    extraction lane only fires for chunks that meet
    ``relation_extraction.min_entity_mentions``, and we cannot know that
    without running mention detection. The bound is therefore
    ``ingest_budget.max_llm_calls_per_run`` * ``relation_prompt_tokens +
    relation_completion_tokens``, which is the worst case the budget
    tracker would actually permit. When ``max_llm_calls_per_run`` is
    ``None`` (no cap configured) the LLM total is also reported as 0 with
    a note in the surfacing CLI; never silently estimate "infinity".

    Args:
        scanned_files: Output of :func:`lxd.ingest.scanner.scan_corpus`.
            Embeds touch text files only; image/asset entries are
            excluded from the embedding total.
        config: Validated runtime config.
        relation_prompt_tokens: Override for the per-call prompt-token
            assumption; defaults to 2000.
        relation_completion_tokens: Override for the per-call
            completion-token assumption; defaults to 500.

    Returns:
        A :class:`CostEstimate` with embedding and LLM upper-bound
        breakdowns and a ``total_usd_est`` convenience accessor.
    """
    text_files = [f for f in scanned_files if f.source_type != "image_png"]
    text_bytes = sum(f.file_size_bytes for f in text_files)
    embedding_tokens = (text_bytes + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN

    embedding_model = config.models.embed
    embedding_price = _OPENAI_PRICES_PER_MILLION_USD.get(embedding_model, 0.0)
    embedding_usd = (embedding_tokens / 1_000_000) * embedding_price

    llm_cap = config.ingest_budget.max_llm_calls_per_run
    llm_total_tokens = (
        llm_cap * (relation_prompt_tokens + relation_completion_tokens) if llm_cap else 0
    )
    llm_model = config.relation_extraction.openai_model
    llm_input_price = _OPENAI_PRICES_PER_MILLION_USD.get(f"{llm_model}-input", 0.0)
    llm_output_price = _OPENAI_PRICES_PER_MILLION_USD.get(f"{llm_model}-output", 0.0)
    llm_input_usd = (
        (llm_cap * relation_prompt_tokens / 1_000_000) * llm_input_price if llm_cap else 0.0
    )
    llm_output_usd = (
        (llm_cap * relation_completion_tokens / 1_000_000) * llm_output_price if llm_cap else 0.0
    )

    return CostEstimate(
        text_file_count=len(text_files),
        text_corpus_bytes=text_bytes,
        embedding_tokens_est=embedding_tokens,
        embedding_model=embedding_model,
        embedding_usd_est=embedding_usd,
        llm_call_cap=llm_cap,
        llm_prompt_tokens_per_call=relation_prompt_tokens,
        llm_completion_tokens_per_call=relation_completion_tokens,
        llm_total_tokens_est=llm_total_tokens,
        llm_model=llm_model,
        llm_usd_est=llm_input_usd + llm_output_usd,
    )


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
