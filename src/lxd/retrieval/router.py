"""Adaptive (Self-RAG / CRAG-style) query router.

The router makes one cheap LLM call before retrieval to classify the
incoming question. The returned :class:`QueryRoute` lets the pipeline:

  - **Skip retrieval** for meta queries ("hello", "what can you do?",
    "how does this work?") — saves cost and emits a graceful canned
    answer instead of stuffing meaningless evidence into synthesis.
  - **Widen** retrieval breadth (``broad``) for survey questions that
    need to cover many entities.
  - **Tighten** breadth (``narrow``) for focused factual lookups so
    the synthesiser sees a small, dense set of relevant chunks.

The router is *mandatory* — there is no enable/disable toggle. It
degrades gracefully: any LLM failure or unparseable response routes
the query through the ``standard`` default, with the warning surfaced
on the envelope so operators see when the router was bypassed.

Output schema (returned by the LLM in JSON mode):
  ``{"retrieve": bool, "breadth": "narrow"|"standard"|"broad",
     "rationale": "one short sentence"}``

Skip rationale: setting ``retrieve=False`` is a strong claim — the
prompt makes the model justify it. The breadth knob is bounded; the
``broad_dense_top_k`` / ``narrow_dense_top_k`` config values translate
the literal into integer depths the retrieval layer understands.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Literal, assert_never

import structlog
from pydantic import BaseModel, ConfigDict, Field

from lxd.ingest.llm_client import call_openai_async
from lxd.settings.models import AdaptiveRetrievalConfig

_log = structlog.get_logger(__name__)

type RouteBreadth = Literal["narrow", "standard", "broad"]


class QueryRoute(BaseModel):
    """Decision returned by the router for one incoming question."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    retrieve: bool = Field(
        description=(
            "False for meta / conversational / out-of-scope questions where "
            "retrieval would add no signal. True for any question the corpus "
            "could plausibly answer."
        )
    )
    breadth: RouteBreadth = Field(
        default="standard",
        description="Retrieval breadth knob — controls dense_top_k.",
    )
    rationale: str = Field(
        default="",
        description="One-sentence reason the model gave for the route.",
    )
    routed: bool = Field(
        default=True,
        description=(
            "False when the router LLM call failed and we fell back to "
            "the default ``retrieve=True / breadth=standard`` route. "
            "Surfaces as a warning on the envelope so operators see "
            "router bypass."
        ),
    )


_ROUTER_SYSTEM_PROMPT = """\
You are a query router for a Retrieval-Augmented Generation system over an
instructional-design knowledge base.

Given the user's QUESTION, return a JSON object with three keys:

  - ``retrieve`` (bool): true if the corpus could plausibly contain
    evidence relevant to the question; false ONLY for meta /
    conversational queries that the corpus cannot answer:
      * Greetings ("hello", "hi", "thanks")
      * Meta-questions about the system itself ("what can you do",
        "how does this work", "who built you")
      * Out-of-scope domains (sports scores, weather, current events,
        anything not about learning / instructional design)
      * Pure arithmetic or trivia not requiring evidence
    Default to ``retrieve=true`` when in doubt — false-negative
    retrieval bypasses are worse than false-positive retrieval.

  - ``breadth``: one of "narrow" / "standard" / "broad".
      * narrow: question asks about ONE specific concept, definition,
        or model (e.g. "What is Bloom's taxonomy?").
      * broad: survey questions covering many concepts or asking for
        comparison across the field (e.g. "What instructional design
        models address adult learners?").
      * standard: everything else — the default.

  - ``rationale``: one short sentence explaining the route.

Return ONLY the JSON object — no preamble, no markdown fences."""


def route_query(
    *,
    question: str,
    config: AdaptiveRetrievalConfig,
    api_key_env: str = "OPENAI_API_KEY",
) -> QueryRoute:
    """Classify a question into a :class:`QueryRoute`.

    Args:
        question: The user's question text. Must be non-empty (caller
            is responsible for empty-question handling).
        config: ``adaptive_retrieval`` section of the runtime config.
        api_key_env: Environment variable for the OpenAI API key.

    Returns:
        Always returns a :class:`QueryRoute`. On any failure (LLM
        timeout, malformed JSON, unknown breadth literal), returns the
        default route ``retrieve=True, breadth=standard, routed=False``.
        Never raises — synthesis bypass on a router glitch would be
        worse than running standard retrieval.
    """
    backend = config.router_backend
    if backend == "openai":
        return _route_openai(question=question, config=config, api_key_env=api_key_env)
    if backend == "ollama":
        return _route_ollama(config)
    assert_never(backend)


def _route_openai(
    *,
    question: str,
    config: AdaptiveRetrievalConfig,
    api_key_env: str,
) -> QueryRoute:
    try:
        raw = asyncio.run(
            call_openai_async(
                system_prompt=_ROUTER_SYSTEM_PROMPT,
                user_prompt=f"QUESTION:\n{question}",
                model=config.router_model,
                temperature=0.0,
                timeout=config.router_timeout_secs,
                max_tokens=200,
                response_format={"type": "json_object"},
                api_key_env=api_key_env,
            )
        )
    except Exception as exc:
        _log.warning("query_router_call_failed", error=str(exc))
        return _fallback_route()
    return _parse_route(raw)


def _route_ollama(config: AdaptiveRetrievalConfig) -> QueryRoute:
    """Placeholder Ollama branch — falls back until an ollama JSON-mode call is wired in.

    Keeping the dispatch table exhaustive (with ``assert_never``) means
    the openai/ollama Literal stays single-source-of-truth in
    :class:`AdaptiveRetrievalConfig`. When a real Ollama route is added,
    this function becomes the implementation. Until then it bypasses
    rather than silently mis-routing.
    """
    _log.info("query_router_ollama_not_implemented", model=config.router_model)
    return _fallback_route()


def _fallback_route() -> QueryRoute:
    """Standard default + ``routed=False`` warning marker."""
    return QueryRoute(
        retrieve=True,
        breadth="standard",
        rationale="router unavailable; defaulting to standard retrieval",
        routed=False,
    )


def _parse_route(raw: str) -> QueryRoute:
    """Parse the JSON the router LLM returned; never raises."""
    payload = _safe_json(raw)
    retrieve = payload.get("retrieve")
    breadth = payload.get("breadth")
    rationale = payload.get("rationale", "")

    if not isinstance(retrieve, bool):
        return _fallback_route()
    if breadth not in {"narrow", "standard", "broad"}:
        return _fallback_route()
    if not isinstance(rationale, str):
        rationale = ""
    return QueryRoute(
        retrieve=retrieve,
        breadth=breadth,
        rationale=rationale[:300],
        routed=True,
    )


def _safe_json(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError, TypeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def resolve_dense_top_k(
    *,
    breadth: RouteBreadth,
    config: AdaptiveRetrievalConfig,
    default_top_k: int,
) -> int:
    """Translate a breadth literal into a dense_top_k integer.

    The standard route keeps the existing ``retrieval.dense_top_k``;
    narrow / broad routes use their dedicated config values so they
    can be tuned independently per deployment.
    """
    if breadth == "narrow":
        return config.narrow_dense_top_k
    if breadth == "broad":
        return config.broad_dense_top_k
    if breadth == "standard":
        return default_top_k
    assert_never(breadth)
