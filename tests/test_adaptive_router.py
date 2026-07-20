"""Tests for the adaptive retrieval router.

The router's contract is "never raise; always return a usable
:class:`QueryRoute`". These tests pin the parser invariants and the
breadth-to-top_k translation. The actual LLM call is not exercised in
CI — it's covered by the integration smoke ``pixi run mcp`` + a
manual route.
"""

import pytest
from pydantic import ValidationError

from lxd.retrieval import router as _router_module
from lxd.retrieval.router import QueryRoute, resolve_dense_top_k, route_query
from lxd.settings.models import AdaptiveRetrievalConfig

_fallback_route = _router_module._fallback_route  # pyright: ignore[reportPrivateUsage]
_heuristic_route = _router_module._heuristic_route  # pyright: ignore[reportPrivateUsage]
_parse_route = _router_module._parse_route  # pyright: ignore[reportPrivateUsage]

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# QueryRoute Pydantic invariants
# ---------------------------------------------------------------------------


def test_query_route_is_frozen() -> None:
    route = QueryRoute(retrieve=True, breadth="standard")
    with pytest.raises(ValidationError):
        route.retrieve = False  # type: ignore[misc]


def test_query_route_rejects_extra_keys() -> None:
    with pytest.raises(ValidationError):
        QueryRoute.model_validate({"retrieve": True, "breadth": "standard", "extra": 1})


def test_query_route_rejects_invalid_breadth() -> None:
    with pytest.raises(ValidationError):
        QueryRoute.model_validate({"retrieve": True, "breadth": "huge"})


def test_query_route_default_breadth_is_standard() -> None:
    route = QueryRoute(retrieve=True)
    assert route.breadth == "standard"


def test_query_route_routed_defaults_true() -> None:
    route = QueryRoute(retrieve=True)
    assert route.routed is True


def test_query_route_router_path_defaults_llm() -> None:
    route = QueryRoute(retrieve=True)
    assert route.router_path == "llm"


def test_query_route_rejects_invalid_router_path() -> None:
    with pytest.raises(ValidationError):
        QueryRoute.model_validate({"retrieve": True, "router_path": "magic"})


# ---------------------------------------------------------------------------
# _parse_route — accepts well-formed JSON, falls back gracefully
# ---------------------------------------------------------------------------


def test_parse_route_accepts_well_formed_json() -> None:
    raw = '{"retrieve": true, "breadth": "narrow", "rationale": "specific concept"}'
    route = _parse_route(raw)
    assert route.retrieve is True
    assert route.breadth == "narrow"
    assert route.rationale == "specific concept"
    assert route.routed is True
    assert route.router_path == "llm"


def test_parse_route_falls_back_on_invalid_json() -> None:
    route = _parse_route("not valid json")
    assert route == _fallback_route()
    assert route.routed is False


def test_parse_route_falls_back_when_retrieve_missing() -> None:
    raw = '{"breadth": "narrow"}'
    route = _parse_route(raw)
    assert route == _fallback_route()


def test_parse_route_falls_back_when_retrieve_not_bool() -> None:
    raw = '{"retrieve": "yes", "breadth": "narrow"}'
    route = _parse_route(raw)
    assert route == _fallback_route()


def test_parse_route_falls_back_when_breadth_unknown() -> None:
    raw = '{"retrieve": true, "breadth": "extreme"}'
    route = _parse_route(raw)
    assert route == _fallback_route()


def test_parse_route_truncates_long_rationale() -> None:
    long_rationale = "x" * 1000
    raw = '{"retrieve": true, "breadth": "standard", "rationale": "' + long_rationale + '"}'
    route = _parse_route(raw)
    assert len(route.rationale) == 300


def test_parse_route_accepts_skip_decision() -> None:
    raw = '{"retrieve": false, "breadth": "standard", "rationale": "greeting"}'
    route = _parse_route(raw)
    assert route.retrieve is False
    assert route.rationale == "greeting"


def test_parse_route_handles_array_payload_as_fallback() -> None:
    """JSON arrays are valid JSON but not the expected object shape."""
    route = _parse_route("[1, 2, 3]")
    assert route == _fallback_route()


# ---------------------------------------------------------------------------
# resolve_dense_top_k — breadth → integer translation
# ---------------------------------------------------------------------------


def _make_adaptive_config(
    *,
    narrow: int = 8,
    broad: int = 40,
) -> AdaptiveRetrievalConfig:
    return AdaptiveRetrievalConfig(
        narrow_dense_top_k=narrow,
        broad_dense_top_k=broad,
    )


def test_resolve_dense_top_k_narrow_uses_dedicated_setting() -> None:
    config = _make_adaptive_config(narrow=5)
    assert resolve_dense_top_k(breadth="narrow", config=config, default_top_k=20) == 5


def test_resolve_dense_top_k_broad_uses_dedicated_setting() -> None:
    config = _make_adaptive_config(broad=50)
    assert resolve_dense_top_k(breadth="broad", config=config, default_top_k=20) == 50


def test_resolve_dense_top_k_standard_uses_caller_default() -> None:
    config = _make_adaptive_config(narrow=5, broad=50)
    # Standard route does NOT override the retrieval-config default.
    assert resolve_dense_top_k(breadth="standard", config=config, default_top_k=20) == 20


# ---------------------------------------------------------------------------
# AdaptiveRetrievalConfig invariants
# ---------------------------------------------------------------------------


def test_adaptive_config_defaults() -> None:
    config = AdaptiveRetrievalConfig()
    assert config.router_backend == "openai"
    assert config.router_model == "gpt-4o-mini"
    assert config.router_timeout_secs == 15.0
    assert config.narrow_dense_top_k == 8
    assert config.broad_dense_top_k == 40


def test_adaptive_config_rejects_zero_top_k() -> None:
    with pytest.raises(ValidationError):
        AdaptiveRetrievalConfig(narrow_dense_top_k=0)
    with pytest.raises(ValidationError):
        AdaptiveRetrievalConfig(broad_dense_top_k=0)


def test_adaptive_config_rejects_top_k_above_200() -> None:
    with pytest.raises(ValidationError):
        AdaptiveRetrievalConfig(broad_dense_top_k=201)


def test_adaptive_config_rejects_zero_timeout() -> None:
    with pytest.raises(ValidationError):
        AdaptiveRetrievalConfig(router_timeout_secs=0)


def test_adaptive_config_rejects_unknown_backend() -> None:
    with pytest.raises(ValidationError):
        AdaptiveRetrievalConfig(router_backend="azure")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Fallback route invariant
# ---------------------------------------------------------------------------


def test_fallback_route_is_safe_default() -> None:
    fallback = _fallback_route()
    # Critically: fallback must NOT short-circuit retrieval — bypassing
    # the corpus on a router glitch would be worse than a slow query.
    assert fallback.retrieve is True
    assert fallback.breadth == "standard"
    assert fallback.routed is False
    assert fallback.router_path == "fallback"
    assert "router unavailable" in fallback.rationale.lower()


# ---------------------------------------------------------------------------
# _heuristic_route — cheap pre-router pattern matching
# ---------------------------------------------------------------------------


def test_heuristic_route_empty_question_returns_fallback_shape() -> None:
    route = _heuristic_route("   ")
    assert route == _fallback_route()


@pytest.mark.parametrize(
    "question",
    [
        "hi",
        "hello there",
        "thanks!",
        "thank you",
        "what can you do?",
        "how does this work?",
        "who built you?",
    ],
)
def test_heuristic_route_greeting_or_meta_skips_retrieval(question: str) -> None:
    route = _heuristic_route(question)
    assert route is not None
    assert route.retrieve is False
    assert route.breadth == "standard"
    assert route.router_path == "heuristic"


@pytest.mark.parametrize(
    "question",
    [
        "What is scaffolding?",
        "What's backward design?",
        "Define cognitive load",
        "Who coined the term andragogy?",
    ],
)
def test_heuristic_route_short_factual_lookup_is_narrow(question: str) -> None:
    route = _heuristic_route(question)
    assert route is not None
    assert route.retrieve is True
    assert route.breadth == "narrow"
    assert route.router_path == "heuristic"


def test_heuristic_route_factual_prefix_but_long_question_falls_through() -> None:
    """A 'what is' opener attached to a long, multi-clause question is not
    a clean single-concept lookup — defer to the LLM router instead of
    forcing narrow breadth."""
    question = (
        "What is the difference between backward design, ADDIE, and the "
        "Kirkpatrick model when applied to a blended adult-learner programme?"
    )
    assert _heuristic_route(question) is None


@pytest.mark.parametrize(
    "question",
    [
        "Compare ADDIE and SAM",
        "ADDIE versus SAM",
        "backward design vs forward design",
        "Which models address adult learners?",
        "survey of instructional design frameworks",
        "trends across the instructional design field",
    ],
)
def test_heuristic_route_survey_cue_is_broad(question: str) -> None:
    route = _heuristic_route(question)
    assert route is not None
    assert route.retrieve is True
    assert route.breadth == "broad"
    assert route.router_path == "heuristic"


def test_heuristic_route_returns_none_for_unclear_question() -> None:
    """A question that matches none of the heuristic buckets falls through
    to the LLM router."""
    assert _heuristic_route("How should I sequence a module on feedback loops?") is None


# ---------------------------------------------------------------------------
# route_query — heuristic short-circuit vs LLM fall-through
# ---------------------------------------------------------------------------


def test_route_query_uses_heuristic_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    config = AdaptiveRetrievalConfig(heuristic_router_enabled=True)

    def _fail_if_called(*_args: object, **_kwargs: object) -> QueryRoute:
        raise AssertionError("LLM router should not be called for a heuristic match.")

    monkeypatch.setattr(_router_module, "_route_openai", _fail_if_called)
    route = route_query(question="hello", config=config)
    assert route.router_path == "heuristic"
    assert route.retrieve is False


def test_route_query_falls_through_to_llm_when_heuristic_declines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = AdaptiveRetrievalConfig(heuristic_router_enabled=True)
    called = {"count": 0}

    def _fake_route_openai(**_kwargs: object) -> QueryRoute:
        called["count"] += 1
        return QueryRoute(retrieve=True, breadth="standard", router_path="llm")

    monkeypatch.setattr(_router_module, "_route_openai", _fake_route_openai)
    route = route_query(
        question="How should I sequence a module on feedback loops?", config=config
    )
    assert called["count"] == 1
    assert route.router_path == "llm"


def test_route_query_skips_heuristic_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    config = AdaptiveRetrievalConfig(heuristic_router_enabled=False)
    called = {"count": 0}

    def _fake_route_openai(**_kwargs: object) -> QueryRoute:
        called["count"] += 1
        return QueryRoute(retrieve=True, breadth="standard", router_path="llm")

    monkeypatch.setattr(_router_module, "_route_openai", _fake_route_openai)
    route_query(question="hello", config=config)
    assert called["count"] == 1
