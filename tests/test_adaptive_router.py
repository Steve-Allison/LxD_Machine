"""Tests for the adaptive retrieval router.

The router's contract is "never raise; always return a usable
:class:`QueryRoute`". These tests pin the parser invariants and the
breadth-to-top_k translation. The actual LLM call is not exercised in
CI — it's covered by the integration smoke ``pixi run mcp`` + a
manual route.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from lxd.retrieval.router import (
    QueryRoute,
    _fallback_route,
    _parse_route,
    resolve_dense_top_k,
)
from lxd.settings.models import AdaptiveRetrievalConfig

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
    assert "router unavailable" in fallback.rationale.lower()
