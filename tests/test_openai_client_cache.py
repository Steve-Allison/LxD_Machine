"""Tests for the OpenAI client process-wide cache."""

from __future__ import annotations

from collections.abc import Generator

import pytest

from lxd.ingest.embedder import get_openai_client, reset_openai_client_cache


@pytest.fixture(autouse=True)
def _reset_client_cache() -> Generator[None]:
    reset_openai_client_cache()
    yield
    reset_openai_client_cache()


def test_get_openai_client_returns_same_instance_for_same_key() -> None:
    """Two lookups for the same api_key yield the same OpenAI client object."""
    first = get_openai_client("test-key-1")
    second = get_openai_client("test-key-1")
    assert first is second, (
        "get_openai_client should cache by api_key — two calls returned different objects."
    )


def test_get_openai_client_returns_distinct_instances_for_different_keys() -> None:
    """Different api_keys produce different OpenAI clients."""
    one = get_openai_client("test-key-A")
    two = get_openai_client("test-key-B")
    assert one is not two, "Distinct api_keys should produce distinct OpenAI clients."


def test_reset_openai_client_cache_invalidates_prior_clients() -> None:
    """After reset, the same api_key constructs a fresh client."""
    before = get_openai_client("test-key-X")
    reset_openai_client_cache()
    after = get_openai_client("test-key-X")
    assert before is not after, "After reset, the cache should hand out a fresh client."
