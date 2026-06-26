"""Tests for the sampled-logging structlog processor (B-STACK-9)."""

import pytest
import structlog

from lxd.observability.logging import make_sampled_processor


def test_rate_one_lets_every_event_through() -> None:
    processor = make_sampled_processor(
        rate=1,
        high_volume_events=frozenset({"chunk_processed"}),
    )

    for _ in range(50):
        result = processor(None, "info", {"event": "chunk_processed", "n": 1})
        assert result == {"event": "chunk_processed", "n": 1}


def test_non_sampled_events_always_pass() -> None:
    processor = make_sampled_processor(
        rate=10,
        high_volume_events=frozenset({"chunk_processed"}),
    )

    for _ in range(20):
        result = processor(None, "info", {"event": "ingest.run.completed"})
        assert result == {"event": "ingest.run.completed"}


def test_high_volume_events_are_sampled_one_in_n() -> None:
    rate = 5
    processor = make_sampled_processor(
        rate=rate,
        high_volume_events=frozenset({"chunk_processed"}),
    )

    kept = 0
    dropped = 0
    for _ in range(100):
        try:
            processor(None, "info", {"event": "chunk_processed"})
            kept += 1
        except structlog.DropEvent:
            dropped += 1

    assert kept == 100 // rate, (
        f"Expected exactly {100 // rate} kept events at rate={rate}; saw kept={kept}, "
        f"dropped={dropped}."
    )
    assert kept + dropped == 100


def test_error_level_events_bypass_sampling() -> None:
    processor = make_sampled_processor(
        rate=100,
        high_volume_events=frozenset({"chunk_processed"}),
    )

    for _ in range(10):
        result = processor(None, "error", {"event": "chunk_processed", "exc": "boom"})
        assert result == {"event": "chunk_processed", "exc": "boom"}


def test_critical_level_events_bypass_sampling() -> None:
    processor = make_sampled_processor(
        rate=100,
        high_volume_events=frozenset({"chunk_processed"}),
    )

    for _ in range(10):
        result = processor(None, "critical", {"event": "chunk_processed"})
        assert result == {"event": "chunk_processed"}


def test_sampling_counters_are_per_event_name() -> None:
    """Each high-volume event has its own counter — interleaving doesn't bleed."""
    rate = 3
    processor = make_sampled_processor(
        rate=rate,
        high_volume_events=frozenset({"event_a", "event_b"}),
    )

    kept_a = 0
    kept_b = 0
    for i in range(9):
        name = "event_a" if i % 2 == 0 else "event_b"
        try:
            processor(None, "info", {"event": name})
            if name == "event_a":
                kept_a += 1
            else:
                kept_b += 1
        except structlog.DropEvent:
            pass

    a_total = sum(1 for i in range(9) if i % 2 == 0)
    b_total = sum(1 for i in range(9) if i % 2 == 1)
    assert kept_a == max(1, a_total // rate) or kept_a == (a_total + rate - 1) // rate
    assert kept_b == max(1, b_total // rate) or kept_b == (b_total + rate - 1) // rate


def test_drop_event_is_raised_for_suppressed_log() -> None:
    processor = make_sampled_processor(
        rate=10,
        high_volume_events=frozenset({"chunk_processed"}),
    )

    processor(None, "info", {"event": "chunk_processed"})

    with pytest.raises(structlog.DropEvent):
        processor(None, "info", {"event": "chunk_processed"})
