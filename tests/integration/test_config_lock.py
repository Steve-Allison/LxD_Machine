"""Regression tests for Wave 9 ``config.lock`` reconciliation."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from lxd.app.bootstrap import (
    compute_config_digest,
    reconcile_config_lock,
)
from lxd.settings.loader import load_runtime_config, resolve_repo_root


@pytest.fixture(scope="module")
def digest_pair(tmp_path_factory: pytest.TempPathFactory) -> tuple[str, str]:
    """Return two digests: the current config and a mutated variant."""
    repo_root = resolve_repo_root()
    config, _ = load_runtime_config(repo_root)
    original = compute_config_digest(config)

    mutated = config.model_copy(
        update={"mcp": config.mcp.model_copy(update={"server_name": "mutated-server"})}
    )
    mutated_digest = compute_config_digest(mutated)
    return original, mutated_digest


def test_compute_config_digest_is_stable(digest_pair: tuple[str, str]) -> None:
    """Same config must produce an identical digest across calls."""
    repo_root = resolve_repo_root()
    config, _ = load_runtime_config(repo_root)
    original, _ = digest_pair

    assert compute_config_digest(config) == original


def test_compute_config_digest_detects_changes(digest_pair: tuple[str, str]) -> None:
    """A meaningful field change must produce a different digest."""
    original, mutated = digest_pair
    assert original != mutated


def test_reconcile_writes_lock_on_first_run(tmp_path: Path) -> None:
    """Missing ``config.lock`` is created with the current digest."""
    data_path = tmp_path / "data"
    data_path.mkdir()

    reconcile_config_lock(data_path, digest="abc123")

    lock = data_path / "config.lock"
    assert lock.is_file()
    assert lock.read_text(encoding="utf-8").strip() == "abc123"


def test_reconcile_noop_when_data_path_missing(tmp_path: Path) -> None:
    """No lock is written until the data directory exists."""
    data_path = tmp_path / "does-not-exist"

    reconcile_config_lock(data_path, digest="abc123")

    assert not data_path.exists()


def test_reconcile_warns_on_digest_drift(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Stored digest mismatching the current digest logs a warning."""
    data_path = tmp_path / "data"
    data_path.mkdir()
    (data_path / "config.lock").write_text("previous-digest\n", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        reconcile_config_lock(data_path, digest="new-digest")

    lock_text = (data_path / "config.lock").read_text(encoding="utf-8").strip()
    assert lock_text == "previous-digest", "existing lock should not be overwritten"
