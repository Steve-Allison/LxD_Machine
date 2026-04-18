"""Shared pytest configuration hooks for the LxD test suite."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register custom CLI flags used by golden-file regression tests."""
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Rewrite golden manifest files instead of asserting equality.",
    )
