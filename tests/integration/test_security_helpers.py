"""Regression tests for the Wave 9 security helpers."""

from __future__ import annotations

import pytest

from lxd.observability.logging import scrub_secrets
from lxd.stores.lance_sql import eq_clause, escape_string_literal
from lxd.stores.lance_sql import in_clause as lance_in_clause
from lxd.stores.sql_helpers import in_clause, in_clause_for


def test_in_clause_formats_placeholders() -> None:
    """``in_clause(n)`` emits ``(?,?,?)`` slots that SQLite can parse."""
    assert in_clause(1) == "(?)"
    assert in_clause(3) == "(?,?,?)"


def test_in_clause_rejects_zero_length() -> None:
    """Empty collections raise rather than producing ``IN ()`` which is invalid SQL."""
    with pytest.raises(ValueError, match="count"):
        in_clause(0)


def test_in_clause_for_binds_values() -> None:
    """``in_clause_for`` returns both the fragment and a fresh params list."""
    fragment, params = in_clause_for(["a", "b", "c"])
    assert fragment == "(?,?,?)"
    assert params == ["a", "b", "c"]


def test_in_clause_for_rejects_empty_sequence() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        in_clause_for([])


def test_scrub_secrets_redacts_common_keys() -> None:
    """Known secret-shaped keys get their values replaced with ``***``."""
    payload = {
        "event": "auth.login",
        "api_key": "sk-live-123",
        "Authorization": "Bearer abc",
        "user": "alice",
    }

    out = scrub_secrets(None, "info", payload)

    assert out["api_key"] == "***"
    assert out["Authorization"] == "***"
    assert out["user"] == "alice"
    assert out["event"] == "auth.login"


def test_escape_string_literal_doubles_quotes() -> None:
    """Single quotes must be doubled to prevent filter injection."""
    assert escape_string_literal("o'connor") == "o''connor"


def test_escape_string_literal_rejects_control_characters() -> None:
    """NUL/newline bytes are rejected rather than silently truncated."""
    with pytest.raises(ValueError, match="forbidden control character"):
        escape_string_literal("line1\nline2")
    with pytest.raises(ValueError, match="forbidden control character"):
        escape_string_literal("has\x00nul")


def test_lance_eq_clause_composes_fragment() -> None:
    """``eq_clause`` returns a filter fragment safe for concatenation."""
    assert eq_clause("source_domain", "docs") == "source_domain = 'docs'"


def test_lance_eq_clause_rejects_bad_identifier() -> None:
    """Column names are restricted to bare identifiers."""
    with pytest.raises(ValueError, match="invalid LanceDB column identifier"):
        eq_clause("DROP TABLE", "x")


def test_lance_in_clause_joins_values() -> None:
    """``in_clause`` builds ``col IN ('a', 'b')`` with escaped values."""
    fragment = lance_in_clause("chunk_id", ["a", "b"])
    assert fragment == "chunk_id IN ('a', 'b')"


def test_lance_in_clause_rejects_empty_iterable() -> None:
    with pytest.raises(ValueError, match="at least one value"):
        lance_in_clause("chunk_id", [])


def test_scrub_secrets_recurses_into_nested_structures() -> None:
    """Nested dicts and lists of dicts are scrubbed too."""
    payload = {
        "config": {
            "openai_api_key": "sk-leak",
            "host": "localhost",
        },
        "clients": [
            {"name": "a", "auth_token": "tok"},
            {"name": "b"},
        ],
    }

    out = scrub_secrets(None, "info", payload)

    assert out["config"]["openai_api_key"] == "***"
    assert out["config"]["host"] == "localhost"
    assert out["clients"][0]["auth_token"] == "***"
    assert out["clients"][0]["name"] == "a"
    assert out["clients"][1] == {"name": "b"}
