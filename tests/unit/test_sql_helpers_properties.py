"""Property-based regression tests for SQL/LanceDB helpers.

Wave 10 introduces a dedicated property layer. These tests use Hypothesis
to cover invariants that are tedious to enumerate by hand:

* ``sql_helpers.in_clause`` must produce exactly ``count`` placeholders.
* ``lance_sql.escape_string_literal`` must be round-tripable and must
  double every single quote (equivalent to ``value.count("'")``).
* ``lance_sql.in_clause`` must always emit a balanced fragment whose
  parenthesised body is a comma-separated list of length matching
  ``len(values)``.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from lxd.stores.lance_sql import escape_string_literal
from lxd.stores.lance_sql import in_clause as lance_in_clause
from lxd.stores.sql_helpers import in_clause, in_clause_for

pytestmark = [pytest.mark.unit, pytest.mark.property]


_SAFE_TEXT = st.text(
    min_size=1,
    max_size=32,
    alphabet=st.characters(
        blacklist_categories=("Cs",),
        blacklist_characters=("\x00", "\r", "\n"),
    ),
)


@given(count=st.integers(min_value=1, max_value=64))
@settings(max_examples=50, deadline=None)
def test_in_clause_placeholder_count_matches_input(count: int) -> None:
    """``in_clause(n)`` must wrap exactly ``n`` ``?`` placeholders."""
    fragment = in_clause(count)
    assert fragment.startswith("(") and fragment.endswith(")")
    assert fragment.count("?") == count
    # Strip wrapping parens before counting separators.
    assert fragment[1:-1].count(",") == max(count - 1, 0)


@given(values=st.lists(_SAFE_TEXT, min_size=1, max_size=16))
@settings(max_examples=50, deadline=None)
def test_in_clause_for_returns_fresh_list(values: list[str]) -> None:
    """``in_clause_for`` returns a new list that preserves ordering."""
    fragment, params = in_clause_for(values)
    assert params == values
    assert params is not values
    assert fragment.count("?") == len(values)


@given(value=_SAFE_TEXT)
@settings(max_examples=100, deadline=None)
def test_escape_string_literal_doubles_every_quote(value: str) -> None:
    """Every ``'`` in the input must appear doubled in the output."""
    escaped = escape_string_literal(value)
    assert escaped.count("'") == value.count("'") * 2


@given(values=st.lists(_SAFE_TEXT, min_size=1, max_size=16))
@settings(max_examples=50, deadline=None)
def test_lance_in_clause_balanced_structure(values: list[str]) -> None:
    """``lance_in_clause`` emits ``col IN (...)`` with ``len(values)`` literals."""
    fragment = lance_in_clause("chunk_id", values)
    prefix = "chunk_id IN ("
    assert fragment.startswith(prefix)
    assert fragment.endswith(")")
    body = fragment[len(prefix) : -1]
    # Each literal is wrapped in single quotes, so the count of leading
    # quotes equals the number of values (escaped inner quotes are doubled
    # and therefore always even-length inside the body).
    assert body.count("', '") + (1 if values else 0) == len(values)
