"""Shared SQL construction helpers for the SQLite store.

Responsibility:
    Centralise fiddly SQL-composition patterns that appear across the
    store layer so call sites do not reinvent them. Functions here never
    execute SQL directly; they return strings (placeholders) or tuples
    (placeholders + parameter list) that callers pass into
    :meth:`sqlite3.Connection.execute`.

Design boundary:
    Private to ``lxd.stores``. Callers must still parameterise values
    through SQLite placeholders; these helpers only build placeholder
    strings, never interpolate data.

Key constraints:
    * Empty collections raise ``ValueError`` rather than producing an
      always-false ``IN ()`` clause.
    * The helpers are deterministic and side-effect free.
"""

from __future__ import annotations

from collections.abc import Sequence


def in_clause(count: int) -> str:
    """Return an ``(?, ?, ?, ...)`` placeholder string with ``count`` slots.

    Args:
        count: Number of placeholders to emit. Must be positive.

    Returns:
        The placeholder fragment suitable for interpolation after ``IN``.

    Raises:
        ValueError: If ``count`` is not positive. SQLite's ``IN ()`` is
            syntactically invalid, so callers must filter empty collections
            at a higher level.
    """
    if count <= 0:
        raise ValueError("in_clause requires count >= 1")
    return "(" + ",".join("?" * count) + ")"


def in_clause_for(values: Sequence[object]) -> tuple[str, list[object]]:
    """Return ``(placeholders, params)`` for a non-empty ``values`` sequence.

    Args:
        values: Values to bind. Must be non-empty.

    Returns:
        Tuple ``(placeholder_fragment, params_list)`` where the fragment is
        ``(?,?,...,?)`` and ``params_list`` is a fresh list of ``values``.

    Raises:
        ValueError: If ``values`` is empty.
    """
    params = list(values)
    if not params:
        raise ValueError("in_clause_for requires a non-empty sequence")
    return in_clause(len(params)), params
