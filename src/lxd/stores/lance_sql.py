"""Safe helpers for building LanceDB ``where`` filter clauses.

LanceDB exposes a SQL-like filter language, but its Python API has no
parameter binding. Injecting user-provided strings directly into the filter
is a latent injection surface (chunk IDs, source paths, domain tags, etc.).

This module centralises escaping and ``IN``/equality clause construction so
every call site goes through a single, tested code path. Values containing
NUL bytes or newlines are rejected because LanceDB's parser cannot handle
them unambiguously and they almost always indicate corrupted input.
"""

from collections.abc import Iterable

_FORBIDDEN_CHARS: tuple[str, ...] = ("\x00", "\r", "\n")


def escape_string_literal(value: str) -> str:
    """Return ``value`` quoted for inclusion in a LanceDB ``where`` clause.

    Single quotes are doubled (SQL-standard escaping). Values containing NUL
    or newline characters are rejected rather than silently truncated.

    Args:
        value: Raw string to embed.

    Returns:
        Escaped string (without surrounding quotes).

    Raises:
        ValueError: If ``value`` contains forbidden control characters.
    """
    for ch in _FORBIDDEN_CHARS:
        if ch in value:
            raise ValueError(f"value contains forbidden control character: {ch!r}")
    return value.replace("'", "''")


def eq_clause(column: str, value: str) -> str:
    """Build ``column = '<escaped value>'`` for LanceDB filters.

    Args:
        column: Column name (must be a bare identifier).
        value: String literal to compare against.

    Returns:
        Safe filter fragment.

    Raises:
        ValueError: If ``column`` is not a valid identifier or ``value`` is
            rejected by :func:`escape_string_literal`.
    """
    _check_identifier(column)
    return f"{column} = '{escape_string_literal(value)}'"


def in_clause(column: str, values: Iterable[str]) -> str:
    """Build ``column IN ('a', 'b', ...)`` for LanceDB filters.

    Args:
        column: Column name (must be a bare identifier).
        values: String literals. May be an iterator; consumed once.

    Returns:
        Safe filter fragment.

    Raises:
        ValueError: If ``values`` is empty, ``column`` is invalid, or any
            value is rejected by :func:`escape_string_literal`.
    """
    _check_identifier(column)
    materialised = [f"'{escape_string_literal(v)}'" for v in values]
    if not materialised:
        raise ValueError("in_clause requires at least one value")
    return f"{column} IN ({', '.join(materialised)})"


def _check_identifier(name: str) -> None:
    if not name or not all(ch.isalnum() or ch == "_" for ch in name):
        raise ValueError(f"invalid LanceDB column identifier: {name!r}")
