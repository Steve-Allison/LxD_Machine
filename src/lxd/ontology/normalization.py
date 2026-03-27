"""Normalize ontology and query text for matching."""

from __future__ import annotations

import re

_WHITESPACE_RE = re.compile(r"\s+")
_QUOTE_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": "'",
        "\u201d": "'",
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
    }
)


def normalize_match_text(text: str) -> str:
    """Normalize text for robust matcher lookups.

    Args:
        text: Text to normalize for matching.

    Returns:
        Casefolded, de-quoted, single-spaced text.
    """
    normalized = text.casefold().translate(_QUOTE_TRANSLATION)
    return _WHITESPACE_RE.sub(" ", normalized).strip()
