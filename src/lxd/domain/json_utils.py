"""Robust JSON extraction and sanitization helpers for LLM-generated output.

Standardises handling of:
- ``<think>...</think>`` and ``<reasoning>...</reasoning>`` blocks emitted by reasoning models (DeepSeek-R1, Qwen 2.5/3, etc.).
- Markdown code blocks (`````json ... `````).
- Conversational preamble / postamble surrounding structured JSON payloads.
- Trailing commas commonly emitted by smaller instruction-tuned models.
"""

import json
import re
from typing import Any, Final

_THINK_BLOCK_PATTERN: Final = re.compile(
    r"<(?:think|reasoning)>.*?</(?:think|reasoning)>",
    flags=re.DOTALL | re.IGNORECASE,
)

_CODE_BLOCK_PATTERN: Final = re.compile(
    r"```(?:json)?\s*([\s\S]*?)\s*```",
    flags=re.IGNORECASE,
)

_TRAILING_COMMA_PATTERN: Final = re.compile(r",\s*([\]}])")


def strip_thinking(text: str) -> str:
    """Remove ``<think>...</think>`` and ``<reasoning>...</reasoning>`` blocks.

    Uses a single non-greedy DOTALL regex pass to eliminate scratchpad
    reasoning blocks emitted by reasoning-capable LLMs.
    """
    if not text:
        return ""
    return _THINK_BLOCK_PATTERN.sub("", text).strip()


def _try_parse(candidate: str) -> Any | None:
    """Attempt json.loads on candidate, retrying with trailing-comma cleanup if needed."""
    candidate = candidate.strip()
    if not candidate:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError, TypeError:
        pass

    # Retry once after stripping trailing commas before closing braces/brackets
    cleaned = _TRAILING_COMMA_PATTERN.sub(r"\1", candidate)
    if cleaned != candidate:
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError, TypeError:
            pass
    return None


def extract_json_payload(raw: str, default: Any = None) -> Any:
    """Extract and parse any JSON payload from an LLM response string.

    Tries, in order:
    1. Direct parse after stripping reasoning tags.
    2. Parsing inside markdown code blocks (`````json ... `````).
    3. Parsing the outermost balanced object (``{...}``) or array (``[...]``).
    4. Fallback to ``default``.
    """
    if not raw:
        return default

    clean = strip_thinking(raw)
    if not clean:
        return default

    # 1. Direct parse
    parsed = _try_parse(clean)
    if parsed is not None:
        return parsed

    # 2. Markdown code blocks
    matches = _CODE_BLOCK_PATTERN.findall(clean)
    for block in matches:
        parsed = _try_parse(block)
        if parsed is not None:
            return parsed

    # 3. Outermost object or array
    first_brace = clean.find("{")
    last_brace = clean.rfind("}")
    first_bracket = clean.find("[")
    last_bracket = clean.rfind("]")

    candidates: list[str] = []
    # If object appears earlier or exclusively
    if first_brace != -1 and last_brace > first_brace:
        candidates.append(clean[first_brace : last_brace + 1])
    if first_bracket != -1 and last_bracket > first_bracket:
        candidates.append(clean[first_bracket : last_bracket + 1])

    # Prioritize the candidate that starts earlier in the text
    if (
        first_bracket != -1
        and first_brace != -1
        and first_bracket < first_brace
        and len(candidates) == 2
    ):
        candidates.reverse()

    for candidate in candidates:
        parsed = _try_parse(candidate)
        if parsed is not None:
            return parsed

    return default


def extract_json_dict(raw: str, default: dict[str, Any] | None = None) -> dict[str, Any]:
    """Extract and parse a JSON dictionary from an LLM response string."""
    fallback = default if default is not None else {}
    parsed = extract_json_payload(raw, default=fallback)
    return parsed if isinstance(parsed, dict) else fallback


def extract_json_list(raw: str, default: list[Any] | None = None) -> list[Any]:
    """Extract and parse a JSON list from an LLM response string."""
    fallback = default if default is not None else []
    parsed = extract_json_payload(raw, default=fallback)
    return parsed if isinstance(parsed, list) else fallback
