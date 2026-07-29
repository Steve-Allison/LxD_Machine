#!/usr/bin/env python3
"""PreToolUse(Bash) — deny shell text tools when they are READING A FILE.

Why this exists: reading a file through `grep`/`sed`/`head`/`cat` returns a
keyhole view that is then mistaken for having read the file. The Edit tool
already refuses to touch an unread file — this is the same gate for reading.

Allowed, deliberately:
  * piped filtering of COMMAND OUTPUT — `git status | grep M`, `pytest | tail -3`
    (the tool is consuming stdin, not opening a file)
  * heredocs, and any invocation with no file operand

Denied:
  * `cat config/render.yaml`, `sed -n '1,40p' file`, `head -20 file`,
    `grep -rn "x" src/` — use the Read tool (whole file, tracked).

Exit 2 = block with the reason on stderr. Exit 0 = allow.
Fails OPEN on any internal error: a broken guard must never wedge a session.
"""

from __future__ import annotations

import json
import re
import shlex
import sys
from pathlib import Path


# Text tools that can take a file operand.
TOOLS = {
    "grep",
    "egrep",
    "fgrep",
    "rg",
    "ag",
    "ack",
    "sed",
    "awk",
    "head",
    "tail",
    "cat",
    "cut",
    "nl",
    "less",
    "more",
}

# Tools whose FIRST non-flag operand is a pattern/script, not a path.
PATTERN_FIRST = {"grep", "egrep", "fgrep", "rg", "ag", "ack", "sed", "awk"}

# Search tools that have a CONTENT-FREE mode. In that mode they are an index —
# they report which files match, never what the matching lines say — so they
# cannot produce the keyhole view this guard exists to prevent. Same carve-out
# the Grep tool gets in grep-locate-only.py, and same reasoning.
GREP_FAMILY = {"grep", "egrep", "fgrep", "rg", "ag", "ack"}

# Long flags that suppress matched lines entirely.
CONTENT_FREE_FLAGS = {
    "-l",
    "-L",
    "-c",
    "--files-with-matches",
    "--files-without-match",
    "--count",
    "--count-matches",
    "--files",
}

# Letters that suppress content when bundled (-rl, -rc …).
CONTENT_FREE_LETTERS = set("lLc")
# Letters that RE-INTRODUCE content and therefore void the carve-out:
# -A/-B/-C context, -o only-matching.
CONTENT_RESTORING_LETTERS = set("ABCo")

# Long forms of the same content-restoring options.
CONTENT_RESTORING_FLAGS = {
    "-A",
    "-B",
    "-C",
    "-o",
    "--after-context",
    "--before-context",
    "--context",
    "--only-matching",
    "--passthru",
    "--passthrough",
}


def is_content_free(argv: list[str]) -> bool:
    """True when this invocation cannot print a line of file content.

    Every argument is inspected before deciding: a suppressing flag does not
    win just because it came first. `rg -l -C3 foo src/` is rejected even
    though ripgrep would honour -l, because reasoning about flag precedence
    across grep/rg/ag/ack variants is exactly where a silent hole would open.
    Over-blocking costs one retry; under-blocking reopens the keyhole.
    """
    has_free = False
    has_restore = False

    for arg in argv[1:]:
        if arg in CONTENT_FREE_FLAGS:
            has_free = True
            continue
        if arg in CONTENT_RESTORING_FLAGS:
            has_restore = True
            continue
        # Context flags carrying an inline count: -C3, -A2, -B1.
        if re.fullmatch(r"-[ABC]\d+", arg):
            has_restore = True
            continue
        # Bundled short flags: -rl, -rc, -rn. Long flags are handled above.
        if re.fullmatch(r"-[A-Za-z]+", arg):
            letters = set(arg[1:])
            if letters & CONTENT_RESTORING_LETTERS:
                has_restore = True
            if letters & CONTENT_FREE_LETTERS:
                has_free = True

    return has_free and not has_restore


# Separators that start a NEW command (stdin not inherited from a pipe).
FRESH = {"&&", "||", ";", "&"}

PATHISH = re.compile(
    r"""\.(py|json|ya?ml|toml|md|pdl|txt|sh|cfg|ini|lock|sqlite3?|csv|xml|html?|js|ts)$""",
    re.IGNORECASE,
)


def looks_like_path(token: str) -> bool:
    if token.startswith("-"):
        return False
    if Path(token).exists():
        return True
    return "/" in token or bool(PATHISH.search(token))


def offending_tool(command: str) -> tuple[str, str] | None:  # noqa: PLR0912
    """Return (tool, path) for the first file-reading invocation, else None.

    Deliberately one function: it walks every shell segment in order,
    tracking pipe state as it goes, and splitting that into smaller
    functions would scatter state that needs to stay together.
    """
    if "<<" in command:  # heredoc: input, not a file read
        return None

    # Split while keeping separators so we know which segments get piped stdin.
    parts = re.split(r"(\|\||&&|\||;|&)", command)
    piped = False
    for part in parts:
        token = part.strip()
        if not token:
            continue
        if token == "|":
            piped = True
            continue
        if token in FRESH:
            piped = False
            continue
        if piped:  # consuming stdin from the previous command — legitimate
            piped = False
            continue

        try:
            argv = shlex.split(token)
        except ValueError:
            argv = token.split()
        # Drop leading env assignments and wrappers.
        while argv and ("=" in argv[0].split("/")[-1][:1] or argv[0] in {"sudo", "command", "time"}):
            argv = argv[1:]
        while argv and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", argv[0]):
            argv = argv[1:]
        if not argv:
            continue

        tool = Path(argv[0]).name
        if tool not in TOOLS:
            continue

        # An index, not a read: locating files is allowed, quoting them is not.
        if tool in GREP_FAMILY and is_content_free(argv):
            continue

        operands = [a for a in argv[1:] if not a.startswith("-")]
        if tool in PATTERN_FIRST and operands:
            operands = operands[1:]  # first operand is the pattern/script
        for operand in operands:
            if looks_like_path(operand):
                return tool, operand
    return None


def main() -> int:
    try:
        data = json.load(sys.stdin)
        command = (data.get("tool_input") or {}).get("command") or ""
    except Exception:  # noqa: BLE001 - fail open: a broken guard must not wedge a session
        return 0

    if not command:
        return 0

    try:
        hit = offending_tool(command)
    except Exception:  # noqa: BLE001 - fail open: a broken guard must not wedge a session
        return 0

    if not hit:
        return 0

    tool, path = hit
    locate = (
        f"To LOCATE without reading, use a content-free mode: "
        f"`{tool} -l <pattern> <path>` (files only) or `{tool} -c` (counts). "
        f"Then Read what it points at.\n"
        if tool in GREP_FAMILY
        else ""
    )
    sys.stderr.write(
        f"BLOCKED: `{tool}` is reading the file {path!r}.\n"
        f"A partial view is not a read. Use the Read tool on {path!r} (whole file, "
        f"tracked).\n"
        f"{locate}"
        f"Piping command OUTPUT through {tool} (e.g. `cmd | {tool} ...`) is still allowed.\n"
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
