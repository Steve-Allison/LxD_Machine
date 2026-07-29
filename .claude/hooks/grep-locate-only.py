#!/usr/bin/env python3
"""PreToolUse(Grep) — permit Grep as an INDEX, never as a reader.

Why this exists: the failure mode is not that `grep` exists, it is that grep
returns matched LINES — a keyhole view the model can quote and reason from
without ever opening the file. Remove the content and the keyhole is
mechanically impossible rather than discouraged.

Advisory text does not work here: injected `additionalContext` nudges are
routinely ignored under task pressure. This guard is enforcing.

Allowed (content-free — you learn WHERE, never WHAT):
  * output_mode: "files_with_matches"  -> paths only
  * output_mode: "count"               -> paths + a number

Denied:
  * output_mode: "content"             -> matched source lines
  * output_mode absent                 -> fail CLOSED; state the mode you want

The doctrine is unchanged: locate freely, but you may not learn anything about
a file without reading it in full. Grep tells you which file to open. Read
opens it.

Exit 2 = block with the reason on stderr. Exit 0 = allow.
Fails OPEN on internal error only: a broken guard must never wedge a session.
A missing/unknown output_mode is a POLICY decision, not an error — it blocks.
"""

from __future__ import annotations

import json
import sys


CONTENT_FREE = {"files_with_matches", "count"}

REASON = """BLOCKED: Grep may locate, but may not read.

output_mode={mode!r} returns matched source lines — a partial view of a file
you have not opened. Wrong causal claims tend to come from exactly that
substitution: quoting a matched line instead of reading the file it came from.

Use Grep as an index, then read what it points at:
  output_mode="files_with_matches"   paths only  (which files mention this?)
  output_mode="count"                paths + counts
Then open the file with Read (whole file, tracked).
"""


def main() -> int:
    try:
        data = json.load(sys.stdin)
        tool_input = data.get("tool_input") or {}
    except Exception:  # noqa: BLE001 - deliberately broad: fail open, never wedge a session
        return 0

    mode = tool_input.get("output_mode")

    if isinstance(mode, str) and mode in CONTENT_FREE:
        return 0

    sys.stderr.write(REASON.format(mode=mode))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
