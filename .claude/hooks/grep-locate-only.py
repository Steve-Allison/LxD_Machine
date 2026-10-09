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
  * output_mode absent                 -> Grep's own default, files_with_matches

Denied:
  * output_mode: "content"             -> matched source lines

The doctrine is unchanged: locate freely, but you may not learn anything about
a file without reading it in full. Grep tells you which file to open. Read
opens it.

Exit 2 = block with the reason on stderr. Exit 0 = allow.
Unreadable hook input allows the call: a broken guard must never wedge a session.
"""

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
    except (json.JSONDecodeError, UnicodeDecodeError):
        return 0
    tool_input = data.get("tool_input") if isinstance(data, dict) else None
    if not isinstance(tool_input, dict):
        return 0

    mode = tool_input.get("output_mode")

    if mode is None or mode in CONTENT_FREE:
        return 0

    sys.stderr.write(REASON.format(mode=mode))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
