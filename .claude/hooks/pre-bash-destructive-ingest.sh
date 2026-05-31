#!/usr/bin/env bash
# PreToolUse:Bash hook — gate destructive ingest / graph / data commands.
#
# This hook does NOT block on its own — exit 2 would block, exit 0 allows.
# Instead it emits a *system warning* on stderr (exit 0) reminding the agent
# of the preflight-is-a-gate rule. The warning fires in the agent's context
# and gives it a chance to stop and re-confirm with the user.
#
# Why warn instead of block: legitimate use cases exist (the user explicitly
# approved the run, an automated test invokes the command). A hard block
# would force --no-verify-style workarounds. A warning preserves discipline
# without breaking the legitimate path.

set -uo pipefail

input="$(cat 2>/dev/null || true)"
tool_name="$(printf '%s' "$input" | jq -r '.tool_name // empty' 2>/dev/null || true)"
[[ "$tool_name" == "Bash" ]] || exit 0

command_text="$(printf '%s' "$input" | jq -r '.tool_input.command // empty' 2>/dev/null || true)"
[[ -z "$command_text" ]] && exit 0

# Patterns that warrant a gate-reminder. Each is a regex matched
# case-insensitively against the command text.
patterns=(
    'pixi run ingest( |.*--full)'
    'pixi run build-graph( |.*--full)'
    'rm -rf? .*data/'
    'rm -rf? .*lancedb'
    'rm -rf? .*\.sqlite3'
    'DROP TABLE'
    'TRUNCATE TABLE'
    'sqlite3.*\.execute.*DROP'
)

matched=""
for pat in "${patterns[@]}"; do
    if printf '%s' "$command_text" | grep -qiE "$pat"; then
        matched="$pat"
        break
    fi
done

[[ -z "$matched" ]] && exit 0

# Emit a structured warning. The agent sees this on its next turn.
cat >&2 <<MSG
⚠ DESTRUCTIVE-INGEST GATE

  Command:  $command_text
  Matched:  $matched

  This command falls under the preflight-is-a-gate rule
  (.claude/rules/ingest-discipline.md). Per ~/.claude/CLAUDE.md §7, destructive
  operations need explicit in-session confirmation EVERY TIME.

  Confirm before running:
    1. Did the user approve THIS specific command in THIS session?
    2. If preflight was the prior step — was its output reviewed AND user
       said go AFTER seeing it?  (Not "yes to preflight then ingest" in
       the same breath.)
    3. Is the rollback path obvious if this destroys the wrong state?

  If any answer is no — STOP. Surface the situation to the user and wait.
MSG

# Exit 0 = warn but allow. The agent decides what to do next.
exit 0
