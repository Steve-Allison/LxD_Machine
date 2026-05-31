#!/usr/bin/env bash
# PreToolUse:Edit|Write|MultiEdit hook — block edits to critical files.
#
# Exit 2 = block the tool call with reason on stderr.
# Exit 0 = allow.
#
# The protected set is the union of:
#   - Secrets and lockfiles that humans manage (`.env`, `.envrc`, `pixi.lock`)
#   - Runtime-managed binary state (SQLite DBs, LanceDB tables, .bak backups,
#     config.lock, ingest_snapshot.json — never hand-edit)
#   - Golden manifest tests (use `pixi run pytest --update-golden`, not Edit)
#
# We intentionally do NOT block edits to schema.py / _base_ddl.py / Plans/*.md
# because those have legitimate edit cases (new migrations, plan updates) —
# the agent should pause and confirm, not be hard-blocked.

set -uo pipefail

input="$(cat 2>/dev/null || true)"

tool_name="$(printf '%s' "$input" | jq -r '.tool_name // empty' 2>/dev/null || true)"
case "$tool_name" in
  Edit|Write|MultiEdit) ;;
  *) exit 0 ;;
esac

file_path="$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty' 2>/dev/null || true)"
[[ -z "$file_path" ]] && exit 0

project_root="${CLAUDE_PROJECT_DIR:-$(pwd)}"
rel_path="${file_path#"$project_root"/}"

# Patterns are evaluated as bash globs against the project-relative path.
protected_globs=(
    ".env"
    ".env.*"
    ".envrc"
    "pixi.lock"
    "data/openai/*.sqlite3"
    "data/openai/*.sqlite3.bak"
    "data/openai/*.db"
    "data/openai/lancedb/*"
    "data/openai/lancedb/**/*"
    "data/openai/config.lock"
    "data/openai/ingest_snapshot.json"
    "tests/golden/*.json"
)

for glob in "${protected_globs[@]}"; do
    # shellcheck disable=SC2053  # left-side glob match is intentional
    if [[ "$rel_path" == $glob ]]; then
        cat >&2 <<MSG
BLOCKED: '$rel_path' is a protected file.

  Reason: file is managed by tooling or contains secrets/binary state.
  Action: do not Edit/Write this file via Claude Code.

  If you genuinely need to change it:
    - .env / .envrc     → edit manually in your editor
    - pixi.lock         → run \`pixi update\` instead
    - *.sqlite3 / *.bak → use the store APIs, never raw edit
    - lancedb/          → managed by LanceDB; never raw edit
    - config.lock       → regenerated automatically by ingest
    - tests/golden/*    → refresh via \`pixi run pytest --update-golden\`
MSG
        exit 2
    fi
done

exit 0
