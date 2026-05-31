#!/usr/bin/env bash
# SessionStart hook: orient the agent on project state.
#
# Reports (one short block) at session start so the agent doesn't have to
# discover state by exploration:
#   - git: branch, dirty file count, recent commits
#   - corpus: configured path, presence
#   - stores: SQLite + LanceDB presence and sizes
#   - knowledge graph: build status (via stored metadata, no DB call)
#   - .env presence (never the value)
#
# Always exits 0. Output goes to stdout for the session. No tool calls,
# no commands beyond filesystem-stat / git plumbing.

set -uo pipefail

project_root="${CLAUDE_PROJECT_DIR:-$(pwd)}"
cd "$project_root" || exit 0

print() { printf '%s\n' "$*"; }

print "── LxD Machine session-start ──"

# --- Git --------------------------------------------------------------------
if git -C "$project_root" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    branch="$(git -C "$project_root" branch --show-current 2>/dev/null || echo '(detached)')"
    dirty_count="$(git -C "$project_root" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
    behind_ahead="$(git -C "$project_root" rev-list --left-right --count HEAD...@{u} 2>/dev/null || echo '0	0')"
    ahead="${behind_ahead%%	*}"
    behind="${behind_ahead##*	}"
    print "  git:     branch=${branch}  dirty=${dirty_count} files  ahead=${ahead}  behind=${behind}"
    recent="$(git -C "$project_root" log --oneline -3 2>/dev/null | sed 's/^/             /')"
    [[ -n "$recent" ]] && print "$recent"
else
    print "  git:     not a repository"
fi

# --- Corpus -----------------------------------------------------------------
corpus_path="$(awk -F': *' '/^[[:space:]]*corpus_path:/{print $2; exit}' config.yaml 2>/dev/null || true)"
if [[ -n "$corpus_path" ]]; then
    if [[ -d "$corpus_path" ]]; then
        file_count="$(find "$corpus_path" -maxdepth 1 -type f \( -name '*.md' -o -name '*.docling.json' \) 2>/dev/null | wc -l | tr -d ' ')"
        print "  corpus:  ${corpus_path} (resolves, ~${file_count} text files at top level)"
    else
        print "  corpus:  ${corpus_path}  ← DOES NOT EXIST"
    fi
else
    print "  corpus:  (no corpus_path in config.yaml)"
fi

# --- Stores -----------------------------------------------------------------
sqlite_path="data/openai/lxd.sqlite3"
lancedb_path="data/openai/lancedb"
if [[ -f "$sqlite_path" ]]; then
    sqlite_size="$(du -h "$sqlite_path" 2>/dev/null | cut -f1)"
    print "  sqlite:  ${sqlite_path}  (${sqlite_size})"
else
    print "  sqlite:  ${sqlite_path}  ← MISSING (run ingest)"
fi
if [[ -d "$lancedb_path" ]]; then
    lancedb_size="$(du -sh "$lancedb_path" 2>/dev/null | cut -f1)"
    table_count="$(find "$lancedb_path" -maxdepth 1 -type d -name '*.lance' 2>/dev/null | wc -l | tr -d ' ')"
    print "  lancedb: ${lancedb_path}  (${lancedb_size}, ${table_count} tables)"
else
    print "  lancedb: ${lancedb_path}  ← MISSING (run ingest)"
fi

# --- .env -------------------------------------------------------------------
if [[ -f .env ]]; then
    print "  .env:    present"
else
    print "  .env:    MISSING — OPENAI_API_KEY must be set before any LLM ingest"
fi

# --- Reminders --------------------------------------------------------------
print "  rules:   ingest-discipline / stores-and-paths / mandatory-features / no-pull-requests"
print "  gates:   pixi run preflight is BEFORE pixi run ingest — never auto-chain"

exit 0
