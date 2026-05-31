#!/usr/bin/env bash
# LxD Machine statusline — model, branch, store sizes, graph version, .env state.
#
# Reads stdin JSON from Claude Code, emits a single status line on stdout.
# Always exits 0 — never blocks the prompt.

set -uo pipefail

input="$(cat 2>/dev/null || true)"
model_name="$(printf '%s' "$input" | jq -r '.model.display_name // "?"' 2>/dev/null || echo '?')"
clean_model="$(printf '%s' "$model_name" | sed 's/ [0-9].*$//')"

project_root="${CLAUDE_PROJECT_DIR:-$(pwd)}"
cd "$project_root" 2>/dev/null || { printf '%s' "$clean_model"; exit 0; }

# Git
git_info=""
if branch="$(git symbolic-ref --short HEAD 2>/dev/null)"; then
    if git diff --quiet 2>/dev/null && git diff --cached --quiet 2>/dev/null; then
        git_info=$'\033[32m⎇ '"$branch"$'\033[0m'
    else
        git_info=$'\033[33m⎇ '"$branch"$'●\033[0m'
    fi
fi

# Store sizes (lightweight stat, no DB calls)
sqlite_info=""
if [[ -f data/openai/lxd.sqlite3 ]]; then
    sqlite_size="$(du -h data/openai/lxd.sqlite3 2>/dev/null | cut -f1 | tr -d ' ')"
    sqlite_info=$'\033[36m◾sqlite '"$sqlite_size"$'\033[0m'
else
    sqlite_info=$'\033[31m◽sqlite\033[0m'
fi

lance_info=""
if [[ -d data/openai/lancedb ]]; then
    lance_size="$(du -sh data/openai/lancedb 2>/dev/null | cut -f1 | tr -d ' ')"
    lance_info=$'\033[36m◾lance '"$lance_size"$'\033[0m'
else
    lance_info=$'\033[31m◽lance\033[0m'
fi

# .env presence (never read the value)
env_info=""
if [[ -f .env ]]; then
    env_info=$'\033[32m✓.env\033[0m'
else
    env_info=$'\033[31m✗.env\033[0m'
fi

# Corpus path resolution
corpus_info=""
corpus_path="$(awk -F': *' '/^[[:space:]]*corpus_path:/{print $2; exit}' config.yaml 2>/dev/null || true)"
if [[ -n "$corpus_path" && -d "$corpus_path" ]]; then
    corpus_info=$'\033[32m◆corpus\033[0m'
elif [[ -n "$corpus_path" ]]; then
    corpus_info=$'\033[31m◇corpus\033[0m'
fi

# Compose
parts=("$clean_model")
[[ -n "$git_info" ]] && parts+=("$git_info")
[[ -n "$corpus_info" ]] && parts+=("$corpus_info")
[[ -n "$sqlite_info" ]] && parts+=("$sqlite_info")
[[ -n "$lance_info" ]] && parts+=("$lance_info")
[[ -n "$env_info" ]] && parts+=("$env_info")

(IFS=' '; printf '%s' "${parts[*]}")
exit 0
