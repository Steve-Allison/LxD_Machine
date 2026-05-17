#!/usr/bin/env bash
# InstructionsLoaded hook: log which path-scoped rule files exist and briefly
# what each scopes to, so Claude has explicit context about which architectural
# constraints are in play. Useful when a path-scoped rule fails to fire as
# expected — the log records what *was* loaded.
#
# Writes to .claude/.session-instructions.log (append) and prints a one-line
# summary to stdout for the session.

set -euo pipefail

project_root="${CLAUDE_PROJECT_DIR:-$(pwd)}"
rules_dir="$project_root/.claude/rules"
log_file="$project_root/.claude/.session-instructions.log"

[[ -d "$rules_dir" ]] || exit 0

timestamp="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
{
    echo "── ${timestamp} ──"
    for rule in "$rules_dir"/*.md; do
        [[ -f "$rule" ]] || continue
        rule_name="$(basename "$rule" .md)"
        # Extract the paths: list from YAML frontmatter (between first two `---`).
        paths="$(awk '/^---$/{flag++; next} flag==1 && /^paths:/{p=1; next} flag==1 && p && /^  - /{gsub(/^  - "/,""); gsub(/"$/,""); print; next} flag==1 && p && !/^  /{p=0} flag==2{exit}' "$rule" | paste -sd ',' -)"
        if [[ -n "$paths" ]]; then
            echo "  rule[${rule_name}] paths=${paths}"
        else
            echo "  rule[${rule_name}] (no path scope — global)"
        fi
    done
    echo
} >>"$log_file"

# Compact stdout summary
rule_count="$(find "$rules_dir" -maxdepth 1 -name '*.md' -type f 2>/dev/null | wc -l | tr -d ' ')"
echo "Loaded ${rule_count} path-scoped rules from .claude/rules/ (full list: .claude/.session-instructions.log)"
