---
name: lxd-status
description: |
  One-shot orientation for the LxD Machine project. Reports git state, configured
  corpus path (and whether it resolves), SQLite + LanceDB store sizes, knowledge
  graph build state, config-drift warnings, and recent commits. Use when the user
  asks "status", "where are we", "what's the state", "is the corpus ingested",
  "is the graph built", or wants orientation at session start.
  Read-only. Never modifies anything.
allowed-tools:
  - Read
  - Bash(pixi run status:*)
  - Bash(pixi run graph-status:*)
  - Bash(git status:*)
  - Bash(git log:*)
  - Bash(git branch:*)
  - Bash(ls -la data/openai/:*)
  - Bash(du -sh data/openai/*:*)
  - Bash(test -f .env:*)
  - Bash(grep:*)
  - Bash(find:*)
---

# LxD Status — Skill

Reads project state and presents a structured status report. **Never writes anything.**

## When this skill is invoked

- User asks "status", "where are we", "what's the state"
- User asks "is the corpus ingested", "is the graph built"
- User asks "how big is the store", "what's in the data dir"
- Session start, after a long context switch, after another teammate's commits

## What the skill produces

A single tight block, no narration:

```
LxD Machine status
  git:     branch=main  dirty=N files  ahead=K  behind=M
           <last 3 commits one-liners>
  corpus:  <corpus_path>  (resolves / missing)  (~N text files)
  sqlite:  data/openai/lxd.sqlite3  (size)  schema_version=X
  lancedb: data/openai/lancedb  (size, M tables: chunk_vectors, ...)
  graph:   built=YYYY-MM-DD  entities=N  communities=M  claims=K
           OR  not built yet
  .env:    present / MISSING
  drift:   <any config.lock.mismatch or paths drift>
```

## Procedure

1. **`pixi run status`** — captures corpus + store + ontology + drift in one call.
2. **`pixi run graph-status`** — knowledge graph metadata.
3. **`git status` + `git log --oneline -3`** — branch and recent activity.
4. **`test -f .env`** — never read the file, just check presence.
5. Format the output into the block above.

## What this skill does NOT do

- Does not run preflight (that's a gate, separate decision — see
  `.claude/rules/ingest-discipline.md`).
- Does not run ingest, build-graph, or any other mutating command.
- Does not modify files or settings.
- Does not propose actions — orientation only.

## When to recommend `lxd-ingest` or `lxd-rebuild`

If `lxd-status` reveals one of these states, surface the next-step suggestion
in one sentence — do not auto-run:

- Store missing or empty → recommend `lxd-rebuild` (full sequence)
- Config drift present → recommend reviewing `config.yaml` + `lxd-ingest`
- Graph never built → recommend `pixi run build-graph` after ingest is sound
- `.env` missing → tell user to create it; do not scaffold a template

## Cross-reference

- `.claude/rules/ingest-discipline.md` — preflight is a gate
- `.claude/hooks/session-start.sh` — automatic short version at every session
  start; this skill is the deeper on-demand version
