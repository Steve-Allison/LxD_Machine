---
name: lxd-rebuild
description: |
  Full rebuild sequence from a fresh slate: confirm destructive intent → wipe
  data store → ingest --full → build-graph --full. Every transition is a
  human-gated stop. Use when the user asks to "start fresh", "rebuild from
  scratch", "wipe and rebuild", "full rebuild", or after a corpus swap that
  invalidates the existing store. NEVER auto-chains across phases.
allowed-tools:
  - Read
  - Bash(pixi run status:*)
  - Bash(pixi run preflight:*)
  - Bash(pixi run graph-status:*)
  - Bash(pixi run ingest:*)
  - Bash(pixi run build-graph:*)
  - Bash(ls -la data/openai/:*)
  - Bash(du -sh data/openai/*:*)
  - Bash(rm -rf data/openai/lxd.sqlite3:*)
  - Bash(rm -rf data/openai/lancedb:*)
  - Bash(rm -rf data/openai/config.lock:*)
  - Bash(rm -rf data/openai/ingest_snapshot.json:*)
---

# LxD Rebuild — Skill

Full rebuild orchestration. Four phases, each gated. **No two phases run
back-to-back without an explicit user go.**

## When this skill is invoked

- User asks to "start fresh", "rebuild from scratch", "wipe and rebuild"
- User changed `corpus_path` and the existing store is now meaningless
- Schema or pipeline changes substantial enough to warrant a clean slate

## Phases

### Phase 1 — Inventory (no cost)

Run `lxd-status` first. Report what's currently in `data/openai/`:

- Active store (lxd.sqlite3, lancedb/, config.lock, ingest_snapshot.json)
- Any `*.bak` files (pre-migration safety snapshots)
- Total disk usage
- Last ingest date (from config.lock or ingest_snapshot.json)

**STOP.** Wait for the user to confirm: *"This is what will be deleted. Proceed?"*
Quote the exact `rm` command before running it.

### Phase 2 — Wipe (destructive)

Only after explicit Phase 1 confirmation. The exact command:

```bash
rm -rf data/openai/lxd.sqlite3 data/openai/lancedb data/openai/config.lock data/openai/ingest_snapshot.json
```

**Do NOT** delete:

- `data/openai/runtime/` — purpose unclear, harmless to leave
- `data/openai/.DS_Store` — macOS metadata, irrelevant
- `*.bak` safety snapshots — separate decision (§7 preserve user state).
  Ask separately if the user wants those gone.

Confirm with `ls -la data/openai/` after the wipe. **STOP.** Report.

### Phase 3 — Ingest (costed)

Hand off to `.claude/skills/lxd-ingest/SKILL.md`. That skill's gates apply
in full: readiness → preflight → user go → ingest. Do not skip them just
because we're inside a rebuild sequence.

After ingest completes, report counts (chunks, mentions, etc.). **STOP.**

### Phase 4 — Build graph (costed)

Only after Phase 3 succeeded and the user has reviewed the ingest results:

```bash
pixi run build-graph --full
```

`--full` is destructive (re-extracts claims, costs API calls). Confirm
again per §7 destructive-op confirmation.

When complete, report graph stats via `pixi run graph-status`.

## Anti-patterns this skill prevents

- **One-command "rebuild" that runs wipe→ingest→build-graph in sequence
  without human checkpoints.** Each phase is a gate.
- **Treating the rebuild request as standing approval for all four phases.**
  The approval covers the *plan*, not each costed step.
- **Auto-deleting `.bak` files** alongside the active store. Those are
  separate user state.

## Cross-reference

- `.claude/skills/lxd-ingest` — Phase 3 detail; gates apply unchanged
- `.claude/skills/lxd-status` — Phase 1 inventory
- `.claude/rules/ingest-discipline.md` — preflight-is-a-gate underpins all
  costed phases
- `~/.claude/CLAUDE.md` §7 — destructive ops need fresh per-session approval
