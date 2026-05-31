---
name: lxd-ingest
description: |
  Run an LxD corpus ingest with mandatory gates: pre-ingest readiness checks,
  preflight cost estimate, EXPLICIT USER GO-AHEAD, then ingest. Codifies the
  preflight-is-a-gate rule (2026-05-31 incident) so the operation cannot
  silently chain from estimate to costed run. Use when the user asks to "run
  ingest", "ingest the corpus", "rebuild the index" — incremental or full.
allowed-tools:
  - Read
  - Bash(pixi run status:*)
  - Bash(pixi run preflight:*)
  - Bash(pixi run ingest:*)
  - Bash(test -f .env:*)
  - Bash(grep:*)
---

# LxD Ingest — Skill

End-to-end ingest sequence with mandatory human gates. **The skill never
auto-chains preflight into ingest.** Every costed step pauses for explicit
go-ahead.

## When this skill is invoked

- User asks to "run ingest", "ingest the corpus", "rebuild the index"
- User asks to ingest a specific corpus or after a corpus_path change
- User runs `--full` rebuild of corpus state

## The mandatory sequence

Each step ends with **STOP. Report. Wait.** No step auto-runs the next.

### Step 1 — Readiness check (no cost)

Verify all of the following. If any fails, surface and stop:

1. `.env` exists at project root. Do NOT read its contents.
2. `OPENAI_API_KEY` loads via dotenv (probe without printing value):

   ```bash
   pixi run python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('set' if os.environ.get('OPENAI_API_KEY') else 'NOT SET')"
   ```

3. `config.yaml :: paths.corpus_path` resolves to a directory that exists.
4. `data/openai/` is writable.

Report: "Readiness: <each check pass/fail>." Stop. Wait.

### Step 2 — `pixi run preflight` (no cost)

Run preflight. Capture the output. Report a structured summary:

- Corpus path
- Schema version + integrity
- Existing chunk_vectors count (will be replaced if `--full`)
- Existing embedding_cache count
- Files to ingest + token estimate
- Embedding cost upper bound
- LLM relation-extraction ceiling — if `ingest_budget.max_llm_calls_per_run`
  is unset, name a realistic order-of-magnitude estimate (gpt-4o-mini × N
  chunks × 15 max relations/chunk) and recommend setting a cap.

**STOP.** Report. Ask the user: *"Press go on ingest, set a budget cap first,
or cancel?"* Do NOT proceed without the explicit decision.

### Step 3 — Ingest (costed)

Only after the user says go in Step 2:

- For incremental: `pixi run ingest`
- For full rebuild: `pixi run ingest --full`  (per §7, this needs a fresh
  destructive-op confirmation — confirm again before running)

Run in the background if the corpus is large. Report progress and final
counts when done.

## Anti-patterns this skill prevents

- **Chaining preflight directly into ingest** — the 2026-05-31 incident. The
  user said "yes to preflight then ingest"; agent ran both back-to-back with
  no human review of preflight output. Counts as a §5 deviation.
- **Treating "yes" to a multi-step proposal as approval for every step** —
  yes-to-preflight is conditional yes-to-ingest only after preflight output
  has been reviewed.
- **Silently skipping the .env check** because "it was set last time."

## Cross-reference

- `.claude/rules/ingest-discipline.md` — the rule this skill implements
- `~/.claude/projects/-Users-steveallison-AI-Projects-Code-LxD-Machine/memory/feedback_preflight_is_a_gate.md`
  — the original incident record
- `.claude/skills/lxd-rebuild` — full sequence including wipe + build-graph
- `~/.claude/CLAUDE.md` §5 (Honor decided plan), §7 (Preserve user state)
