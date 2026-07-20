---
paths:
  - "src/lxd/ingest/**/*.py"
  - "src/lxd/cli/**/*.py"
  - "config.yaml"
  - "Plans/03_INGEST_SPEC.md"
---

# Ingest Discipline

Operational rules for the ingest pipeline. This rule auto-loads when editing any
ingest module, the CLI, the corpus config, or the ingest spec.

## Preflight is a gate, never the flight

`pixi run preflight`, `pixi run status`, `pixi run graph-status` — and any
analogous dry-run / cost-estimate command — exist so the human can review
**before** pressing go. They are decision points, not pipeline steps.

- After running preflight or any equivalent gate, **STOP**. Report the numbers.
  The next action belongs to the user.
- Do **not** chain `pixi run preflight && pixi run ingest …`. Treat the two as
  separate sessions.
- Same rule applies to `pixi run build-graph` after `pixi run graph-status`.
- When the user says "preflight then ingest" in a single sentence, treat that as
  conditional consent: run preflight, surface the result, **wait** for explicit
  go-ahead before launching the costed run.

Detail: `~/.claude/projects/-Users-steveallison-AI-Projects-Code-LxD-Machine/memory/feedback_preflight_is_a_gate.md`.

## Required state before any ingest

Before invoking `pixi run ingest` (full or incremental):

1. `.env` exists at the project root with `OPENAI_API_KEY` set.
   Confirm via `python -c "from dotenv import load_dotenv; load_dotenv(); ..."`
   — never read or print the key value.
2. `config.yaml :: paths.corpus_path` resolves to an existing directory.
3. The data path (`data/openai/` by default) is writable.
4. Preflight passes — its check is the schema-integrity gate.

If any of the four fails, surface it and stop. Don't try to "be helpful" by
creating files the user owns (`.env`, corpus paths).

## Destructive operations require fresh confirmation

`pixi run ingest --full` and `pixi run build-graph --full` are destructive: they
supersede or rebuild large amounts of state. Every invocation needs explicit
in-session approval per `~/.claude/CLAUDE.md` §7.

- Approval from a previous session does not carry over.
- Approval given conditionally ("if preflight is clean, do X") still needs
  the gate-stop in between.
- Wiping `data/openai/` (or any subset) by hand requires the same approval, and
  the exact command should be quoted before running it.

## The two stores are coupled — never edit one without the other

Chunk persistence is **LanceDB-first, then SQLite**. Before the LanceDB write
the orchestrator snapshots existing vectors for that source; if the SQLite
half fails it restores the snapshot (empty for first ingest, prior vectors for
re-ingest) so the two stores never diverge. Move detection writes the new path
first, then deletes the old path — never delete-before-write. When touching
ingest code:

- Never write to `chunk_vectors` without a paired SQLite upsert in the same
  transaction-shaped block.
- Never write to SQLite chunk rows without ensuring the LanceDB vector exists.
- If you find yourself reasoning about "fixing inconsistent state", you've
  already introduced a bug — the invariant should be unbreakable by
  construction, not patched up after the fact.

## Cost ceilings are explicit, not "trust the model"

`config.yaml :: ingest_budget.max_llm_calls_per_run` is unset by default.
Preflight surfaces this as a warning: *"no cap configured — cost ceiling
unknown"*.

When proposing an ingest run, name the realistic cost ceiling out loud
(`gpt-4o-mini` × chunk count × max relations per chunk), and suggest setting
`max_llm_calls_per_run` for any new corpus the operator hasn't run before.

## Cross-reference

- `~/.claude/CLAUDE.md` §5–§7 — plan, report honestly, preserve state
- `~/.claude/rules/agent-prompt-discipline.md` — same discipline at the agent
  layer when ingest work is delegated
- `Plans/03_INGEST_SPEC.md` — the canonical pipeline spec
- `.claude/rules/stores-and-paths.md` — store-layer invariants
