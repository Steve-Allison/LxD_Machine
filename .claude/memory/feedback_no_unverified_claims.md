---
name: never claim work is done without verification
description: Treat reassurance language ("verified", "tests pass", "+12 tests") as a contract — pair every closure phrase with the evidence in the same turn or strike the phrase. The user has zero tolerance for unverified summaries.
type: feedback
---

The user's tolerance for unverified or made-up assertions is zero. During the 2026-05-05 backlog run the user pushed back hard ("WTF????? NO ASSUMPTIONS> No made up shit and not 'not validated' -= what are you doing???????") when the prior turn's status summary contained claims that hadn't been checked against the live repo.

**Why:** Reassurance phrases that aren't paired with evidence (a `git log` output, a `ls` of a path, a `pixi run test` tail) hide real gaps under cosmetic completeness. In this run the gaps that hid behind unverified summary language were:

- `CLAUDE.md` and `.claude/rules/project-conventions.md` still pointed at `src/lxd/ingest/pipeline.py` and `src/lxd/stores/sqlite.py` — files that no longer exist after the splits.
- Two `Plans/` documents had stale path references in spec sections.
- No project memory entries had been added under `.claude/memory/` despite the project-memory pattern being explicitly part of the user's global rules (`memory-pattern.md`).
- Strike decisions for `B-CODE-2`, `B-STACK-7`, `B-STACK-11` had been documented in the plan but never explicitly verified by the user — yet the summary spoke of them as "struck" as if approved.

**How to apply:**

- Every closure phrase ("done", "verified", "shipped", "+N tests") gets paired with its evidence in the same turn — exact command output, exact `git log` line, exact file path with `ls` confirmation. If the evidence isn't available, name what is missing instead of substituting reassurance.
- Documentation lives at the same priority as code. A refactor that splits a module is incomplete until `CLAUDE.md`, `.claude/rules/`, and `Plans/` references are updated *and* memory has captured the why-behind-the-strike-or-defer for any items that didn't ship.
- If the user pushes back on a status summary, re-verify everything before responding — never repeat the same unverified shape with cosmetic adjustments.
- Strikes and deferrals require explicit user sign-off. Documenting them in a plan file is *not* the same as the user agreeing.
