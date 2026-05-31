---
name: feedback-preflight-is-a-gate
description: Preflight / dry-run / estimate commands are deliberation gates — never auto-chain them with the operation they precede.
metadata:
  type: feedback
---

Preflight, dry-run, cost-estimate, and any "review before pressing go" command is a **gate for the human**, not a step in a pipeline I get to chain.

**Rule:** Run preflight (or equivalent). Report numbers. **STOP.** Wait for the user to decide whether to proceed. The actual operation (`ingest --full`, `build-graph --full`, any deploy, any migration, any costed/destructive run) is a separate command initiated by the user — never an automatic next step from my side.

**Why:** Preflight exists *because* the operation is costly, destructive, or irreversible. Chaining it with the real run negates the purpose. 2026-05-31: I ran `pixi run preflight` against the new wiki corpus, got a clean result with the literal text *"review before pressing go"*, and then immediately launched `pixi run ingest --full` in the background. The user was rightly furious: *"Preflight is literally BEFORE FLIGHT - NEVER THE FLIGHT"*. Only saved by the OPENAI_API_KEY probe failing before any API spend.

**How to apply:**

- Commands named `preflight`, `plan`, `dry-run`, `--check`, `--estimate`, `--noop`, `status`, `diff` — all gates. Run, report, stop.
- Even when the user has said "yes" to a paragraph that includes both the gate and the operation, treat that as **conditional consent**: yes to *running preflight*, and then conditional yes to the operation *if* preflight passes — but the user still owns the press-go decision after seeing the actual numbers.
- The right shape: *"Preflight: <numbers>. Ready for `pixi run ingest --full` when you say go."* Then wait.
- If preflight surfaces something the user wouldn't have known up front (cost ceiling, schema-version bump, integrity warning, larger-than-expected corpus), that's exactly the moment the gate is most useful — never skip it.
- This rule generalises to: any `--full`, `--force`, `--purge`, `--reset`, anything with an explicit confirm prompt, anything that touches paid APIs or persistent state.

Companion rules: `~/.claude/rules/decision-fatigue.md` (don't *over*-ask), `~/.claude/CLAUDE.md` §7 (preserve user state), §6 (report honestly), `.claude/rules/ingest-discipline.md` (the in-repo expression of this rule).
