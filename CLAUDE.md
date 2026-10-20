@AGENTS.md

## Claude Code conventions

This project's `.claude/` directory carries scoped guidance that auto-loads when relevant files are touched. Reach for these before redefining anything.

| Layer | Where | Triggers / use |
| --- | --- | --- |
| Rules | `.claude/rules/` | `ingest-discipline`, `stores-and-paths`, `mandatory-features`, `citations-and-evidence`, `mcp-tools-readonly`, `python-style`, `testing` (each path-scoped by its `paths:` frontmatter; none loads unconditionally) |
| Skills | `.claude/skills/` | `/lxd-status`, `/lxd-ingest`, `/lxd-rebuild`, `/lxd-add-mcp-tool` |
| Agents | `.claude/agents/` | `ingest-pipeline-auditor`, `mcp-tool-reviewer`, `schema-migration-reviewer` (read-only audits) |
| Hooks | `.claude/hooks/` | `session-start` (orient), `protect-critical` (blocks edits to `.env` / lockfiles / DBs / golden tests), `pre-bash-destructive-ingest` (warns on `ingest --full` / `build-graph --full` / `rm -rf data`), `instructions-loaded` (logs path-scoped rules per session); a PostToolUse hook runs `ruff check --fix` and `ruff format` on every edited Python file. The global hooks apply too, and the global Stop gate refuses to end a turn while `pixi run lint`, `typecheck` or `test` is red for files changed this session. |
| Memory | `.claude/memory/` | Project context, indexed by `MEMORY.md`; read on demand, not auto-loaded |

Two non-negotiables baked into these:

- **Preflight is a gate, never the flight.** `pixi run preflight` / `pixi run status` / `pixi run graph-status` always pause for explicit user go-ahead before the costed step runs.
- **No PRs in this workflow.** Stage → commit → push directly.
