---
paths:
  - "src/lxd/mcp/**/*.py"
  - "tests/golden/mcp_tool_manifest.json"
  - "Plans/05_MCP_SPEC.md"
---

# MCP Tools Are Read-Only — Contract

Every LxD MCP tool, resource, and prompt is **read-only**. This is the SOTA
contract established in commit `3381325`. Auto-loads on edits to the MCP
surface.

## What "read-only" means here

A read-only tool:

- Does not modify SQLite rows
- Does not modify LanceDB rows
- Does not write to disk (except structured logs via `structlog`)
- Does not mutate the lifespan `_LxDLifespan` state
- Does not call out to external systems that have side effects
  (LLM read calls are fine; LLM fine-tuning is not)

If you find yourself needing a writeable MCP tool, that's a design decision
upstream of this rule — discuss with the user; do not unilaterally extend
the contract.

## The five-part contract

Every tool registration in `src/lxd/mcp/server.py` must satisfy ALL of:

### 1. Typed Pydantic output

Returns a model from `src/lxd/mcp/models.py` (or `Model | None`). Never
`dict[str, object]` / `list[dict[str, object]]`.

### 2. `ToolAnnotations` with `readOnlyHint=True`

Use one of the three module-level constants:

- `_HINT_IDEMPOTENT` — ontology-bound, deterministic per lifespan
- `_HINT_OPEN_WORLD` — reads the store; results can change with state
- `_HINT_LLM` — open-world AND non-deterministic (LLM synthesis)

Never set `destructiveHint=True`. Never set `readOnlyHint=False`.

### 3. Async wrapper via `run_tool`

```python
return await run_tool(
    "tool_name",
    lambda: tool_function(...),
    timeout_secs=_tool_timeout(lxd),
)
```

The `run_tool` wrapper provides per-tool timeout (via `MCPConfig.tool_timeout_secs`)
and contextvar logging. Direct sync work in the tool body bypasses both.

### 4. Annotated parameters with `Field`

Every parameter (except `ctx`) carries
`Annotated[T, Field(description=..., ge=..., le=...)]`. Numeric ranges are
explicit. Optional params have explicit `= None` defaults.

For `entity_id` and similar discoverable params: the description mentions
`get_entity_types` as the discovery path.

### 5. Progress reporting on long-running tools

Tools that synthesise via LLM or do large retrieval emit:

```python
await ctx.report_progress(progress=0, total=N, message="phase 1")
# ... work ...
await ctx.report_progress(progress=N, total=N, message="complete")
```

LxD currently applies this to `search_corpus`, `search_knowledge`, and
`search_knowledge_deep`.

## Resources and prompts

### Resources

- `@mcp.resource(uri, mime_type=...)` — no `annotations=` arg.
- Resources that return structured data use `mime_type="application/json"`
  and call `model.model_dump_json()`.
- Resources that return text use `mime_type="text/markdown"` or `"text/plain"`.
- Path-shaped params get path-traversal protection (see `_read_corpus_file`).

### Prompts

- `@mcp.prompt(name=...)` — return type is `str`.
- Prompts are documentation-as-code; they expose what the system tells the
  LLM so clients can audit the behaviour.

## Golden manifest contract

`tests/golden/mcp_tool_manifest.json` is regenerated via:

```bash
pixi run pytest tests/integration/test_mcp_tool_manifest.py --update-golden
```

If the diff adds, removes, or renames a tool, refresh the golden in the same
commit. CI verifies the golden matches the live manifest — drift fails the
build.

## Anti-pattern signals to recognise

Stop and re-think if you write any of:

- `dict[str, object]` or `list[dict[str, object]]` as a tool return type
- `@mcp.tool` without `annotations=`
- A new tool that lacks `readOnlyHint=True`
- A tool body that calls SQLite/LanceDB outside `run_tool`
- A tool param without `Annotated[T, Field(...)]`
- LLM/synthesis tool without `ctx.report_progress(...)` calls
- Adding/renaming a tool without refreshing the golden

## Cross-reference

- `Plans/05_MCP_SPEC.md` — canonical tool catalogue (20 tools, 3 resources,
  2 prompts as of commit `3381325`)
- `src/lxd/mcp/async_runtime.py` — `run_tool` wrapper detail
- `src/lxd/mcp/models.py` — typed output models
- `.claude/skills/lxd-add-mcp-tool/SKILL.md` — scaffolding helper
- `.claude/agents/mcp-tool-reviewer.md` — independent audit before merging
- `~/.claude/CLAUDE.md` §3 (MVC) — don't expand the contract speculatively
