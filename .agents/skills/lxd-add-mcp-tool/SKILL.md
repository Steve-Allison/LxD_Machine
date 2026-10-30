---
name: lxd-add-mcp-tool
description: |
  Scaffold a new MCP tool for the LxD server with all required boilerplate:
  Pydantic output model, tool function in tools.py, async wrapper + annotations
  in server.py, and golden manifest refresh. Use when the user asks to "add an
  MCP tool", "expose X as MCP", or "register a new tool". Enforces the
  read-only + async_runtime + typed-output contract from the SOTA pass.
allowed-tools:
  - Read
  - Edit
  - Write
  - Grep
  - Bash(pixi run pyright:*)
  - Bash(pixi run ruff:*)
  - Bash(pixi run pytest tests/integration/test_mcp_tool_manifest.py:*)
---

# Add an MCP Tool — Skill

Walk through the canonical pattern for adding an MCP tool to LxD without
drifting from the SOTA shape.

## When this skill is invoked

- User asks to "add an MCP tool", "expose X as an MCP tool"
- User asks to "register a new tool", "wrap Y for MCP"

## The canonical pattern (every tool)

1. **Define a Pydantic output model** in `src/lxd/mcp/models.py`.
   - Inherit from `_Frozen` (frozen, extra=forbid).
   - Field-by-field with `Field(description=...)` where the meaning isn't
     obvious from the type alone.
   - For `None`-return-on-missing, return `Model | None`, never an empty
     instance.

2. **Write the tool function** in `src/lxd/mcp/tools.py`.
   - Signature: takes `app_context: AppContext` and/or `plan: IngestPlan`
     plus its tool-specific args.
   - Returns the Pydantic model (or `Model | None`).
   - Wraps store reads in `with pooled_connection(...)`.
   - Tolerates missing stores: return `[]` / `None` if `sqlite_path` doesn't
     exist.
   - Uses `_require_non_empty(...)` for non-empty string params.

3. **Register the tool** in `src/lxd/mcp/server.py`.
   - Pick the right semantic hint:
     - `_HINT_IDEMPOTENT` — ontology-bound, deterministic
     - `_HINT_OPEN_WORLD` — store-bound, state can change
     - `_HINT_LLM` — synthesises via LLM (non-deterministic)
   - Async wrapper goes through `run_tool("tool_name", lambda: ..., timeout_secs=_tool_timeout(lxd))`.
   - Add `Annotated[T, Field(description=..., ge=..., le=...)]` for every
     parameter.
   - Long-running tools (LLM, retrieval) emit `ctx.report_progress(...)`
     before and after.
   - Docstring is the MCP client-facing description.

4. **Refresh the golden manifest** if the tool list changed:

   ```bash
   pixi run pytest tests/integration/test_mcp_tool_manifest.py --update-golden
   ```

5. **Verify**:

   ```bash
   pixi run pyright src/lxd/mcp/
   pixi run ruff check src/lxd/mcp/
   pixi run pytest tests/integration/test_mcp_tool_manifest.py
   ```

## What the skill DOES NOT do

- Does **not** add `dict[str, object]` / `list[dict[str, object]]` returns.
  The SOTA pass eliminated all of those; this skill is the contract that
  prevents regression.
- Does **not** register a tool with `readOnlyHint=False`. LxD tools are
  read-only by contract.
- Does **not** skip the async wrapper. Sync bodies run through
  `run_tool` for timeout + structured logging.
- Does **not** raise un-typed exceptions. `_require_non_empty` → `ValueError`
  is the existing pattern; mirror it for new validation.
- Does **not** silently change other tools while adding a new one.

## Suggested template

When the user says "add a tool for X", produce a draft like:

```python
# models.py
class XResponse(_Frozen):
    """One-line purpose."""
    foo: str
    bar: int


# tools.py
def x_tool(app_context: AppContext, foo: str) -> XResponse:
    """One-line description."""
    _require_non_empty(foo, "foo")
    store_paths = build_store_paths(app_context.config.paths.data_path)
    with pooled_connection(store_paths.sqlite_path) as connection:
        ...
    return XResponse(foo=..., bar=...)


# server.py
@mcp.tool(annotations=_HINT_OPEN_WORLD)
async def x(
    foo: Annotated[str, Field(description="...")],
    ctx: Context,
) -> XResponse:
    """Public docstring — what the client sees."""
    lxd = _lxd(ctx)
    return await run_tool("x", lambda: x_tool(lxd.app_context, foo), timeout_secs=_tool_timeout(lxd))
```

## Cross-reference

- `.claude/rules/mcp-tools-readonly.md` — the contract this skill enforces
- `Plans/05_MCP_SPEC.md` — canonical tool catalogue
- `src/lxd/mcp/async_runtime.py` — `run_tool` wrapper
- `src/lxd/mcp/models.py` — existing output models for reference
