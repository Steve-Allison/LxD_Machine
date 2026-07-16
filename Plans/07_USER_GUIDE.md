# LxD Machine — User Guide

---

## 1. What To Expect

- first full ingest may be long-running
- progress must be visible
- committed progress must survive interruption

---

## 2. Setup

### 2.1 Choose Runtime Config

Default portable runtime:

```bash
config.yaml
```

The `--profile <name>` CLI option is supported by `pixi run ingest`, `pixi run status`, and the MCP server — when set it resolves `config.<name>.yaml` in the project root. No profile files are shipped by default; the single `config.yaml` is authoritative. Add a profile file if you want to keep per-machine overrides (e.g. `config.m1max.yaml`) and select it via `--profile m1max`.

### 2.2 Install Environment

```bash
pixi install
```

### 2.3 Verify Local Checks

```bash
pixi run lint
pixi run typecheck
pixi run pytest -q
```

---

## 3. First Ingest

Run:

```bash
pixi run ingest --full
```

Expected behavior:

- long-running build
- progress should be visible
- committed progress should survive interruption

---

## 4. Check Status

Run:

```bash
pixi run status
```

This should show:

- tracked corpus counts by file type
- retrieval counts by role (`searchable`, `asset_only`, `not_searchable`)
- chunk count
- mention count
- ontology snapshot and matcher hashes
- ontology coverage-path and graph-relation counts
- ontology validation issues when present
- config drift warnings if relevant

---

## 5. Query Through MCP

Start the server:

```bash
pixi run mcp
```

Then connect from your MCP client.

For `stdio` clients, the launch contract is:

```json
{
  "command": "pixi",
  "args": ["run", "mcp"],
  "cwd": "/Users/steveallison/AI_Projects+Code/LxD_Machine"
}
```

If you need a non-default config file, pass it explicitly at launch time.

Minimal useful tools:

- `search_knowledge` (graph-augmented answer synthesis; was `query_lxd`)
- `search_knowledge_deep` (same plus structured graph context)
- `search_corpus`
- `get_entity_types`
- `get_related_concepts`
- `corpus_status`

See `05_MCP_SPEC.md` for the full list of 20 read-only tools.

---

## 6. Working State

Working state:

- ingest commits progress while it runs
- `status` shows committed progress
- MCP tools answer against the real store
- MCP tools are asynchronous; each call is bounded by
  `mcp.tool_timeout_secs` (default `60s`), so a stuck backend can never
  stall the server
- on first launch, `data/config.lock` is seeded with the current config
  digest; subsequent mismatches log a `config.lock.mismatch` warning. To
  accept a new configuration, delete `data/config.lock`

---

## 7. Build The Knowledge Graph (Optional)

Once ingest is complete you can build the knowledge graph:

```bash
pixi run build-graph         # resumable, incremental
pixi run graph-status        # phase state + counts
pixi run build-graph --full  # requires interactive confirmation (re-runs LLM extraction)
```

Graph build phases (claim extraction, entity graph, centrality, community
detection, entity profiles, community reports) are tracked in
`graph_build_state` and resume from the last incomplete phase. If the
graph is absent, all MCP tools still work — graph context is additive.
