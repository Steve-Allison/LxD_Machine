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

Optional machine-specific variants:

```bash
config.m1max.yaml
config.m4mini.yaml
```

Use `--profile m1max` or `--profile m4mini` to select a profile file explicitly.

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

- `query_lxd`
- `search_corpus`
- `get_entity_types`
- `get_related_concepts`
- `corpus_status`

---

## 6. Working State

Working state:

- ingest commits progress while it runs
- `status` shows committed progress
- MCP tools answer against the real store
