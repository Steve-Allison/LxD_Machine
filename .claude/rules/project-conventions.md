---
description: Project architecture and tooling conventions
globs: "**/*.py,**/pixi.toml,**/pyproject.toml"
---

# Project Conventions

## Package Management
- Pixi is the package manager. Use `pixi run <task>` and `pixi add <dep>`. Never use `pip install` or `conda install` directly.
- Dependencies are declared in `pyproject.toml`. Pixi tasks are in `pixi.toml`.

## Architecture
- Domain models live in `src/lxd/domain/` as Pydantic models.
- Settings/config use Pydantic models in `src/lxd/settings/`.
- Logging uses `structlog` throughout. Never use `print()` or stdlib `logging` directly.
- The ingest pipeline uses Hamilton DAG patterns. See the `hamilton-dag` skill for guidance.
- MCP server is in `src/lxd/mcp/` using FastMCP (>=3.0). 20 read-only tools registered.
- Knowledge graph pipeline lives in `src/lxd/ontology/` (entity_graph, communities, profiles, evidence) and `src/lxd/ingest/claims.py`. CLI in `src/lxd/cli/graph.py`.

## Data Stores
- LanceDB for vector storage (chunk vectors + entity embeddings), SQLite for metadata (including 8 knowledge graph tables). Both are rebuildable via `pixi run ingest` and `pixi run build-graph`.
- Store data lives under `data/` (gitignored). Never commit database files.

## Knowledge Graph
- The graph build is a resumable state machine with phases: evidence → claims → entity_graph → centrality → communities → entity_profiles → community_reports → complete.
- Graph context augments synthesis additively — prepended to prompts as structured framing, not fused via RRF with chunks.
- All graph features degrade gracefully. When the KG is not built or disabled, the system behaves identically to the pre-Phase-5 baseline.
- graspologic is optional (not in pixi.toml). Louvain via NetworkX is the default community detection algorithm. Leiden is available if graspologic is manually installed.

## Configuration
- YAML configs use block style, not flow style.
- All config is loaded via `src/lxd/settings/loader.py`.
