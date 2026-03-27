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
- MCP server is in `src/lxd/mcp/` using FastMCP (>=3.0).

## Data Stores
- LanceDB for vector storage, SQLite for metadata. Both are rebuildable via `pixi run ingest`.
- Store data lives under `data/` (gitignored). Never commit database files.

## Configuration
- YAML configs use block style, not flow style.
- All config is loaded via `src/lxd/settings/loader.py`.
