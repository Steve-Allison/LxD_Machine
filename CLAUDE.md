# LxD Machine

A local-only, single-user knowledge system for instructional design content. Ingests a mixed-format corpus (Markdown, Docling JSON, PNGs), builds a searchable index with ontology-driven entity recognition, constructs a knowledge graph with community detection and centrality analysis, and exposes retrieval via MCP tools.

## Architecture

```
src/lxd/
├── cli/          # Typer CLI: ingest, status, eval, build-graph, graph-status
├── app/          # Bootstrap and status reporting
├── domain/       # Pydantic models (citations, IDs, status enums)
├── ingest/       # Corpus pipeline: scan → chunk → embed → mention → relation → persist
│   └── claims.py # Claim extraction from chunks (Phase 5)
├── ontology/     # YAML ontology loading, graph building, Aho-Corasick matching
│   ├── entity_graph.py  # Combined entity graph + 5 centrality metrics
│   ├── communities.py   # Louvain community detection (Leiden optional)
│   ├── evidence.py      # Canonical relation deduplication + evidence provenance
│   └── profiles.py      # Entity profiles, community reports, LLM enrichment
├── stores/       # SQLite (metadata/manifest) + LanceDB (vectors + entity embeddings)
├── retrieval/    # Query pipeline: dense search → rerank → expansion → graph routing → synthesis
│   └── graph_routing.py # Graph context augmentation for synthesis
├── synthesis/    # Answer generation with citations and graph context
├── mcp/          # FastMCP server (20 read-only tools)
├── observability/# structlog configuration
└── settings/     # Pydantic config models + YAML loader
```

Key directories outside `src/`:
- `Knowledge_Base/` — corpus root (gitignored)
- `Yamls/` — ontology definitions
- `Plans/` — architecture and design specs
- `tests/` — pytest suite
- `data/` — SQLite + LanceDB stores (gitignored, rebuildable)
- `.env` — API keys (OPENAI_API_KEY, etc.). Loaded by `app/bootstrap.py` via `python-dotenv` at startup. Never commit.

## Common Commands

```bash
pixi run ingest          # Incremental corpus ingestion
pixi run ingest --full   # Full rebuild
pixi run status          # Corpus and ontology status
pixi run eval            # Retrieval evaluation against tests/eval/eval_set.json
pixi run mcp             # Launch MCP server
pixi run build-graph     # Build knowledge graph (incremental, resumable)
pixi run graph-status    # Knowledge graph build state and statistics
pixi run test            # pytest -q
pixi run lint            # ruff check src tests
pixi run fmt             # ruff format src tests
pixi run typecheck       # pyright src
```

## MCP Tools

20 read-only tools exposed via FastMCP (>=3.0) over stdio transport:

**Corpus tools (Phase 4):** `corpus_status`, `get_entity_types`, `get_related_concepts`, `search_corpus`, `find_documents_for_concept`, `get_corpus_relations`

**Knowledge graph tools (Phase 5):** `get_entity_summary`, `get_community_context`, `get_similar_entities`, `search_entities`, `inspect_evidence`, `find_path_between_entities`, `find_weighted_path`, `get_hub_entities`, `find_bridge_entities`, `find_foundational_entities`, `get_entity_graph_stats`

**Full answer pipeline:** `search_knowledge` (graph-augmented synthesis), `search_knowledge_deep` (same + structured graph context), `get_graph_overview` (KG health check)

## Knowledge Graph (Phase 5)

The knowledge graph pipeline builds on top of ingested corpus data:

1. **Claim extraction** — LLM-based extraction of assertions, definitions, comparisons, causal, and procedural claims from chunks
2. **Entity graph** — combined graph from ontology edges + corpus-extracted relations
3. **Centrality** — PageRank, betweenness (unweighted), closeness, in/out degree (raw), eigenvector (numpy)
4. **Community detection** — Louvain via NetworkX (Leiden available via optional graspologic)
5. **Entity profiles** — deterministic summaries with centrality scores, optional LLM enrichment
6. **Community reports** — deterministic summaries per community, optional LLM enrichment
7. **Graph-augmented synthesis** — entity profiles, community reports, and claims prepended to synthesis prompt when entities match the query

The graph build is a resumable state machine (`pixi run build-graph`). Graph context is additive — when the graph is not yet built or no entities match a query, the pipeline degrades gracefully to the pre-Phase-5 baseline.

## Design Principles

- **Incremental by default**: ingest skips unchanged files (BLAKE3 content hash), detects moves. Graph build resumes from last incomplete phase.
- **Rebuildable**: all stores can be rebuilt from source via `pixi run ingest --full` and `pixi run build-graph --full`.
- **Explicit provenance**: every chunk traces back to source document, page, and extraction method. Every claim and relation traces to source chunk.
- **Ontology-first**: entity recognition uses Aho-Corasick automaton built from YAML definitions; relations extracted via LLM.
- **Graceful degradation**: the system remains usable when the knowledge graph is not yet built or the reranker service is unavailable.

## Key Patterns

- IDs are deterministic (BLAKE3 hash of content + context), not random UUIDs.
- Configuration is fully Pydantic v2 validated, loaded from YAML profiles via `settings/loader.py`.
- MCP tools are read-only with FastMCP (>=3.0) hints for client compatibility.
- Chunking uses recursive context refinement when embedding dimensions exceed limits.
- Graph context is additive — prepended to synthesis prompts as structured framing, not fused via RRF with chunks.

## Design Specs

Detailed specifications live in `Plans/`:
- `00_PURPOSE_AND_BACKGROUND.md` — scope, outcomes, constraints
- `01_ARCHITECTURE.md` — system architecture and store design
- `01b_CODEBASE_STRUCTURE.md` — module boundaries
- `03_INGEST_SPEC.md` — ingestion pipeline detail
- `05_MCP_SPEC.md` — MCP interface specification (Phase 0–4 baseline; see `08_KNOWLEDGE_GRAPH_SPEC.md` for Phase 5+)
- `08_KNOWLEDGE_GRAPH_SPEC.md` — knowledge graph pipeline specification (Phase 5)
