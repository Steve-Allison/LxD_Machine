# LxD Machine

An **Instructional Designer / Learning Experience Designer in RAG format**. The corpus encodes pedagogical and delivery best practice grounded in academic theory and industry evidence: instructional design frameworks, cognitive load theory, assessment design, modality selection, delivery formats, and performance support.

Built for Adobe field enablement. The system ingests a mixed-format corpus (Markdown, Docling JSON, PNGs), constructs an ontology-driven knowledge graph with entity recognition, relation extraction, community detection, and centrality analysis, and exposes retrieval, graph-augmented synthesis, and design-artefact agents via 23 Model Context Protocol (MCP) tools. Everything runs locally; MCP is the external interface.

---

## Key Features

- **Mixed-Format Ingestion**: Hybrid Docling chunking for Markdown and Docling JSON with recursive context refinement, parent-link inference for PNG assets, and frontmatter extraction (`sources:`, `[[slug]]` cross-references).
- **Ontology & Knowledge Graph**: LinkML and Central Configs runtime ontology integration. Extracts claims, canonical deduplicated relations, Louvain community hierarchies, 6 graph centrality metrics (PageRank, betweenness, closeness, in/out-degree, eigenvector), and deterministic/LLM-enriched entity profiles and community reports.
- **Advanced Retrieval Pipeline**:
  - LanceDB native hybrid search fusing dense vectors with Tantivy BM25 via Weighted RRF.
  - Adaptive query routing (`narrow`, `standard`, `broad`) with optional HyDE query rewriting and multi-query paraphrasing.
  - Semantic entity expansion combining Aho-Corasick automaton surface matching with vector nearest-neighbour lookups.
  - Multi-lane ranking fusing dense similarity, cross-encoder rerank scores, extracted relation links, graph claims, and centrality signals with community diversity filtering.
- **Apple Silicon Native Acceleration (MLX)**: In-process cross-encoder reranking (`mlx-community/Qwen3-Reranker-4B-mxfp8`) and resident chat/synthesis server (`mlx_lm.server` with `mlx-community/Qwen2.5-7B-Instruct-4bit`) delivering sub-second inference with zero external daemon dependencies.
- **FastMCP Server (23 Tools)**: Full MCP surface providing corpus search, knowledge graph navigation, path finding, graph health, and multi-step learning design and design critique agents.
- **SOTA Python 3.14 Engineering**: Strict Pixi-only environment, Pyright strict typechecking, Ruff linting/formatting, parameter-safe SQL queries, zero tech debt.

---

## Architecture

```text
src/lxd/
├── cli/                      # Typer CLI: ingest, status, eval, eval-gaps, build-graph, preflight, caption-assets
│   ├── preflight.py          # Schema-integrity + corpus-readiness gate
│   └── graph.py              # build-graph + graph-status state machine
├── app/                      # Bootstrap, AppContext, config.lock drift reconciliation
├── agents/                   # Multi-step design-artefact agent (design_learning) + critique agent
├── domain/                   # Pydantic models (citations, IDs, briefs, status enums)
├── eval/                     # RAGAS-style answer-quality eval (faithfulness, answer relevance, context precision)
├── net/                      # Shared httpx.Client / httpx.AsyncClient connection-pooled factories
├── ingest/                   # Corpus pipeline (sequential orchestrator + per-phase modules)
│   ├── pipeline/             # Top-level orchestrator: run_ingest, build_ingest_plan, persist loop
│   ├── scanner.py            # Filesystem scan + streaming BLAKE3 hashing of corpus files
│   ├── chunking.py           # Hybrid Docling chunker; recursive context refinement
│   ├── embedder.py           # Batched MLX/OpenAI embedding with context-aware retry
│   ├── embedding_cache.py    # Content-addressed LanceDB cache: (chunk_hash, model, dims)
│   ├── relations.py          # LLM-based relation extraction
│   └── claims.py             # LLM-based claim extraction
├── mlx/                      # Native Apple Silicon acceleration
│   ├── chat.py               # Local MLX chat/synthesis engine + streaming
│   ├── rerank.py             # Native MLX cross-encoder reranker
│   ├── embed_server.py       # In-process MLX embedding service
│   └── rerank_server.py      # In-process MLX reranking service
├── ontology/                 # YAML ontology loading, graph building, Aho-Corasick matching
│   ├── entity_graph.py       # Combined entity graph + 6 centrality metrics
│   ├── communities.py        # Louvain community detection (hierarchical)
│   ├── evidence.py           # Canonical relation deduplication + evidence provenance
│   └── profiles.py           # Entity profiles, community reports, batch-loaded enrichment
├── stores/                   # SQLite + LanceDB (vectors canonical in LanceDB)
│   ├── schema.py             # Numbered migrations (PRAGMA user_version) + integrity check
│   ├── sqlite/               # SQLite query/upsert subpackage with per-thread connection pool
│   └── lancedb.py            # Canonical vector store + chunk_vectors / embedding_cache tables
├── retrieval/                # Query pipeline: dense → rerank → expansion → graph routing → synthesis
│   └── query_pipeline.py     # End-to-end query orchestrator
├── synthesis/                # Answer generation with citations + graph context + transitive sources
├── mcp/                      # FastMCP server (23 read-only tools + prompts + resources)
└── settings/                 # Pydantic config models + YAML profile loader
```

---

## Getting Started

### Prerequisites

- macOS (Apple Silicon `osx-arm64`)
- [Pixi](https://pixi.sh/) package manager
- OpenAI API Key (configured in `.env` if using OpenAI embedding backend)

```bash
# Clone and enter the repository
git clone https://github.com/Steve-Allison/LxD_Machine.git
cd LxD_Machine

# Install all dependencies into the isolated Pixi environment
pixi install
```

### Common Commands

```bash
# Schema integrity and corpus readiness preflight gate
pixi run preflight

# Incremental corpus ingestion (skips unchanged files via BLAKE3 hash)
pixi run ingest

# Full corpus rebuild (recreates SQLite + LanceDB tables)
pixi run ingest --full

# Corpus and ontology summary status
pixi run status

# Build or resume knowledge graph (claims, communities, centrality, profiles)
pixi run build-graph

# Knowledge graph build status and statistics
pixi run graph-status

# Launch FastMCP server over stdio
pixi run mcp

# Retrieval evaluation against tests/eval/eval_set.json (Recall@10, MRR@10)
pixi run retrieval-check

# Verification suite
pixi run test           # Run pytest test suite (556 tests)
pixi run lint           # Ruff linter
pixi run fmt            # Ruff formatter
pixi run typecheck      # Pyright strict mode

# Interactive CLI launcher
./start.sh
```

---

## MCP Tools (23 Tools)

FastMCP (>=3.0) stdio interface exposing:

### Corpus Tools
- `corpus_status`: Corpus file counts, ingestion progress, and index summary.
- `get_entity_types`: List registered ontology entity types and their frequencies.
- `get_related_concepts`: Find concept relationships and linked terms.
- `search_corpus`: Hybrid vector + keyword search over ingested chunks.
- `find_documents_for_concept`: Locate documents defining or mentioning an ontology concept.
- `get_corpus_relations`: Extract corpus-grounded relations for an entity.

### Knowledge Graph Tools
- `get_entity_summary`: Deterministic profile, centrality scores, and top claims for an entity.
- `get_community_context`: Community report, member concepts, and thematic summaries.
- `get_similar_entities`: Cosine-similar entities based on knowledge graph embeddings.
- `search_entities`: Substring and alias search over indexed entities.
- `inspect_evidence`: Inspect verbatim provenance chunks supporting an extracted relation.
- `find_path_between_entities`: Shortest path between two entities in the knowledge graph.
- `find_weighted_path`: Relation-confidence-weighted path between entities.
- `get_hub_entities`: Highest PageRank and in-degree hub concepts.
- `find_bridge_entities`: Highest betweenness centrality bottleneck/bridge concepts.
- `find_foundational_entities`: Closeness-central concepts foundational to the domain.
- `get_entity_graph_stats`: Graph order, size, density, and component statistics.

### Answer & Synthesis Pipeline
- `search_knowledge`: Full graph-augmented synthesis pipeline with inline citations and learner brief conditioning.
- `search_knowledge_deep`: Extended synthesis returning structured graph framing context alongside the cited answer.
- `get_graph_overview`: High-level graph topology and health metrics.

### Design Agents
- `design_learning`: Multi-step instructional design agent generating structured learning blueprints, activities, and assessments.
- `critique_design`: Evaluates an existing learning design against pedagogical evidence in the corpus.

### Eval
- `list_eval_gaps`: Identify retrieval test cases falling below Recall/MRR thresholds.

---

## Configuration

Configuration is managed via `config.yaml` and validated at bootstrap with Pydantic v2:

- `paths`: Paths to curated wiki corpus, staged ontology, and SQLite/LanceDB data directory.
- `models`: Primary embedding model (`text-embedding-3-small` or `Octen-Embedding-8B-mlx`), reranker (`mlx-community/Qwen3-Reranker-4B-mxfp8`), and LLM (`mlx-community/Qwen2.5-7B-Instruct-4bit`).
- `mlx`: Local MLX chat/embed endpoints, autostart flags, and inference parameters.
- `retrieval`: k-NN limits, reranking depth, RRF weights, and community diversity settings.
- `knowledge_graph`: Extraction models, community detection resolution, and centrality caps.
