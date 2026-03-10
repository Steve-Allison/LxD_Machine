# LxD Machine — Build Plan

**Document type:** Build plan  
**Status:** Active  
**Version:** 1.1  
**Created:** 2026-03-09  

---

## 1. Rule Zero

Do not plan from guesses.
Every performance claim or workload claim must be benchmarked first.
The technology stack must be explicitly locked down to prevent AI hallucinations.

---

## 2. Locked Tech Stack

The following libraries and technologies are mandatory and strictly scoped:

- **Environment & Execution:** `pixi` (Dependency management), `rich` (CLI progress bars and tables).
- **Document Conversion & Chunking:** `docling` for source conversion to `DoclingDocument`, plus Docling native chunking with tokenizer-aware refinement via `HybridChunker`, aligned to the configured tokenizer.
- **Local Models:** `ollama` for embeddings and synthesis, plus `llama.cpp` server for reranking.
- **Vector Database:** `LanceDB` configured with **Flat vector search**, **Cosine Similarity**, and store-level metadata filtering. Native Full-Text Search is deferred until hybrid retrieval is explicitly started.
- **Relational Metadata:** `SQLite` initialized with **WAL mode** (`PRAGMA journal_mode=WAL;`) for concurrent MCP access.
- **Ontology Graph:** `networkx.MultiDiGraph` built in-memory at startup.
- **Entity Matching:** `pyahocorasick` (Exact-string, zero ML overhead).
- **External Interface:** `fastmcp` (Model Context Protocol SDK) running over `stdio` transport exclusively.

Later optional additions are not part of the mandatory baseline stack and must not appear in the base runtime environment until the relevant phase begins.

---

## 3. Phase Order

### Phase 0 — Reality Baseline & Evaluation Set
Deliver:
- Real corpus counts and chunk estimates.
- `tests/eval/eval_set.json` created, containing at least 20 real user questions mapped to expected markdown or Docling JSON source files.
- A baseline Docling conversion and chunking pass on a subset of files to verify markdown structural parsing (including image alt-text preservation) and chunk stability.
- A proof that ontology loading resolves `!include`, hashes the resolved ontology snapshot, and hashes the canonical normalized matcher term set rather than only top-level files.

Acceptance:
- `eval_set.json` exists and is formatted correctly.
- ontology loading over the real `Yamls/` tree succeeds without unresolved include handling
- the persisted `matcher_termset_hash` reproduces exactly from the same resolved ontology inputs

### Phase 0.5 — The Embedding Model Benchmark
Deliver:
- Benchmark English-first embedding models with practical local trade-offs:
  - `nomic-embed-text` as the default live Ollama candidate on this machine
  - `mixedbread-ai/mxbai-embed-large-v1` as the compact English retrieval baseline
  - `BAAI/bge-m3` only if dense-only results suggest the heavier model is justified
- Run the fixed `eval_set.json` through LanceDB Flat search for each model to ensure recall quality does not drop while speed increases.
- Verify the actual Ollama runtime/backend path on this machine rather than assuming CPU or Metal/MPS behavior.
- Verify the actual `llama.cpp` reranker server path on this machine rather than assuming endpoint or transport behavior.
- Verify the configured embedder, reranker, and synthesis model IDs pass the defined readiness probes.
- Verify tokenizer/backend correctness for the chosen embedding model.
- Run a corpus-subset oversize audit against the installed embedder using `truncate=false`; no chunk-size claim is accepted until the live embedder either accepts the chunk or the emergency split path proves it can refine it automatically.
- Benchmark rerank quality and latency for:
  - `dengcao/Qwen3-Reranker-0.6B:F16` as the current local efficiency default
  - `dengcao/Qwen3-Reranker-4B:Q4_K_M` as the current local higher-quality option
- Benchmark answer synthesis for grounded citation-heavy responses using:
  - `mistral-small3.1` as the current local best-balance option
  - `qwen3:30b-a3b` as the current local best-quality option
- Lock the chosen embedder, reranker, and synthesis model into `config.{profile}.yaml`.

Model readiness probe definition:

- embedder probe: one embedding request on a fixed probe string with `truncate=false`
- reranker probe: one `llama.cpp` rerank request on a fixed probe query with two fixed probe candidates
- synthesis probe: one short generation request on a fixed probe prompt

Acceptance:
- Ingest throughput improves by at least 25% over the measured baseline on the same machine profile.
- The benchmark report records whether tokenizer counts are exact, approximate, or unavailable for the chosen embedder runtime. Approximate counts may be used only for initial chunk construction, never as proof that a chunk is safe to embed.
- Corpus-subset ingest completes without any final unhandled embed-context failures under `truncate=false`.
- The chosen reranker improves `MRR@10` by at least 10% over dense-only retrieval while keeping p95 rerank latency <= 400 ms for a fixed dense candidate set of 20.
- The chosen synthesis model keeps `query_lxd` p95 <= 12.0 seconds on the target local profile.

### Phase 1 — Durable Core Ingest (The V1 Baseline)
Deliver:
- `pixi run ingest` with incremental, SQLite-tracked progress commits.
- `rich` CLI progress bars.
- Move/Rename detection using Blake3 hashing.
- LanceDB Flat vector storage with JSON metadata for Docling structure.
- Pre-flight check verifying the Ollama embed runtime is responding and that the configured embed model is installed before any searchable-store mutation.
- Aho-Corasick exact-string matcher for ontology entity detection.
- A defined compaction/cleanup path for delete-heavy LanceDB updates.

Acceptance:
- Interrupting the ingest leaves the `corpus_manifest` stable and resumable.
- Moving files on disk does not trigger 30 minutes of re-embedding.
- Interrupted updates do not leave silent missing-chunk gaps unreconciled.

### Phase 2 — The Query Pipeline & MCP Server
Deliver:
- Search and cited answer synthesis tools.
- Source-aware retrieval ranking:
  - adaptive dense candidate fetch up to the query cap when the first dense window does not contain enough unique sources
  - one representative chunk per `source_rel_path` before reranking
  - final fusion of dense rank, lexical rank from committed source metadata, and rerank rank
- FastMCP server running over `stdio`.
- Concurrent SQLite read access (WAL mode) allowing queries while ingest runs.

Acceptance:
- AI assistants (like Claude Desktop) can connect via MCP and query the built store.
- Query evaluation uses explicit `source_rel_path` expectations and rejects ambiguous basename guesses.

### Phase 3 — V1 Live Verification
Deliver:
- Full real ingest on the entire corpus using the optimized model.
- MCP smoke test against the full built store.
- System is declared "Operational Baseline".

### Phase 4 — Optional Retrieval Enrichment
Deliver:
- Optional reranker acceleration such as `FlashRank` if benchmarks show no statistically meaningful loss on `MRR@10` while reducing p95 latency.
- Optional native LanceDB FTS if benchmarks show hybrid retrieval improves `Recall@10` enough to justify its added complexity.

Phase-entry rule:

- optional dependencies such as `FlashRank` are added only when this phase is explicitly started
- the baseline `pixi.toml` must not claim optional later dependencies as if they are already part of the operational core

Acceptance:
- Optional retrieval features improve the chosen eval metrics without violating the baseline latency and durability targets.
