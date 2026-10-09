---
name: Knowledge Graph Phase 5 implementation
description: Phase 5 KG pipeline implemented 2026-03-28 — Louvain communities, centrality, claims, entity profiles, graph-augmented synthesis
type: project
---

Phase 5 Knowledge Graph pipeline fully implemented and wired into retrieval/synthesis on 2026-03-28.

**Why:** Adds structured entity/community/claim context to synthesis prompts, improving answer quality for entity-rich queries.

**How to apply:**
- Spec: Plans/08_KNOWLEDGE_GRAPH_SPEC.md (v4.1, status: Implemented)
- Build via: `pixi run build-graph` (resumable state machine)
- Community detection: Louvain via NetworkX (default). graspologic removed from deps due to beartype conflict with fastmcp. Leiden codepath preserved with ImportError fallback.
- 6 centrality metrics: PageRank, betweenness (unweighted), closeness, in-degree, out-degree (raw), eigenvector (numpy)
- Graph context is additive — prepended to synthesis prompts, not fused via RRF
- Mandatory: there is no `knowledge_graph.enabled` toggle (removed; see `.claude/rules/mandatory-features.md`). Synthesis falls back to a graph-free baseline only when the graph is not yet built
- 8 new SQLite tables, LanceDB entity embeddings table
- Graph build phases: evidence → claims → entity_graph → centrality → communities → entity_profiles → community_reports → complete
