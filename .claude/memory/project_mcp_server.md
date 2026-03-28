---
name: MCP server implementation status
description: FastMCP 3.0 server with 20 read-only tools (6 corpus + 11 KG + 3 full-pipeline), reviewed 2026-03-28
type: project
---

MCP server (src/lxd/mcp/server.py) uses FastMCP >=3.0, 20 read-only tools.

**Why:** The MCP server is the primary interface for Claude Desktop and other LLM clients to query the LxD corpus and knowledge graph.

**How to apply:**
- 6 corpus tools: corpus_status, get_entity_types, get_related_concepts, search_corpus, find_documents_for_concept, get_corpus_relations
- 11 knowledge graph tools: get_entity_summary, get_community_context, get_similar_entities, search_entities, inspect_evidence, find_path_between_entities, find_weighted_path, get_hub_entities, find_bridge_entities, find_foundational_entities, get_entity_graph_stats
- 3 full-pipeline tools: search_knowledge (graph-augmented synthesis), search_knowledge_deep (same + graph context data), get_graph_overview (KG health)
- All tools are readOnlyHint
- Uses Annotated types with Field(description=...) for parameter docs
- Tool implementations live in src/lxd/mcp/tools.py (separate from server.py)
- Lifespan pattern bootstraps AppContext + IngestPlan at startup
- Graph context augments synthesis when KG is enabled and entities match
