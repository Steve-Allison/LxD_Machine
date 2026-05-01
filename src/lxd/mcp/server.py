"""Launch the MCP server process for corpus tools."""

from __future__ import annotations

import argparse
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from fastmcp import Context, FastMCP
from pydantic import Field

from lxd.app.bootstrap import AppContext, bootstrap_app
from lxd.ingest.pipeline import IngestPlan, build_ingest_plan
from lxd.mcp.async_runtime import run_tool
from lxd.mcp.tools import (
    corpus_status_tool,
    find_bridge_entities_tool,
    find_documents_for_concept_tool,
    find_foundational_entities_tool,
    find_path_between_entities_tool,
    find_weighted_path_tool,
    get_community_context_tool,
    get_corpus_relations_tool,
    get_entity_graph_stats_tool,
    get_entity_summary_tool,
    get_entity_types_tool,
    get_graph_overview_tool,
    get_hub_entities_tool,
    get_related_concepts_tool,
    get_similar_entities_tool,
    inspect_evidence_tool,
    search_corpus_tool,
    search_entities_tool,
    search_knowledge_deep_tool,
    search_knowledge_tool,
)

_READ_ONLY = {"readOnlyHint": True}
_LIFESPAN_KEY = "lxd"


@dataclass(frozen=True, slots=True)
class _LxDLifespan:
    """Immutable bundle of server-scoped resources, initialised once at startup."""

    app_context: AppContext
    ingest_plan: IngestPlan


def _make_lifespan(cwd: Path, profile: str | None, config_path: Path | None):
    @asynccontextmanager
    async def lifespan(server: FastMCP) -> AsyncGenerator[dict[str, _LxDLifespan]]:
        app_context = bootstrap_app(cwd, profile=profile, config_path=config_path)
        ingest_plan = build_ingest_plan(app_context.config)
        yield {_LIFESPAN_KEY: _LxDLifespan(app_context=app_context, ingest_plan=ingest_plan)}

    return lifespan


def _lxd(ctx: Context) -> _LxDLifespan:
    """Extract the typed lifespan bundle from the request context."""
    lxd = ctx.lifespan_context.get(_LIFESPAN_KEY)
    if not isinstance(lxd, _LxDLifespan):
        raise RuntimeError(
            "LxD lifespan context is not available. "
            "Ensure the server was started via create_server() or main()."
        )
    return lxd


def _tool_timeout(lxd: _LxDLifespan) -> float:
    """Resolve the current tool timeout from the config, with a safe default."""
    mcp_cfg = lxd.app_context.config.mcp
    return float(mcp_cfg.tool_timeout_secs)


def create_server(
    cwd: Path | None = None,
    profile: str | None = None,
    config_path: Path | None = None,
) -> FastMCP:
    """Create and return a fully configured LxD MCP server instance.

    Bootstraps the app context and ingest plan eagerly during the lifespan
    startup phase, so any misconfiguration fails immediately at launch rather
    than on the first tool call.

    Every tool is registered as ``async def`` so the FastMCP event loop can
    interleave requests; synchronous store work runs in a worker thread via
    :func:`lxd.mcp.async_runtime.run_tool`, which also enforces a per-tool
    timeout sourced from :class:`lxd.settings.models.MCPConfig`.

    This is the single construction entry point. ``main()`` calls it at
    startup; tests can call it with a custom ``cwd`` / ``config_path`` to get
    an isolated server without touching any global state.
    """
    mcp = FastMCP(
        "lxd-machine",
        lifespan=_make_lifespan(cwd or Path.cwd(), profile, config_path),
    )

    @mcp.tool(annotations=_READ_ONLY)
    async def corpus_status(ctx: Context) -> dict[str, object]:
        """Return a health snapshot of the LxD corpus and ontology.

        Reports document counts (total, text, asset), chunk and entity-mention
        counts, ontology file and entity counts, matcher term counts, hash
        fingerprints for drift detection, and any validation or config-drift
        warnings. Use this first to confirm the corpus is ingested and healthy
        before running searches.
        """
        lxd = _lxd(ctx)
        return await run_tool(
            "corpus_status",
            lambda: corpus_status_tool(lxd.app_context, lxd.ingest_plan),
            timeout_secs=_tool_timeout(lxd),
        )

    @mcp.tool(annotations=_READ_ONLY)
    async def get_entity_types(ctx: Context) -> list[str]:
        """Return a sorted list of all canonical entity IDs defined in the ontology.

        Use the returned IDs as valid values for the ``entity_id`` parameter of
        ``get_related_concepts`` and ``find_documents_for_concept``.
        """
        lxd = _lxd(ctx)
        return await run_tool(
            "get_entity_types",
            lambda: get_entity_types_tool(lxd.ingest_plan),
            timeout_secs=_tool_timeout(lxd),
        )

    @mcp.tool(annotations=_READ_ONLY)
    async def get_related_concepts(
        entity_id: Annotated[
            str,
            Field(
                description=(
                    "Canonical entity ID to look up (e.g. 'bloom_remember'). "
                    "Must be non-empty. Call get_entity_types to list valid IDs."
                )
            ),
        ],
        ctx: Context,
    ) -> list[dict[str, object]]:
        """Return the direct neighbours of an entity in the ontology graph.

        Each result dict contains ``entity_id``, ``relation``, and ``direction``
        keys describing a single edge. Returns an empty list if the entity is
        not found in the graph.
        """
        lxd = _lxd(ctx)
        return await run_tool(
            "get_related_concepts",
            lambda: get_related_concepts_tool(lxd.ingest_plan, entity_id),
            timeout_secs=_tool_timeout(lxd),
        )

    @mcp.tool(annotations=_READ_ONLY)
    async def search_corpus(
        terms: Annotated[
            str,
            Field(description="Natural-language query or keywords to search for in the corpus."),
        ],
        ctx: Context,
        domain: Annotated[
            str | None,
            Field(
                description=(
                    "Optional domain filter (e.g. 'cognitive_load'). "
                    "Pass null to search across all domains."
                )
            ),
        ] = None,
        limit: Annotated[
            int,
            Field(description="Maximum number of ranked chunks to return.", ge=1, le=100),
        ] = 10,
    ) -> list[dict[str, object]]:
        """Search the corpus using semantic similarity and return ranked text chunks.

        Performs a vector search over ingested document chunks. Each result
        includes ``chunk_id``, ``document_id``, ``citation_label``,
        ``source_rel_path``, ``score``, ``text``, and ``metadata_json``.
        Results are ordered highest-score first.
        """
        lxd = _lxd(ctx)
        return await run_tool(
            "search_corpus",
            lambda: search_corpus_tool(lxd.app_context, terms, domain, limit),
            timeout_secs=_tool_timeout(lxd),
        )

    @mcp.tool(annotations=_READ_ONLY)
    async def find_documents_for_concept(
        entity_id: Annotated[
            str,
            Field(
                description=(
                    "Canonical entity ID to find documents for (e.g. 'bloom_apply'). "
                    "Must be non-empty. Call get_entity_types to list valid IDs."
                )
            ),
        ],
        ctx: Context,
        hops: Annotated[
            int,
            Field(
                description=(
                    "Number of graph hops to expand from the entity before searching. "
                    "1 = direct neighbours only."
                ),
                ge=1,
                le=5,
            ),
        ] = 1,
        limit: Annotated[
            int,
            Field(description="Maximum number of document chunks to return.", ge=1, le=100),
        ] = 10,
    ) -> list[dict[str, object]]:
        """Find document chunks that mention a concept or its graph neighbours.

        Expands ``entity_id`` outward by ``hops`` edges in the ontology graph,
        then retrieves corpus chunks that contain entity-mention annotations for
        any of the resulting entity IDs. Each result includes ``chunk_id``,
        ``document_id``, ``citation_label``, ``source_rel_path``, ``score``,
        ``entity_match_count``, ``matched_from_total``, ``text``, and
        ``metadata_json``. Returns an empty list if the entity is not in the
        graph.
        """
        lxd = _lxd(ctx)
        return await run_tool(
            "find_documents_for_concept",
            lambda: find_documents_for_concept_tool(
                lxd.app_context, lxd.ingest_plan, entity_id, hops, limit
            ),
            timeout_secs=_tool_timeout(lxd),
        )

    @mcp.tool(annotations=_READ_ONLY)
    async def get_corpus_relations(
        entity_id: Annotated[
            str,
            Field(
                description=(
                    "Canonical entity ID to find corpus-extracted relations for "
                    "(e.g. 'bloom_apply'). Call get_entity_types to list valid IDs."
                )
            ),
        ],
        ctx: Context,
        limit: Annotated[
            int,
            Field(description="Maximum number of relations to return.", ge=1, le=200),
        ] = 50,
    ) -> list[dict[str, object]]:
        """Return semantic relations extracted from the corpus for an entity.

        Unlike ``get_related_concepts`` (which returns hand-coded ontology edges),
        this tool returns relations learned from document text during ingest —
        e.g. ``(bloom_apply) → [requires] → (cognitive_load)`` as stated in a
        specific chunk. Each result includes ``subject``, ``predicate``, ``object``,
        ``confidence``, ``source_rel_path``, and ``chunk_id``.

        Returns an empty list if relation extraction has not been run yet.
        """
        lxd = _lxd(ctx)
        return await run_tool(
            "get_corpus_relations",
            lambda: get_corpus_relations_tool(lxd.app_context, entity_id, limit),
            timeout_secs=_tool_timeout(lxd),
        )

    @mcp.tool(annotations=_READ_ONLY)
    async def get_entity_summary(
        entity_id: Annotated[str, Field(description="Canonical entity ID.")],
        ctx: Context,
    ) -> dict[str, object]:
        """Return the full entity profile including centrality, claims, and community."""
        lxd = _lxd(ctx)
        return await run_tool(
            "get_entity_summary",
            lambda: get_entity_summary_tool(lxd.app_context, entity_id),
            timeout_secs=_tool_timeout(lxd),
        )

    @mcp.tool(annotations=_READ_ONLY)
    async def get_community_context(
        entity_id: Annotated[str, Field(description="Canonical entity ID to find community for.")],
        ctx: Context,
    ) -> dict[str, object]:
        """Return the community report for an entity's community."""
        lxd = _lxd(ctx)
        return await run_tool(
            "get_community_context",
            lambda: get_community_context_tool(lxd.app_context, entity_id),
            timeout_secs=_tool_timeout(lxd),
        )

    @mcp.tool(annotations=_READ_ONLY)
    async def get_similar_entities(
        entity_id: Annotated[
            str, Field(description="Canonical entity ID to find similar entities for.")
        ],
        ctx: Context,
        limit: Annotated[int, Field(description="Maximum results.", ge=1, le=50)] = 10,
    ) -> list[dict[str, object]]:
        """Return entities most similar to the given entity via vector embedding similarity."""
        lxd = _lxd(ctx)
        return await run_tool(
            "get_similar_entities",
            lambda: get_similar_entities_tool(lxd.app_context, entity_id, limit),
            timeout_secs=_tool_timeout(lxd),
        )

    @mcp.tool(annotations=_READ_ONLY)
    async def search_entities(
        query: Annotated[str, Field(description="Search query for entity name/alias.")],
        ctx: Context,
        limit: Annotated[int, Field(description="Maximum results.", ge=1, le=100)] = 20,
    ) -> list[dict[str, object]]:
        """Search entity profiles by name or alias, ranked by PageRank."""
        lxd = _lxd(ctx)
        return await run_tool(
            "search_entities",
            lambda: search_entities_tool(lxd.app_context, query, limit),
            timeout_secs=_tool_timeout(lxd),
        )

    @mcp.tool(annotations=_READ_ONLY)
    async def inspect_evidence(
        relation_id: Annotated[str, Field(description="Canonical relation ID to audit.")],
        ctx: Context,
    ) -> list[dict[str, object]]:
        """Return all evidence records for a canonical relation.

        Includes surface forms and chunk provenance for downstream auditing.
        """
        lxd = _lxd(ctx)
        return await run_tool(
            "inspect_evidence",
            lambda: inspect_evidence_tool(lxd.app_context, relation_id),
            timeout_secs=_tool_timeout(lxd),
        )

    @mcp.tool(annotations=_READ_ONLY)
    async def find_path_between_entities(
        source: Annotated[str, Field(description="Source entity ID.")],
        target: Annotated[str, Field(description="Target entity ID.")],
        ctx: Context,
        max_hops: Annotated[int, Field(description="Maximum path length.", ge=1, le=10)] = 5,
    ) -> dict[str, object]:
        """Find shortest unweighted path between two entities."""
        lxd = _lxd(ctx)
        return await run_tool(
            "find_path_between_entities",
            lambda: find_path_between_entities_tool(
                lxd.app_context, lxd.ingest_plan, source, target, max_hops
            ),
            timeout_secs=_tool_timeout(lxd),
        )

    @mcp.tool(annotations=_READ_ONLY)
    async def find_weighted_path(
        source: Annotated[str, Field(description="Source entity ID.")],
        target: Annotated[str, Field(description="Target entity ID.")],
        ctx: Context,
    ) -> dict[str, object]:
        """Find confidence-weighted Dijkstra shortest path between two entities."""
        lxd = _lxd(ctx)
        return await run_tool(
            "find_weighted_path",
            lambda: find_weighted_path_tool(lxd.app_context, lxd.ingest_plan, source, target),
            timeout_secs=_tool_timeout(lxd),
        )

    @mcp.tool(annotations=_READ_ONLY)
    async def get_hub_entities(
        ctx: Context,
        limit: Annotated[int, Field(description="Maximum results.", ge=1, le=100)] = 20,
    ) -> list[dict[str, object]]:
        """Return top entities by PageRank (hub concepts)."""
        lxd = _lxd(ctx)
        return await run_tool(
            "get_hub_entities",
            lambda: get_hub_entities_tool(lxd.app_context, limit),
            timeout_secs=_tool_timeout(lxd),
        )

    @mcp.tool(annotations=_READ_ONLY)
    async def find_bridge_entities(
        ctx: Context,
        limit: Annotated[int, Field(description="Maximum results.", ge=1, le=100)] = 20,
    ) -> list[dict[str, object]]:
        """Return top entities by betweenness centrality (bridge concepts)."""
        lxd = _lxd(ctx)
        return await run_tool(
            "find_bridge_entities",
            lambda: find_bridge_entities_tool(lxd.app_context, limit),
            timeout_secs=_tool_timeout(lxd),
        )

    @mcp.tool(annotations=_READ_ONLY)
    async def find_foundational_entities(
        ctx: Context,
        limit: Annotated[int, Field(description="Maximum results.", ge=1, le=100)] = 20,
    ) -> list[dict[str, object]]:
        """Return top entities by closeness centrality (foundational concepts)."""
        lxd = _lxd(ctx)
        return await run_tool(
            "find_foundational_entities",
            lambda: find_foundational_entities_tool(lxd.app_context, limit),
            timeout_secs=_tool_timeout(lxd),
        )

    @mcp.tool(annotations=_READ_ONLY)
    async def get_entity_graph_stats(ctx: Context) -> dict[str, object]:
        """Return knowledge graph statistics: node/edge counts, community count, version."""
        lxd = _lxd(ctx)
        return await run_tool(
            "get_entity_graph_stats",
            lambda: get_entity_graph_stats_tool(lxd.app_context),
            timeout_secs=_tool_timeout(lxd),
        )

    @mcp.tool(annotations=_READ_ONLY)
    async def search_knowledge(
        question: Annotated[
            str,
            Field(description="Natural-language question to answer using the knowledge base."),
        ],
        ctx: Context,
        domain: Annotated[
            str | None,
            Field(description="Optional domain filter. Pass null to search all domains."),
        ] = None,
    ) -> dict[str, object]:
        """Answer a question using semantic retrieval with graph-augmented synthesis.

        Performs dense vector search, reranking, ontology expansion, and
        graph context augmentation (entity profiles, community reports, claims)
        before synthesising an answer via LLM. Returns ``answer_status``
        (``answered``, ``no_results``, ``insufficient_evidence``, or
        ``synthesis_unavailable``), ``answer_text``, ``citations``, and
        query ``metadata`` including matched entities and expansion terms.
        """
        lxd = _lxd(ctx)
        return await run_tool(
            "search_knowledge",
            lambda: search_knowledge_tool(lxd.app_context, question, domain),
            timeout_secs=_tool_timeout(lxd),
        )

    @mcp.tool(annotations=_READ_ONLY)
    async def search_knowledge_deep(
        question: Annotated[
            str,
            Field(description="Complex question requiring deep graph context."),
        ],
        ctx: Context,
        domain: Annotated[
            str | None,
            Field(description="Optional domain filter. Pass null to search all domains."),
        ] = None,
    ) -> dict[str, object]:
        """Answer a question with full graph context returned alongside the answer.

        Like ``search_knowledge`` but also returns structured ``graph_context``
        data: entity profiles with centrality scores, community reports, and
        claims for matched entities. Use this for complex queries where the
        caller needs to inspect the graph evidence behind the answer.
        """
        lxd = _lxd(ctx)
        return await run_tool(
            "search_knowledge_deep",
            lambda: search_knowledge_deep_tool(lxd.app_context, question, domain),
            timeout_secs=_tool_timeout(lxd),
        )

    @mcp.tool(annotations=_READ_ONLY)
    async def get_graph_overview(ctx: Context) -> dict[str, object]:
        """Return a high-level overview of the knowledge graph.

        Reports the graph version, build timestamp,
        community algorithm, and counts for entity profiles, communities,
        community reports, canonical relations, relation evidence, and claims.
        Use this to check graph health before running graph-dependent queries.
        """
        lxd = _lxd(ctx)
        return await run_tool(
            "get_graph_overview",
            lambda: get_graph_overview_tool(lxd.app_context),
            timeout_secs=_tool_timeout(lxd),
        )

    return mcp


def main() -> None:
    """Run the module entrypoint."""
    parser = argparse.ArgumentParser(description="LxD Machine MCP server (stdio transport).")
    parser.add_argument("--profile", help="Named config profile to load.")
    parser.add_argument("--config", help="Explicit path to a config YAML file.")
    args = parser.parse_args()
    config_path = Path(args.config).resolve() if args.config else None
    mcp = create_server(profile=args.profile, config_path=config_path)
    mcp.run()


if __name__ == "__main__":
    main()
