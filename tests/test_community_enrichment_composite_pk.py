"""Regression: community LLM enrichment must honour the composite PK.

Migration 0008 made ``community_reports`` keyed by
``(community_id, community_level)``. Hierarchical Louvain reuses the same
numeric ``community_id`` at every level, so an UPDATE that filters only on
``community_id`` overwrites every level's ``llm_summary`` with one level's
prose.
"""

import sqlite3
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from lxd.ontology.profiles import enrich_entity_profiles_with_llm
from lxd.settings.models import (
    EmbeddingConfig,
    KnowledgeGraphConfig,
    ModelsConfig,
    OllamaConfig,
    OpenAIEmbeddingConfig,
    RuntimeConfig,
)
from lxd.stores.models import CommunityReportRecord
from lxd.stores.schema import ensure_schema
from lxd.stores.sqlite.kg_profiles import load_community_report, upsert_community_report

pytestmark = [pytest.mark.unit]


@pytest.fixture()
def kg_db(tmp_path: Path) -> Generator[sqlite3.Connection]:
    db_path = tmp_path / "lxd.sqlite3"
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    ensure_schema(connection)
    try:
        yield connection
    finally:
        connection.close()


def _report(*, community_id: int, community_level: int, summary: str) -> CommunityReportRecord:
    return CommunityReportRecord(
        community_id=community_id,
        community_level=community_level,
        member_count=3,
        member_entity_ids_json='["a","b","c"]',
        deterministic_summary=summary,
        llm_summary=None,
        top_entities_json="[]",
        top_claims_json="[]",
        intra_community_edge_count=2,
        source_hash=f"hash-{community_id}-{community_level}",
        generated_at="2026-07-20T00:00:00Z",
        parent_community_id=None,
    )


def _config() -> RuntimeConfig:
    return RuntimeConfig.model_construct(
        models=ModelsConfig.model_construct(embed="text-embedding-3-small", embed_dims=1536),
        embedding=EmbeddingConfig.model_construct(),
        knowledge_graph=KnowledgeGraphConfig(
            claim_extraction_max_concurrent=2,
            claim_extraction_sub_batch_size=10,
        ),
        openai=OpenAIEmbeddingConfig(api_key_env="OPENAI_API_KEY"),
        ollama=OllamaConfig.model_construct(url="http://127.0.0.1:11434"),
    )


def test_community_enrichment_scopes_llm_summary_to_composite_pk(
    kg_db: sqlite3.Connection,
) -> None:
    """Same community_id at two levels must keep distinct llm_summary values."""
    upsert_community_report(kg_db, _report(community_id=0, community_level=0, summary="fine"))
    upsert_community_report(kg_db, _report(community_id=0, community_level=1, summary="coarse"))
    kg_db.commit()

    async def _fake_llm(**kwargs: Any) -> str:
        prompt = str(kwargs.get("user_prompt", ""))
        # Prompt includes "Community {id} (level {level}, …)" — see profiles.py.
        if "level 0" in prompt:
            return "summary-level-0"
        if "level 1" in prompt:
            return "summary-level-1"
        raise AssertionError(f"unexpected enrichment prompt:\n{prompt}")

    with patch(
        "lxd.ontology.profiles.call_with_fallback_async",
        new=AsyncMock(side_effect=_fake_llm),
    ):
        enriched = enrich_entity_profiles_with_llm(kg_db, _config(), force=False)

    assert enriched == 2
    level_0 = load_community_report(kg_db, 0, community_level=0)
    level_1 = load_community_report(kg_db, 0, community_level=1)
    assert level_0 is not None and level_0.llm_summary == "summary-level-0"
    assert level_1 is not None and level_1.llm_summary == "summary-level-1"
