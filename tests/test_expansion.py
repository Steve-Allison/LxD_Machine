from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import networkx as nx
import pytest

from lxd.retrieval import expansion
from lxd.settings.models import RuntimeConfig


def test_expand_question_uses_query_mentions_and_entity_neighbors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph: nx.MultiDiGraph[str] = nx.MultiDiGraph()
    graph.add_node("mayer_principle", node_type="entity")
    graph.add_node("coherence_principle", node_type="entity")
    graph.add_node("multimedia_learning", node_type="entity")
    graph.add_edge("mayer_principle", "coherence_principle")
    graph.add_edge("coherence_principle", "multimedia_learning")

    runtime = SimpleNamespace(
        ontology=SimpleNamespace(
            graph=graph,
            entity_definitions=[
                {
                    "canonical_id": "mayer_principle",
                    "label": "Mayer Principle",
                    "aliases": ["multimedia principle"],
                },
                {
                    "canonical_id": "coherence_principle",
                    "label": "Coherence Principle",
                    "aliases": [],
                },
                {
                    "canonical_id": "multimedia_learning",
                    "label": "Multimedia Learning",
                    "aliases": [],
                },
            ],
        ),
        automaton=object(),
        entity_by_id={
            "mayer_principle": {
                "canonical_id": "mayer_principle",
                "label": "Mayer Principle",
                "aliases": ["multimedia principle"],
            },
            "coherence_principle": {
                "canonical_id": "coherence_principle",
                "label": "Coherence Principle",
                "aliases": [],
            },
            "multimedia_learning": {
                "canonical_id": "multimedia_learning",
                "label": "Multimedia Learning",
                "aliases": [],
            },
        },
    )

    def _ontology_runtime(_config: RuntimeConfig) -> SimpleNamespace:
        return runtime

    def _detect_mentions(_question: str, _automaton: Any) -> list[SimpleNamespace]:
        return [SimpleNamespace(entity_id="mayer_principle")]

    monkeypatch.setattr(expansion, "_ontology_runtime", _ontology_runtime)
    monkeypatch.setattr(expansion, "detect_mentions", _detect_mentions)

    config = cast(
        "RuntimeConfig",
        SimpleNamespace(
            expansion=SimpleNamespace(hops=2, max_terms=4),
            paths=SimpleNamespace(data_path=Path("/nonexistent")),
        ),
    )

    outcome = expansion.expand_question("What is Mayer's principle?", config)

    assert outcome.matched_entity_ids == ["mayer_principle"]
    assert "coherence_principle" in outcome.added_terms
    assert "Multimedia Learning" in outcome.added_terms
    assert "Related concepts:" in outcome.expanded_question
