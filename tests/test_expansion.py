from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import networkx as nx

from lxd.retrieval import expansion


def test_expand_question_uses_query_mentions_and_entity_neighbors(monkeypatch) -> None:
    graph = nx.MultiDiGraph()
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

    monkeypatch.setattr(expansion, "_ontology_runtime", lambda config: runtime)
    monkeypatch.setattr(
        expansion,
        "detect_mentions",
        lambda question, automaton: [
            SimpleNamespace(entity_id="mayer_principle"),
        ],
    )

    config = SimpleNamespace(
        expansion=SimpleNamespace(enabled=True, hops=2, max_terms=4),
        paths=SimpleNamespace(data_path=Path("/nonexistent")),
    )

    outcome = expansion.expand_question("What is Mayer's principle?", config)

    assert outcome.matched_entity_ids == ["mayer_principle"]
    assert "coherence_principle" in outcome.added_terms
    assert "Multimedia Learning" in outcome.added_terms
    assert "Related concepts:" in outcome.expanded_question
