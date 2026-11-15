"""Tests for Central_Configs v5.0.0 ontology-aware query expansion without raw CURIE leakage."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import duckdb
import networkx as nx
import pytest

from lxd.retrieval import expansion
from lxd.settings.models import RuntimeConfig


def test_expand_question_never_injects_raw_curies_into_lexical_terms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph: nx.MultiDiGraph[str] = nx.MultiDiGraph()
    graph.add_node("coe:C05121", node_type="entity")
    graph.add_node("coe:C05103", node_type="entity")
    graph.add_node("coe:C05101", node_type="entity")
    graph.add_edge("coe:C05121", "coe:C05103")
    graph.add_edge("coe:C05103", "coe:C05101")

    entity_by_id: dict[str, dict[str, Any]] = {
        "coe:C05121": {
            "canonical_id": "coe:C05121",
            "label": "Multimedia Principle",
            "aliases": ["Mayer multimedia principle"],
        },
        "coe:C05103": {
            "canonical_id": "coe:C05103",
            "label": "Coherence Principle",
            "aliases": ["extraneous load trimming"],
        },
        "coe:C05101": {
            "canonical_id": "coe:C05101",
            "label": "Cognitive Load Theory",
            "aliases": ["CLT"],
        },
    }

    runtime = SimpleNamespace(
        ontology=SimpleNamespace(
            graph=graph,
            entity_definitions=list(entity_by_id.values()),
        ),
        automaton=object(),
        entity_by_id=entity_by_id,
        concept_surfaces=(),
        entity_catalog={},
    )

    def _ontology_runtime(_config: RuntimeConfig) -> SimpleNamespace:
        return runtime

    def _detect_mentions(
        _question: str, _automaton: Any, **_kwargs: object
    ) -> list[SimpleNamespace]:
        return [SimpleNamespace(entity_id="coe:C05121")]

    monkeypatch.setattr(expansion, "_ontology_runtime", _ontology_runtime)
    monkeypatch.setattr(expansion, "detect_mentions", _detect_mentions)

    config = cast(
        "RuntimeConfig",
        SimpleNamespace(
            expansion=SimpleNamespace(hops=2, max_terms=6),
            paths=SimpleNamespace(data_path=Path("/nonexistent")),
        ),
    )

    outcome = expansion.expand_question("What is the multimedia principle?", config)

    assert outcome.matched_entity_ids == ["coe:C05121"]
    assert "Coherence Principle" in outcome.added_terms
    assert "Cognitive Load Theory" in outcome.added_terms
    assert all(not term.startswith("coe:C") for term in outcome.added_terms)
    assert "coe:C05121" not in outcome.expanded_question
    assert "coe:C05103" not in outcome.expanded_question
    assert "Related concepts:" in outcome.expanded_question


def test_expand_question_matches_craft_concepts_from_duckdb_without_automaton_pollution(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "central.duckdb"
    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            """
            CREATE TABLE all_entities (
                id VARCHAR,
                collection_name VARCHAR,
                pref_label VARCHAR,
                definition VARCHAR
            );
            INSERT INTO all_entities VALUES
                ('coe:C05105', 'cognitive_principles', 'Worked Example Effect', 'Novices learn from worked examples.'),
                ('coe:C05204', 'instructional_methods', 'Faded Guidance', 'Gradually remove worked steps.');

            CREATE TABLE all_terms (
                resource_id VARCHAR,
                written_form VARCHAR,
                term_kind VARCHAR
            );
            INSERT INTO all_terms VALUES
                ('coe:C05105', 'worked examples', 'synonym'),
                ('coe:C05204', 'backward fading', 'synonym');

            CREATE TABLE instructional_methods (
                id VARCHAR,
                broader VARCHAR[]
            );
            INSERT INTO instructional_methods VALUES
                ('coe:C05204', ['coe:C05105']);
            """
        )
    finally:
        con.close()

    config = cast(
        "RuntimeConfig",
        SimpleNamespace(
            expansion=SimpleNamespace(hops=1, max_terms=5),
            paths=SimpleNamespace(
                central_duckdb_path=db_path,
                data_path=tmp_path / "data",
            ),
        ),
    )

    outcome = expansion.expand_question("How do worked examples help novices?", config)

    assert outcome.matched_entity_ids == ["coe:C05105"]
    assert "Faded Guidance" in outcome.added_terms
    assert "backward fading" in outcome.added_terms
    assert all(not term.startswith("coe:C") for term in outcome.added_terms)
