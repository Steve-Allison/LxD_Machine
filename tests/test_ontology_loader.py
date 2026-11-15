"""Tests for Central_Configs v5.0.0 DuckDB ontology loading and graph construction."""

from pathlib import Path

import duckdb

from lxd.ontology.graph import direct_neighbors
from lxd.ontology.loader import load_ontology


def test_load_ontology_projects_typed_relations_and_graph_neighbors(tmp_path: Path) -> None:
    central_root = tmp_path / "Central_Configs"
    db_path = central_root / "dist" / "build" / "central.duckdb"
    db_path.parent.mkdir(parents=True)

    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            """
            CREATE TABLE all_entities (
                id VARCHAR,
                collection_name VARCHAR,
                pref_label VARCHAR,
                definition VARCHAR,
                status VARCHAR,
                replaced_by VARCHAR,
                wiki_slug VARCHAR,
                facets_json VARCHAR
            );
            INSERT INTO all_entities VALUES
                ('coe:C00001', 'organisations', 'Adobe', 'Parent organisation.', 'active', NULL, NULL, '["portfolio"]'),
                ('coe:C01001', 'offerings', 'Adobe Analytics', 'Analytics offering.', 'active', NULL, NULL, '["portfolio"]'),
                ('coe:C01099', 'offerings', 'Competitor Analytics', 'Rival offering.', 'active', NULL, NULL, '["portfolio"]'),
                ('coe:C02001', 'competitive_positions', 'Analytics vs Competitor', 'Head-to-head position.', 'active', NULL, NULL, '["evangelism"]'),
                ('coe:C06001', 'story_frameworks', 'SOAR Framework', 'Case story structure.', 'active', NULL, 'soar-framework', '["rhetoric"]'),
                ('coe:C06011', 'framework_stages', 'Situation Stage', 'Opening stage of SOAR.', 'active', NULL, NULL, '["rhetoric"]');

            CREATE TABLE offerings (
                id VARCHAR,
                offered_by VARCHAR
            );
            INSERT INTO offerings VALUES
                ('coe:C01001', 'coe:C00001');

            CREATE TABLE framework_stages (
                id VARCHAR,
                stage_of_framework VARCHAR,
                broader VARCHAR[]
            );
            INSERT INTO framework_stages VALUES
                ('coe:C06011', 'coe:C06001', ['coe:C06001']);

            CREATE TABLE competitive_positions (
                id VARCHAR,
                subject_offering VARCHAR,
                competitor_offering VARCHAR
            );
            INSERT INTO competitive_positions VALUES
                ('coe:C02001', 'coe:C01001', 'coe:C01099');
            """
        )
    finally:
        con.close()

    result = load_ontology(central_root)

    assert result.db_path == db_path
    assert result.coverage_report.unclassified_paths == []
    assert result.coverage_report.discovered_path_count == 4
    assert result.validation_issues == []

    relation_types = {record.relation_type for record in result.relation_records}
    assert {
        "offered_by",
        "stage_of_framework",
        "broader",
        "subject_offering",
        "competitor_offering",
        "competes_with",
    }.issubset(relation_types)

    entity_records = [
        record for record in result.metadata_records if record.record_kind == "entity"
    ]
    assert len(entity_records) == 6

    neighbors = direct_neighbors(result.graph, "coe:C01001")
    neighbor_ids = {record["neighbor_node_id"] for record in neighbors}
    assert {"coe:C00001", "coe:C01099", "coe:C02001"}.issubset(neighbor_ids)
    assert {record["neighbor_node_type"] for record in neighbors} == {"entity"}


def test_load_ontology_records_dangling_curie_validation_issue(tmp_path: Path) -> None:
    db_path = tmp_path / "dangling.duckdb"
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
                ('coe:C01001', 'offerings', 'Adobe Analytics', 'Analytics offering.');

            CREATE TABLE offerings (
                id VARCHAR,
                offered_by VARCHAR
            );
            INSERT INTO offerings VALUES
                ('coe:C01001', 'coe:C99999');
            """
        )
    finally:
        con.close()

    result = load_ontology(db_path)

    assert len(result.validation_issues) == 1
    assert result.validation_issues[0].issue_kind == "dangling_relation_target"
    assert "coe:C99999" in result.validation_issues[0].message
