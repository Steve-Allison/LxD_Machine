from __future__ import annotations

from pathlib import Path

from lxd.ontology.graph import direct_neighbors
from lxd.ontology.loader import load_ontology
from lxd.settings.loader import resolve_repo_root


def test_load_ontology_uses_file_entity_and_taxonomy_relationships(tmp_path: Path) -> None:
    ontology_root = tmp_path / "ontology"
    (ontology_root / "entities").mkdir(parents=True)
    (ontology_root / "taxonomy").mkdir(parents=True)
    (ontology_root / "methodology").mkdir(parents=True)

    (ontology_root / "ontology.yaml").write_text(
        """
_meta:
  id: ontology
  purpose: reference
  relationships:
    - target: entities/sample_entities.yaml
      type: depends_on
file_relationships:
  depends_on:
    description: dependency
entity_relations:
  supports:
    inverse: supported_by
entity_relation_weights:
  required:
    description: required
""".strip(),
        encoding="utf-8",
    )

    (ontology_root / "taxonomy" / "slide_types_taxonomy.yaml").write_text(
        """
_meta:
  id: slide_types_taxonomy
  purpose: taxonomy
  relationships:
    - target: entities/sample_entities.yaml
      type: depends_on
architecture_stack:
  label: Architecture Stack
""".strip(),
        encoding="utf-8",
    )

    (ontology_root / "methodology" / "pedagogical_frameworks.yaml").write_text(
        """
_meta:
  id: pedagogical_frameworks
  purpose: methodology
gagne_nine_events:
  label: Gagne
""".strip(),
        encoding="utf-8",
    )

    (ontology_root / "entities" / "sample_entities.yaml").write_text(
        """
_meta:
  id: sample_entities
  purpose: entity
  relationships:
    - target: taxonomy/slide_types_taxonomy.yaml
      type: depends_on
entity_types:
  parent_entity:
    description: Parent
    aliases: ["top level"]
    indicators: ["parent"]
  child_entity:
    description: Child
    aliases: ["child alias"]
    indicators: ["child"]
    parent_entity: parent_entity
    relates_to:
      - target: parent_entity
        relation: supports
        weight: required
        description: Child supports parent
    taxonomy_mapping:
      - taxonomy: slide_types_taxonomy
        dimension: slide_category
        values: [structural]
        strength: primary
    maps_to_taxonomy_types: [architecture_stack]
    taxonomy_reference: gagne_nine_events
    validate_against_taxonomy: true
""".strip(),
        encoding="utf-8",
    )

    result = load_ontology(ontology_root, ["**/*.yaml"], [])

    relation_types = {record.relation_type for record in result.relation_records}
    assert "depends_on" in relation_types
    assert "parent_entity" in relation_types
    assert "supports" in relation_types
    assert "maps_to_taxonomy_value" in relation_types
    assert "maps_to_taxonomy_type" in relation_types
    assert "references_taxonomy" in relation_types
    assert "validated_against_taxonomy" in relation_types
    assert result.coverage_report.unclassified_paths == []
    assert result.validation_issues == []

    file_records = [record for record in result.metadata_records if record.record_kind == "file"]
    entity_records = [record for record in result.metadata_records if record.record_kind == "entity"]
    assert len(file_records) == 4
    assert {record.entity_id for record in entity_records} == {"child_entity", "parent_entity"}

    neighbors = direct_neighbors(result.graph, "child_entity")
    neighbor_types = {record["neighbor_node_type"] for record in neighbors}
    assert "entity" in neighbor_types
    assert "taxonomy_value" in neighbor_types
    assert "taxonomy_type" in neighbor_types
    assert "ontology_file" in neighbor_types


def test_real_yaml_tree_has_no_unclassified_ontology_paths() -> None:
    repo_root = resolve_repo_root(Path.cwd())
    result = load_ontology(repo_root / "Yamls", ["**/*.yaml"], [])

    assert result.coverage_report.unclassified_paths == []
    assert result.coverage_report.classification_counts["graph_input"] > 0
    assert result.coverage_report.classification_counts["matcher_input"] > 0
    assert result.coverage_report.classification_counts["metadata_input"] > 0
    assert any(record.relation_type == "depends_on" for record in result.relation_records)
    assert any(record.relation_type == "maps_to_taxonomy_value" for record in result.relation_records)
    assert any(record.relation_type == "references_taxonomy" for record in result.relation_records)


def test_real_yaml_graph_includes_file_level_relationships() -> None:
    repo_root = resolve_repo_root(Path.cwd())
    result = load_ontology(repo_root / "Yamls", ["**/*.yaml"], [])

    file_edges = [
        record
        for record in result.relation_records
        if record.origin_kind == "file_meta"
        and record.source_file_rel_path == "entities/writing_entities.yaml"
        and record.relation_type == "depends_on"
    ]

    assert any(record.target_file_rel_path == "methodology/editorial_standards.yaml" for record in file_edges)
    assert any(record.target_file_rel_path == "methodology/style_guide_standards.yaml" for record in file_edges)
