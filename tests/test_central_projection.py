"""Projection of a Central Configs distribution and library into runtime records."""

from pathlib import Path

from lxd.ingest.mentions import detect_mentions
from lxd.ontology.loader import load_ontology
from lxd.ontology.matcher import build_automaton
from lxd.settings.loader import resolve_repo_root


def _write_distribution(root: Path, library: Path) -> None:
    (root / "domains" / "learning").mkdir(parents=True)
    (root / "distribution.yaml").write_text("version: 4.0.0\n", encoding="utf-8")
    (root / "domains" / "learning" / "methods.yaml").write_text(
        """
_meta:
  file_id: coe:file/domain/learning/methods
  title: Methods
  last_updated: '2026-09-23'
  authority: Central_Configs
  scope: domain
  ontology_domain: learning
  governing_class: coe:DomainModule
module_id: coe:module/learning/methods
domain: learning
entity_collections:
- id: coe:artifact/learning/methods
  label: Methods
  description: A collection.
  home_domain: learning
  status: canonical
  entities:
  - id: coe:entity/scaffolding
    label: Scaffolding
    description: Temporary instructional support.
    home_domain: learning
    status: canonical
    terms:
    - literal: scaffold
      term_type: short_form
      term_status: admitted
    edges:
    - subject_id: coe:entity/scaffolding
      predicate: coe:supports
      object_id: coe:entity/fading
      statement_id: coe:edge/scaffolding-supports-fading
      home_domain: learning
  - id: coe:entity/fading
    label: Fading
    description: Removing support as competence grows.
    home_domain: learning
    status: canonical
""".strip(),
        encoding="utf-8",
    )
    (library / "data" / "learning").mkdir(parents=True)
    (library / "distribution.yaml").write_text("version: 1.0.0\n", encoding="utf-8")
    (library / "data" / "learning" / "recognition.yaml").write_text(
        """
module_id: lib:module/learning/recognition
domain: learning
recognition_calibrations:
  lib:recognition/learning/scaffolding:
    label: Scaffolding Recognition
    description: How to recognise scaffolding.
    about: [coe:entity/scaffolding]
    status: canonical
    confidence_threshold: 0.7
    surfaces: ["zone of proximal development"]
    negative_surfaces: ["construction scaffold"]
    recognition_patterns: ["temporary (support|scaffold)"]
    context_anchors: ["learner"]
    anchor_window_tokens: 8
prescriptions:
  lib:prescription/learning/fade_support:
    label: Fade Support
    description: Withdraw the scaffold.
    about: [coe:entity/scaffolding]
    status: canonical
    statement: Fade the scaffold as the learner's competence grows.
thresholds:
  lib:threshold/learning/scaffold_steps:
    label: Scaffold Steps
    description: Steps of support.
    about: [coe:entity/scaffolding]
    status: canonical
    quantity: support steps
    unit: steps
    bound_kind: ceiling
    maximum_value: 4
""".strip(),
        encoding="utf-8",
    )


def test_central_projection_builds_entities_edges_and_craft(tmp_path: Path) -> None:
    ontology = tmp_path / "ontology"
    library = tmp_path / "library"
    _write_distribution(ontology, library)

    result = load_ontology(ontology, ["**/*.yaml"], [], library_root=library)

    by_id = {entity["canonical_id"]: entity for entity in result.entity_definitions}
    scaffolding = by_id["coe:entity/scaffolding"]
    assert scaffolding["label"] == "Scaffolding"
    assert "scaffold" in scaffolding["aliases"]
    assert "zone of proximal development" in scaffolding["indicators"]
    assert "Fade the scaffold" in scaffolding["craft_summary"]
    assert "support steps ceiling 4 steps" in scaffolding["craft_summary"]
    assert any(record.relation_type == "supports" for record in result.relation_records)
    assert not any(record.normalized_term.startswith("coe:") for record in result.matcher_records)
    assert result.recognition_patterns
    assert "construction scaffold" in result.suppressed_terms["coe:entity/scaffolding"]


def test_recognition_pattern_anchor_and_negative_surface(tmp_path: Path) -> None:
    ontology = tmp_path / "ontology"
    library = tmp_path / "library"
    _write_distribution(ontology, library)
    result = load_ontology(ontology, ["**/*.yaml"], [], library_root=library)
    automaton = build_automaton(result.matcher_records)

    kept = detect_mentions(
        "The learner needs temporary support during practice.",
        automaton,
        recognition_patterns=result.recognition_patterns,
        suppressed_terms=result.suppressed_terms,
        anchor_constraints=result.anchor_constraints,
    )
    assert any(mention.entity_id == "coe:entity/scaffolding" for mention in kept)

    rejected = detect_mentions(
        "A construction scaffold stood beside the building.",
        automaton,
        recognition_patterns=result.recognition_patterns,
        suppressed_terms=result.suppressed_terms,
        anchor_constraints=result.anchor_constraints,
    )
    assert all(mention.entity_id != "coe:entity/scaffolding" for mention in rejected)


def test_vendored_distribution_projects_central_identifiers() -> None:
    repo_root = resolve_repo_root(Path.cwd())
    result = load_ontology(
        repo_root / "ontology" / "vendor" / "central-configs",
        ["**/*.yaml"],
        [],
        library_root=repo_root / "library" / "vendor" / "central-library",
    )

    ids = {entity["canonical_id"] for entity in result.entity_definitions}
    assert "coe:entity/learning/term/scaffolding" in ids
    assert "coe:concept/argumentation_schemes_taxonomy/scheme/expert_opinion" in ids
    assert any(record.relation_type == "supports" for record in result.relation_records)
    assert any(entity.get("craft_summary") for entity in result.entity_definitions)
    assert result.recognition_patterns
