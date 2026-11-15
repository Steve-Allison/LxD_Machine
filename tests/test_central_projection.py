"""Projection of Central_Configs v5.0.0 DuckDB authority into runtime records."""

import json
from pathlib import Path

import duckdb
import pytest

from lxd.ingest.mentions import detect_mentions
from lxd.ontology.central_projection import format_ontology_context, search_central_corpus
from lxd.ontology.loader import load_ontology
from lxd.ontology.matcher import build_automaton

CENTRAL_CONFIGS_ROOT = Path(__file__).resolve().parents[2] / "Central_Configs"


def _build_fixture_duckdb(db_path: Path) -> Path:
    con = duckdb.connect(str(db_path))
    try:
        con.execute("INSTALL fts; LOAD fts;")
        con.execute(
            """
            CREATE TABLE all_entities (
                id VARCHAR,
                collection_name VARCHAR,
                pref_label VARCHAR,
                definition VARCHAR,
                status VARCHAR,
                replaced_by VARCHAR,
                volatility_tier VARCHAR,
                last_verified_date VARCHAR,
                source_uri VARCHAR,
                wiki_slug VARCHAR,
                facets_json VARCHAR
            );
            INSERT INTO all_entities VALUES
                ('coe:C00001', 'organisations', 'Adobe', 'Enterprise software vendor.', 'active', NULL, 'stable', '2026-10-01', NULL, NULL, '["portfolio"]'),
                ('coe:C01008', 'offerings', 'Adobe Target', 'Personalisation and experimentation offering.', 'active', NULL, 'moderate', '2026-10-01', NULL, NULL, '["portfolio"]'),
                ('coe:C05105', 'cognitive_principles', 'Worked Example Effect', 'Novices learn faster from worked examples.', 'active', NULL, 'stable', '2026-10-01', NULL, 'worked-example-effect', '["pedagogy"]'),
                ('coe:C05301', 'pedagogical_effect_rules', 'Expertise Reversal on Worked Examples', 'Fade worked examples for experts.', 'active', NULL, 'stable', '2026-10-01', NULL, NULL, '["pedagogy"]'),
                ('coe:C09001', 'cross_domain_bridges', 'Worked Example to Demo Script Bridge', 'Operationalise worked examples in product demos.', 'active', NULL, 'stable', '2026-10-01', NULL, NULL, '["pedagogy"]'),
                ('coe:C06002', 'story_frameworks', 'Minto Pyramid', 'Answer-first structured argument.', 'active', NULL, 'stable', '2026-10-01', NULL, 'pyramid-principle', '["rhetoric"]');
            """
        )
        con.execute(
            """
            CREATE TABLE all_terms (
                resource_id VARCHAR,
                pref_label VARCHAR,
                collection_name VARCHAR,
                written_form VARCHAR,
                term_kind VARCHAR,
                term_status VARCHAR,
                spoken_form VARCHAR,
                usage_note VARCHAR
            );
            INSERT INTO all_terms VALUES
                ('coe:C05105', 'Worked Example Effect', 'cognitive_principles', 'worked example', 'synonym', 'preferred', NULL, NULL),
                ('coe:C01008', 'Adobe Target', 'offerings', 'Target', 'short_form', 'context_only', NULL, NULL);
            """
        )
        con.execute(
            """
            CREATE TABLE match_lexicon (
                curie VARCHAR,
                pref_label VARCHAR,
                collection_name VARCHAR,
                surface VARCHAR,
                match_policy VARCHAR,
                case_sensitive BOOLEAN,
                anchor_window_tokens INTEGER,
                requires_llm_disambiguation BOOLEAN,
                context_anchors_json VARCHAR,
                negative_phrases_json VARCHAR
            );
            INSERT INTO match_lexicon VALUES
                ('coe:C00001', 'Adobe', 'organisations', 'Adobe', 'unambiguous', FALSE, 25, FALSE, '[]', '["adobe brick"]'),
                ('coe:C01008', 'Adobe Target', 'offerings', 'Adobe Target', 'unambiguous', FALSE, 25, FALSE, '[]', '[]'),
                ('coe:C01008', 'Adobe Target', 'offerings', 'Target', 'context_anchored', TRUE, 12, TRUE, '["Adobe", "A/B"]', '["Target store"]');
            """
        )
        con.execute(
            """
            CREATE TABLE offerings (
                id VARCHAR,
                offered_by VARCHAR,
                broader VARCHAR[]
            );
            INSERT INTO offerings VALUES ('coe:C01008', 'coe:C00001', []);
            """
        )
        con.execute(
            """
            CREATE TABLE pedagogical_effect_rules (
                id VARCHAR,
                pref_label VARCHAR,
                definition VARCHAR,
                governed_principle_or_method VARCHAR,
                conditioned_expertise_tier VARCHAR,
                conditioned_language_baseline VARCHAR,
                modality_or_task_condition VARCHAR,
                effect_polarity VARCHAR,
                adapted_prescription VARCHAR,
                evidence_grade VARCHAR,
                citations VARCHAR[]
            );
            INSERT INTO pedagogical_effect_rules VALUES (
                'coe:C05301',
                'Expertise Reversal on Worked Examples',
                'Worked examples reverse for experts.',
                'coe:C05105',
                'expert',
                NULL,
                'problem solving',
                'reverses_to_harmful',
                'Replace full worked examples with open problem-solving tasks for experts.',
                'meta_analysis',
                ['Kalyuga 2003']
            );
            """
        )
        con.execute(
            """
            CREATE TABLE cross_domain_bridges (
                id VARCHAR,
                pref_label VARCHAR,
                definition VARCHAR,
                source_concept VARCHAR,
                target_concept VARCHAR,
                bridge_kind VARCHAR,
                applicability_condition VARCHAR,
                actionable_guidance VARCHAR,
                evidence_grade VARCHAR,
                citations VARCHAR[]
            );
            INSERT INTO cross_domain_bridges VALUES (
                'coe:C09001',
                'Worked Example to Demo Script Bridge',
                'Connects worked examples to structured storylines.',
                'coe:C05105',
                'coe:C06002',
                'operationalizes',
                'technical enablement',
                'State the conclusion first, then walk the worked steps.',
                'convergent_empirical',
                ['Sweller 2011']
            );
            """
        )
        bands_json = json.dumps(
            [
                {
                    "zone": "optimal",
                    "lower_bound": 130.0,
                    "upper_bound": 160.0,
                    "coaching_feedback": "Maintain conversational narration pace.",
                }
            ]
        )
        con.execute(
            """
            CREATE TABLE empirical_norms (
                id VARCHAR,
                pref_label VARCHAR,
                definition VARCHAR,
                measured_subject VARCHAR,
                quantity_kind_uri VARCHAR,
                unit_uri VARCHAR,
                aggregation_window VARCHAR,
                applicable_delivery_genres VARCHAR[],
                applicable_language_baseline VARCHAR,
                bands VARCHAR,
                evidence_grade VARCHAR,
                wiki_slug VARCHAR
            );
            INSERT INTO empirical_norms VALUES (
                'coe:C09102',
                'Narration Speaking Rate Norm',
                'Target speaking rate.',
                'coe:C05105',
                'qudt:Rate',
                'unit:WPM',
                'module',
                ['video_narration'],
                'l1_fluent',
                ?,
                'convergent_empirical',
                NULL
            );
            """,
            [bands_json],
        )
        con.execute(
            """
            CREATE TABLE search_corpus (
                id VARCHAR,
                collection_name VARCHAR,
                pref_label VARCHAR,
                definition VARCHAR,
                alt_terms VARCHAR
            );
            INSERT INTO search_corpus VALUES
                ('coe:C05105', 'cognitive_principles', 'Worked Example Effect', 'Novices learn faster from worked examples.', 'worked example'),
                ('coe:C01008', 'offerings', 'Adobe Target', 'Personalisation and experimentation offering.', 'Target');
            PRAGMA create_fts_index('search_corpus', 'id', 'pref_label', 'definition', 'alt_terms', overwrite=1);
            """
        )
    finally:
        con.close()
    return db_path


def test_central_projection_builds_v5_entities_edges_craft_and_fts(tmp_path: Path) -> None:
    db_path = _build_fixture_duckdb(tmp_path / "central.duckdb")
    result = load_ontology(db_path)

    by_id = {entity["canonical_id"]: entity for entity in result.entity_definitions}
    worked = by_id["coe:C05105"]
    assert worked["label"] == "Worked Example Effect"
    assert worked["wiki_slug"] == "worked-example-effect"
    assert "worked example" in worked["aliases"]
    assert "Replace full worked examples" in worked["craft_summary"]
    assert "optimal 130–160" in worked["craft_summary"]

    relation_types = {record.relation_type for record in result.relation_records}
    assert "offered_by" in relation_types
    assert "governed_principle_or_method" in relation_types
    assert "operationalizes" in relation_types

    # LinkableProperNoun firewall: cognitive_principles (coe:C05105) is absent from match_lexicon.
    lexicon_curies = {rule.curie for rule in result.matcher_records}
    assert lexicon_curies == {"coe:C00001", "coe:C01008"}
    assert "coe:C05105" not in lexicon_curies

    context = format_ontology_context(result.entity_definitions, ["coe:C05105"])
    assert "Worked Example Effect" in context
    assert "coe:C05105" in context

    bm25_hits = search_central_corpus(db_path, "worked examples", limit=5)
    assert bm25_hits
    assert bm25_hits[0]["id"] == "coe:C05105"
    assert bm25_hits[0]["bm25_score"] != 0.0


def test_stage1_anchor_and_negative_phrase_filtering(tmp_path: Path) -> None:
    db_path = _build_fixture_duckdb(tmp_path / "central.duckdb")
    result = load_ontology(db_path)
    automaton = build_automaton(result.matcher_records)

    kept = detect_mentions("Adobe runs A/B tests in Target.", automaton)
    assert {m.entity_id for m in kept} == {"coe:C00001", "coe:C01008"}

    vetoed = detect_mentions("They built an adobe brick wall near the Target store.", automaton)
    assert vetoed == []


@pytest.mark.skipif(
    not (CENTRAL_CONFIGS_ROOT / "dist" / "build" / "central.duckdb").is_file(),
    reason="Central_Configs v5.0.0 authority store not present locally",
)
def test_live_central_configs_v5_authority_projects_all_collections() -> None:
    result = load_ontology(CENTRAL_CONFIGS_ROOT)

    assert len(result.entity_definitions) >= 564
    assert len(result.matcher_records) >= 330
    assert result.validation_issues == []

    ids = {entity["canonical_id"] for entity in result.entity_definitions}
    assert "coe:C00001" in ids  # Adobe
    assert "coe:C05301" in ids  # PedagogicalEffectRule
    assert "coe:C09001" in ids  # CrossDomainBridge
    assert "coe:C09501" in ids  # Rubric
    assert "coe:C09601" in ids  # DiagnosticDetector
    assert "coe:C08504" in ids  # VoiceRegister

    assert len(result.pedagogical_effect_rules) >= 8
    assert len(result.cross_domain_bridges) >= 18
    assert len(result.empirical_norms) >= 12
    assert len(result.rubrics) >= 3
    assert len(result.diagnostic_detectors) >= 8
    assert len(result.voice_registers) >= 2
    assert len(result.learner_archetypes) >= 10

    # Craft concepts never enter match_lexicon.
    lexicon_collections = {rule.collection_name for rule in result.matcher_records}
    assert "cognitive_principles" not in lexicon_collections
    assert "rhetorical_moves" not in lexicon_collections
    assert "rubrics" not in lexicon_collections

    hits = search_central_corpus(CENTRAL_CONFIGS_ROOT, "cognitive load multimedia", limit=5)
    assert hits
    assert all(hit["id"].startswith("coe:C") for hit in hits)
