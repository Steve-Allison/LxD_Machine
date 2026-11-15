"""Read-only DuckDB 2.0 consumer projection for ``Central_Configs`` v5.0.0."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import duckdb

from lxd.domain.ids import blake3_hex
from lxd.ontology.graph import OntologyNodeRecord, RelationRecord, build_graph
from lxd.ontology.loader.types import (
    CrossDomainBridgeRecord,
    DiagnosticDetectorRecord,
    EditorialRuleRecord,
    EmpiricalNormRecord,
    LearnerArchetypeRecord,
    NormBandRecord,
    OntologyCoverageReport,
    OntologyLoadResult,
    OntologyMetadataRecord,
    OntologySource,
    OntologyTermRecord,
    OntologyValidationIssue,
    PedagogicalEffectRuleRecord,
    RubricCriterionRecord,
    RubricRecord,
    VoiceRegisterRecord,
)
from lxd.ontology.matcher import LexiconRule, matcher_termset_hash, offset_preserving_lower

_META_TABLES: frozenset[str] = frozenset(
    {"all_entities", "all_terms", "match_lexicon", "search_corpus"}
)
_CRAFT_CHAR_LIMIT = 800
_PROMPT_ENTITY_LIMIT = 5
_PROMPT_CHAR_LIMIT = 1400

_COLLECTION_DOMAIN_FALLBACK: dict[str, str] = {
    "organisations": "portfolio",
    "business_units": "portfolio",
    "offerings": "portfolio",
    "editions": "portfolio",
    "surfaces": "portfolio",
    "add_ons": "portfolio",
    "product_capabilities": "portfolio",
    "ai_models": "portfolio",
    "ai_agents": "portfolio",
    "technical_standards": "portfolio",
    "customer_problems": "evangelism",
    "use_cases": "evangelism",
    "competitive_positions": "evangelism",
    "objections": "evangelism",
    "claim_guardrails": "evangelism",
    "sales_methodologies": "evangelism",
    "buyer_personas": "personas",
    "practitioner_roles": "personas",
    "learner_archetypes": "personas",
    "cognitive_levels": "pedagogy",
    "evaluation_levels": "pedagogy",
    "transfer_tiers": "pedagogy",
    "cognitive_principles": "pedagogy",
    "instructional_methods": "pedagogy",
    "assessment_instruments": "pedagogy",
    "pedagogical_effect_rules": "pedagogy",
    "story_frameworks": "rhetoric",
    "framework_stages": "rhetoric",
    "argument_schemes": "rhetoric",
    "discourse_relations": "rhetoric",
    "rhetorical_moves": "rhetoric",
    "visual_principles": "visual_design",
    "visual_encodings": "visual_design",
    "slide_archetypes": "visual_design",
    "diagram_patterns": "visual_design",
    "delivery_channels": "delivery",
    "vocal_signals": "delivery",
    "kinematic_signals": "delivery",
    "prosody_cues": "delivery",
    "cross_domain_bridges": "pedagogy",
    "empirical_norms": "delivery",
    "rubrics": "pedagogy",
    "diagnostic_detectors": "editorial_voice",
    "voice_registers": "editorial_voice",
    "editorial_rules": "editorial_voice",
}

_SCALAR_REL_SLOTS: tuple[tuple[str, str], ...] = (
    ("organisations", "parent_organisation"),
    ("business_units", "parent_organisation"),
    ("offerings", "offered_by"),
    ("offerings", "owning_unit"),
    ("offerings", "replaced_by"),
    ("editions", "edition_of"),
    ("surfaces", "surface_of"),
    ("ai_models", "offered_by"),
    ("ai_agents", "offered_by"),
    ("technical_standards", "steward_organisation"),
    ("competitive_positions", "subject_offering"),
    ("competitive_positions", "competitor_offering"),
    ("competitive_positions", "in_use_case"),
    ("pedagogical_effect_rules", "governed_principle_or_method"),
    ("framework_stages", "stage_of_framework"),
    ("cross_domain_bridges", "source_concept"),
    ("cross_domain_bridges", "target_concept"),
    ("empirical_norms", "measured_subject"),
)

_ARRAY_REL_SLOTS: tuple[tuple[str, str], ...] = (
    ("offerings", "part_of_suite"),
    ("offerings", "built_on_platform"),
    ("offerings", "integrates_with"),
    ("offerings", "implements_standard"),
    ("offerings", "powered_by_models"),
    ("offerings", "embeds_agents"),
    ("editions", "included_capabilities"),
    ("add_ons", "addon_for"),
    ("add_ons", "included_capabilities"),
    ("product_capabilities", "capability_of"),
    ("product_capabilities", "powered_by_models"),
    ("product_capabilities", "implements_standard"),
    ("ai_models", "implements_standard"),
    ("ai_agents", "host_offerings"),
    ("ai_agents", "powered_by_models"),
    ("customer_problems", "affected_personas"),
    ("use_cases", "affected_personas"),
    ("use_cases", "addresses_problems"),
    ("use_cases", "enabled_by_offerings"),
    ("use_cases", "required_capabilities"),
    ("competitive_positions", "relevant_buyer_personas"),
    ("objections", "affected_personas"),
    ("objections", "related_offerings"),
    ("claim_guardrails", "related_offerings"),
    ("practitioner_roles", "primary_offerings_used"),
    ("learner_archetypes", "primary_offerings_used"),
    ("sales_methodologies", "methodology_stages"),
    ("instructional_methods", "targets_cognitive_levels"),
    ("instructional_methods", "targets_transfer_tiers"),
    ("assessment_instruments", "assesses_cognitive_levels"),
    ("assessment_instruments", "targets_transfer_tiers"),
    ("diagnostic_detectors", "linked_concepts"),
    ("voice_registers", "target_personas"),
)


def resolve_central_duckdb_path(root_or_db: Path) -> Path:
    """Resolve either a ``Central_Configs`` root directory or a ``.duckdb`` file path."""
    if root_or_db.suffix == ".duckdb" or root_or_db.is_file():
        return root_or_db
    return root_or_db / "dist" / "build" / "central.duckdb"


def load_central_ontology(root_or_db: Path) -> OntologyLoadResult:
    """Load ``Central_Configs`` v5.0.0 directly from ``central.duckdb`` (``read_only=True``)."""
    db_path = resolve_central_duckdb_path(root_or_db)
    if not db_path.is_file():
        raise FileNotFoundError(f"Missing Central_Configs DuckDB store: {db_path}")

    source_rel = "dist/build/central.duckdb"
    issues: list[OntologyValidationIssue] = []

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        tables = _existing_tables(con)
        table_counts = _load_table_counts(con, tables)
        terms = _load_terms(con, tables)
        terms_by_entity: dict[str, list[OntologyTermRecord]] = defaultdict(list)
        for term in terms:
            terms_by_entity[term.resource_id].append(term)

        entities = _load_entities(
            con,
            tables=tables,
            terms_by_entity=terms_by_entity,
            source_rel=source_rel,
            issues=issues,
        )
        entity_ids = {str(entity["canonical_id"]) for entity in entities}

        matcher_records = _load_match_lexicon(con, tables=tables, entity_ids=entity_ids, issues=issues)
        pedagogical_rules = _load_pedagogical_effect_rules(con, tables)
        bridges = _load_cross_domain_bridges(con, tables)
        norms = _load_empirical_norms(con, tables)
        rubrics = _load_rubrics(con, tables)
        detectors = _load_diagnostic_detectors(con, tables)
        voices = _load_voice_registers(con, tables)
        archetypes = _load_learner_archetypes(con, tables)
        editorial_rules = _load_editorial_rules(con, tables)

        _attach_craft_summaries(
            entities,
            pedagogical_rules=pedagogical_rules,
            bridges=bridges,
            norms=norms,
            detectors=detectors,
            editorial_rules=editorial_rules,
        )

        relations = _load_relations(
            con,
            tables=tables,
            entities=entities,
            entity_ids=entity_ids,
            bridges=bridges,
            source_rel=source_rel,
            issues=issues,
        )
    finally:
        con.close()

    termset_hash = matcher_termset_hash(matcher_records)
    snap_hash = _compute_snapshot_hash(entities, terms, relations, termset_hash)
    source = OntologySource(
        file_path=db_path,
        file_rel_path=source_rel,
        blake3_hash=snap_hash,
        data={"table_counts": table_counts},
    )
    metadata_records = _build_metadata_records(entities, source_rel=source_rel)
    nodes = _build_node_records(entities, source_rel=source_rel)
    graph = build_graph(nodes, relations)

    return OntologyLoadResult(
        db_path=db_path,
        sources=[source],
        entity_definitions=entities,
        terms=terms,
        matcher_records=matcher_records,
        matcher_termset_hash=termset_hash,
        snapshot_hash=snap_hash,
        relation_records=relations,
        metadata_records=metadata_records,
        coverage_report=OntologyCoverageReport(table_counts=table_counts),
        validation_issues=issues,
        graph=graph,
        pedagogical_effect_rules=pedagogical_rules,
        cross_domain_bridges=bridges,
        empirical_norms=norms,
        rubrics=rubrics,
        diagnostic_detectors=detectors,
        voice_registers=voices,
        learner_archetypes=archetypes,
        editorial_rules=editorial_rules,
    )


def search_central_corpus(
    root_or_db: Path,
    query: str,
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Run a DuckDB 2.x BM25 full-text query over ``search_corpus``.

    Uses a ``WITH scored AS (...)`` CTE before filtering ``WHERE bm25_score IS NOT NULL``
    as required by DuckDB 2.x FTS binding semantics.
    """
    cleaned = query.strip()
    if not cleaned or limit <= 0:
        return []
    db_path = resolve_central_duckdb_path(root_or_db)
    if not db_path.is_file():
        return []
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        tables = _existing_tables(con)
        if "search_corpus" not in tables:
            return []
        rows = con.execute(
            """
            WITH scored AS (
                SELECT
                    id,
                    collection_name,
                    pref_label,
                    definition,
                    alt_terms,
                    fts_main_search_corpus.match_bm25(id, $q) AS bm25_score
                FROM search_corpus
            )
            SELECT id, collection_name, pref_label, definition, alt_terms, bm25_score
            FROM scored
            WHERE bm25_score IS NOT NULL
            ORDER BY bm25_score DESC, id ASC
            LIMIT $limit
            """,
            {"q": cleaned, "limit": limit},
        ).fetchall()
    except duckdb.Error:
        return []
    finally:
        con.close()

    return [
        {
            "id": str(row[0]),
            "collection_name": str(row[1] or ""),
            "pref_label": str(row[2] or ""),
            "definition": str(row[3] or ""),
            "alt_terms": str(row[4] or ""),
            "bm25_score": float(row[5]),
        }
        for row in rows
    ]


def format_ontology_context(
    entity_definitions: list[dict[str, Any]],
    entity_ids: list[str],
) -> str:
    """Format definitions and v5 craft prescriptions for matched entities."""
    by_id = {
        entity["canonical_id"]: entity
        for entity in entity_definitions
        if isinstance(entity.get("canonical_id"), str)
    }
    lines: list[str] = []
    for entity_id in entity_ids:
        if len(lines) >= _PROMPT_ENTITY_LIMIT:
            break
        entity = by_id.get(entity_id)
        if entity is None:
            continue
        parts: list[str] = []
        description = entity.get("description")
        craft = entity.get("craft_summary")
        if isinstance(description, str) and description.strip():
            parts.append(description.strip())
        if isinstance(craft, str) and craft.strip():
            parts.append(craft.strip())
        if not parts:
            continue
        label = entity.get("label") if isinstance(entity.get("label"), str) else entity_id
        lines.append(f"- **{label}** (`{entity_id}`): {' '.join(parts)}")
    if not lines:
        return ""
    block = "## Ontology context\n\n" + "\n".join(lines) + "\n"
    if len(block) <= _PROMPT_CHAR_LIMIT:
        return block
    return block[: _PROMPT_CHAR_LIMIT - 1].rstrip() + "…\n"


def _existing_tables(con: duckdb.DuckDBPyConnection) -> set[str]:
    return {str(row[0]) for row in con.execute("SHOW TABLES").fetchall()}


def _table_columns(con: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    return {str(row[0]) for row in con.execute(f"DESCRIBE {table}").fetchall()}


def _load_table_counts(con: duckdb.DuckDBPyConnection, tables: set[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for table in sorted(tables):
        row = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        counts[table] = int(row[0]) if row is not None else 0
    return counts


def _load_terms(
    con: duckdb.DuckDBPyConnection,
    tables: set[str],
) -> tuple[OntologyTermRecord, ...]:
    if "all_terms" not in tables:
        return ()
    cols = _table_columns(con, "all_terms")
    fk_col = "resource_id" if "resource_id" in cols else "entity_id"
    select_exprs = [
        fk_col,
        "pref_label" if "pref_label" in cols else "''",
        "collection_name" if "collection_name" in cols else "''",
        "written_form",
        "term_kind" if "term_kind" in cols else "'synonym'",
        "term_status" if "term_status" in cols else "NULL",
        "language" if "language" in cols else "NULL",
        "ipa_phonetic" if "ipa_phonetic" in cols else "NULL",
        "spoken_form" if "spoken_form" in cols else "NULL",
        "usage_note" if "usage_note" in cols else "NULL",
    ]
    rows = con.execute(
        f"SELECT {', '.join(select_exprs)} FROM all_terms ORDER BY {fk_col}, written_form"
    ).fetchall()
    return tuple(
        OntologyTermRecord(
            resource_id=str(row[0]),
            pref_label=str(row[1] or ""),
            collection_name=str(row[2] or ""),
            written_form=str(row[3]),
            term_kind=str(row[4] or ""),
            term_status=str(row[5]) if row[5] is not None else None,
            language=str(row[6]) if row[6] is not None else None,
            ipa_phonetic=str(row[7]) if row[7] is not None else None,
            spoken_form=str(row[8]) if row[8] is not None else None,
            usage_note=str(row[9]) if row[9] is not None else None,
        )
        for row in rows
        if row[0] and row[3]
    )


def _load_entities(
    con: duckdb.DuckDBPyConnection,
    *,
    tables: set[str],
    terms_by_entity: dict[str, list[OntologyTermRecord]],
    source_rel: str,
    issues: list[OntologyValidationIssue],
) -> list[dict[str, Any]]:
    if "all_entities" not in tables:
        raise ValueError("Central_Configs DuckDB store is missing required 'all_entities' table.")
    cols = _table_columns(con, "all_entities")
    select_exprs = [
        "id",
        "collection_name",
        "pref_label",
        "definition" if "definition" in cols else "''",
        "status" if "status" in cols else "'active'",
        "replaced_by" if "replaced_by" in cols else "NULL",
        "volatility_tier" if "volatility_tier" in cols else "NULL",
        "CAST(last_verified_date AS VARCHAR)" if "last_verified_date" in cols else "NULL",
        "source_uri" if "source_uri" in cols else "NULL",
        "wiki_slug" if "wiki_slug" in cols else "NULL",
        "facets_json" if "facets_json" in cols else "'[]'",
    ]
    rows = con.execute(
        f"SELECT {', '.join(select_exprs)} FROM all_entities ORDER BY id"
    ).fetchall()

    entities: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        curie = str(row[0] or "").strip()
        if not curie:
            continue
        if curie in seen:
            issues.append(
                OntologyValidationIssue(
                    issue_kind="duplicate_resource",
                    source_file_rel_path=source_rel,
                    path=curie,
                    message=f"Duplicate Central_Configs entity identifier '{curie}'.",
                )
            )
            continue
        seen.add(curie)
        collection_name = str(row[1] or "").strip()
        pref_label = str(row[2] or "").strip() or curie
        definition = str(row[3] or "").strip()
        status = str(row[4]) if row[4] is not None else "active"
        replaced_by = str(row[5]) if row[5] is not None else None
        volatility_tier = str(row[6]) if row[6] is not None else None
        last_verified_date = str(row[7]) if row[7] is not None else None
        source_uri = str(row[8]) if row[8] is not None else None
        wiki_slug = str(row[9]).strip() if row[9] is not None and str(row[9]).strip() else None
        facets = _parse_str_list_json(row[10])
        domain = facets[0] if facets else _COLLECTION_DOMAIN_FALLBACK.get(collection_name, collection_name)
        written_forms = [
            term.written_form.strip()
            for term in terms_by_entity.get(curie, [])
            if term.written_form.strip()
        ]
        aliases = _dedupe_strings([pref_label, *written_forms])

        entities.append(
            {
                "canonical_id": curie,
                "label": pref_label,
                "description": definition,
                "aliases": aliases,
                "indicators": [],
                "entity_kind": collection_name,
                "entity_type": collection_name,
                "domain": domain,
                "facets": facets,
                "family": "",
                "source_file_rel_path": source_rel,
                "source_meta_id": collection_name,
                "status": status,
                "replaced_by": replaced_by,
                "volatility_tier": volatility_tier,
                "last_verified_date": last_verified_date,
                "source_uri": source_uri,
                "wiki_slug": wiki_slug,
                "craft_summary": "",
            }
        )
    return entities


def _load_match_lexicon(
    con: duckdb.DuckDBPyConnection,
    *,
    tables: set[str],
    entity_ids: set[str],
    issues: list[OntologyValidationIssue],
) -> list[LexiconRule]:
    if "match_lexicon" not in tables:
        return []
    cols = _table_columns(con, "match_lexicon")
    anchors_expr = (
        "context_anchors_json"
        if "context_anchors_json" in cols
        else ("CAST(context_anchors AS VARCHAR)" if "context_anchors" in cols else "'[]'")
    )
    neg_expr = (
        "negative_phrases_json"
        if "negative_phrases_json" in cols
        else ("CAST(negative_phrases AS VARCHAR)" if "negative_phrases" in cols else "'[]'")
    )
    select_exprs = [
        "curie",
        "pref_label" if "pref_label" in cols else "''",
        "collection_name" if "collection_name" in cols else "''",
        "surface",
        "match_policy" if "match_policy" in cols else "'unambiguous'",
        "case_sensitive" if "case_sensitive" in cols else "FALSE",
        "anchor_window_tokens" if "anchor_window_tokens" in cols else "25",
        "requires_llm_disambiguation" if "requires_llm_disambiguation" in cols else "FALSE",
        anchors_expr,
        neg_expr,
    ]
    rows = con.execute(
        f"SELECT {', '.join(select_exprs)} FROM match_lexicon ORDER BY curie, surface"
    ).fetchall()

    rules: list[LexiconRule] = []
    for row in rows:
        curie = str(row[0] or "").strip()
        surface = str(row[3] or "").strip()
        if not curie or not surface:
            continue
        if entity_ids and curie not in entity_ids:
            issues.append(
                OntologyValidationIssue(
                    issue_kind="dangling_lexicon_curie",
                    source_file_rel_path="dist/build/central.duckdb",
                    path=f"match_lexicon.{curie}",
                    message=f"match_lexicon CURIE '{curie}' is absent from all_entities.",
                )
            )
            continue
        rules.append(
            LexiconRule(
                surface_lower=offset_preserving_lower(surface),
                curie=curie,
                surface=surface,
                pref_label=str(row[1] or ""),
                collection_name=str(row[2] or ""),
                match_policy=str(row[4] or "unambiguous"),
                case_sensitive=bool(row[5]),
                anchor_window_tokens=int(row[6]) if row[6] is not None else 25,
                requires_llm_disambiguation=bool(row[7]),
                context_anchors=tuple(_parse_str_list_any(row[8])),
                negative_phrases=tuple(_parse_str_list_any(row[9])),
            )
        )
    return sorted(set(rules))


def _load_pedagogical_effect_rules(
    con: duckdb.DuckDBPyConnection,
    tables: set[str],
) -> tuple[PedagogicalEffectRuleRecord, ...]:
    if "pedagogical_effect_rules" not in tables:
        return ()
    rows = con.execute(
        """
        SELECT
            id,
            pref_label,
            definition,
            governed_principle_or_method,
            conditioned_expertise_tier,
            conditioned_language_baseline,
            modality_or_task_condition,
            effect_polarity,
            adapted_prescription,
            evidence_grade,
            citations
        FROM pedagogical_effect_rules
        ORDER BY id
        """
    ).fetchall()
    return tuple(
        PedagogicalEffectRuleRecord(
            id=str(row[0]),
            pref_label=str(row[1] or ""),
            definition=str(row[2] or ""),
            governed_principle_or_method=str(row[3] or ""),
            conditioned_expertise_tier=str(row[4]) if row[4] is not None else None,
            conditioned_language_baseline=str(row[5]) if row[5] is not None else None,
            modality_or_task_condition=str(row[6] or ""),
            effect_polarity=str(row[7] or ""),
            adapted_prescription=str(row[8] or ""),
            evidence_grade=str(row[9]) if row[9] is not None else None,
            citations=tuple(_parse_str_list_any(row[10])),
        )
        for row in rows
        if row[0]
    )


def _load_cross_domain_bridges(
    con: duckdb.DuckDBPyConnection,
    tables: set[str],
) -> tuple[CrossDomainBridgeRecord, ...]:
    if "cross_domain_bridges" not in tables:
        return ()
    rows = con.execute(
        """
        SELECT
            id,
            pref_label,
            definition,
            source_concept,
            target_concept,
            bridge_kind,
            applicability_condition,
            actionable_guidance,
            evidence_grade,
            citations
        FROM cross_domain_bridges
        ORDER BY id
        """
    ).fetchall()
    return tuple(
        CrossDomainBridgeRecord(
            id=str(row[0]),
            pref_label=str(row[1] or ""),
            definition=str(row[2] or ""),
            source_concept=str(row[3] or ""),
            target_concept=str(row[4] or ""),
            bridge_kind=str(row[5] or ""),
            applicability_condition=str(row[6] or ""),
            actionable_guidance=str(row[7] or ""),
            evidence_grade=str(row[8]) if row[8] is not None else None,
            citations=tuple(_parse_str_list_any(row[9])),
        )
        for row in rows
        if row[0]
    )


def _load_empirical_norms(
    con: duckdb.DuckDBPyConnection,
    tables: set[str],
) -> tuple[EmpiricalNormRecord, ...]:
    if "empirical_norms" not in tables:
        return ()
    rows = con.execute(
        """
        SELECT
            id,
            pref_label,
            definition,
            measured_subject,
            quantity_kind_uri,
            unit_uri,
            aggregation_window,
            applicable_delivery_genres,
            applicable_language_baseline,
            bands,
            evidence_grade,
            wiki_slug
        FROM empirical_norms
        ORDER BY id
        """
    ).fetchall()
    norms: list[EmpiricalNormRecord] = []
    for row in rows:
        if not row[0]:
            continue
        bands_raw = _parse_json_obj_list(row[9])
        bands = tuple(
            NormBandRecord(
                zone=str(item.get("zone", "")),
                lower_bound=float(item["lower_bound"])
                if isinstance(item.get("lower_bound"), int | float)
                else None,
                upper_bound=float(item["upper_bound"])
                if isinstance(item.get("upper_bound"), int | float)
                else None,
                coaching_feedback=str(item.get("coaching_feedback", "")),
            )
            for item in bands_raw
        )
        norms.append(
            EmpiricalNormRecord(
                id=str(row[0]),
                pref_label=str(row[1] or ""),
                definition=str(row[2] or ""),
                measured_subject=str(row[3] or ""),
                quantity_kind_uri=str(row[4] or ""),
                unit_uri=str(row[5] or ""),
                aggregation_window=str(row[6]) if row[6] is not None else None,
                applicable_delivery_genres=tuple(_parse_str_list_any(row[7])),
                applicable_language_baseline=str(row[8]) if row[8] is not None else None,
                bands=bands,
                evidence_grade=str(row[10]) if row[10] is not None else None,
                wiki_slug=str(row[11]) if row[11] is not None else None,
            )
        )
    return tuple(norms)


def _load_rubrics(
    con: duckdb.DuckDBPyConnection,
    tables: set[str],
) -> tuple[RubricRecord, ...]:
    if "rubrics" not in tables:
        return ()
    rows = con.execute(
        """
        SELECT id, pref_label, definition, evaluated_artifact_kind, criteria, wiki_slug
        FROM rubrics
        ORDER BY id
        """
    ).fetchall()
    rubrics: list[RubricRecord] = []
    for row in rows:
        if not row[0]:
            continue
        crit_raw = _parse_json_obj_list(row[4])
        criteria = tuple(
            RubricCriterionRecord(
                pref_label=str(item.get("pref_label", "")),
                definition=str(item.get("definition", "")),
                criterion_weight=float(item["criterion_weight"])
                if isinstance(item.get("criterion_weight"), int | float)
                else None,
                linked_concepts=tuple(_parse_str_list_any(item.get("linked_concepts"))),
                levels=tuple(
                    lvl for lvl in (item.get("levels") or []) if isinstance(lvl, dict)
                ),
            )
            for item in crit_raw
        )
        rubrics.append(
            RubricRecord(
                id=str(row[0]),
                pref_label=str(row[1] or ""),
                definition=str(row[2] or ""),
                evaluated_artifact_kind=str(row[3] or ""),
                criteria=criteria,
                wiki_slug=str(row[5]) if row[5] is not None else None,
            )
        )
    return tuple(rubrics)


def _load_diagnostic_detectors(
    con: duckdb.DuckDBPyConnection,
    tables: set[str],
) -> tuple[DiagnosticDetectorRecord, ...]:
    if "diagnostic_detectors" not in tables:
        return ()
    rows = con.execute(
        """
        SELECT
            id,
            pref_label,
            definition,
            linked_concepts,
            detection_signal_description,
            remediation_instruction,
            wiki_slug
        FROM diagnostic_detectors
        ORDER BY id
        """
    ).fetchall()
    return tuple(
        DiagnosticDetectorRecord(
            id=str(row[0]),
            pref_label=str(row[1] or ""),
            definition=str(row[2] or ""),
            linked_concepts=tuple(_parse_str_list_any(row[3])),
            detection_signal_description=str(row[4] or ""),
            remediation_instruction=str(row[5] or ""),
            wiki_slug=str(row[6]) if row[6] is not None else None,
        )
        for row in rows
        if row[0]
    )


def _load_voice_registers(
    con: duckdb.DuckDBPyConnection,
    tables: set[str],
) -> tuple[VoiceRegisterRecord, ...]:
    if "voice_registers" not in tables:
        return ()
    rows = con.execute(
        """
        SELECT
            id,
            pref_label,
            definition,
            default_locale,
            target_personas,
            register_traits,
            preferred_syntactic_patterns,
            prohibited_anti_patterns,
            wiki_slug
        FROM voice_registers
        ORDER BY id
        """
    ).fetchall()
    return tuple(
        VoiceRegisterRecord(
            id=str(row[0]),
            pref_label=str(row[1] or ""),
            definition=str(row[2] or ""),
            default_locale=str(row[3] or "en-US"),
            target_personas=tuple(_parse_str_list_any(row[4])),
            register_traits=tuple(_parse_str_list_any(row[5])),
            preferred_syntactic_patterns=tuple(_parse_str_list_any(row[6])),
            prohibited_anti_patterns=tuple(_parse_str_list_any(row[7])),
            wiki_slug=str(row[8]) if row[8] is not None else None,
        )
        for row in rows
        if row[0]
    )


def _load_learner_archetypes(
    con: duckdb.DuckDBPyConnection,
    tables: set[str],
) -> tuple[LearnerArchetypeRecord, ...]:
    if "learner_archetypes" not in tables:
        return ()
    rows = con.execute(
        """
        SELECT
            id,
            pref_label,
            definition,
            primary_offerings_used,
            primary_goals,
            core_frustrations,
            expertise_tier,
            language_baseline,
            typical_time_budget_minutes,
            wiki_slug
        FROM learner_archetypes
        ORDER BY id
        """
    ).fetchall()
    return tuple(
        LearnerArchetypeRecord(
            id=str(row[0]),
            pref_label=str(row[1] or ""),
            definition=str(row[2] or ""),
            primary_offerings_used=tuple(_parse_str_list_any(row[3])),
            primary_goals=tuple(_parse_str_list_any(row[4])),
            core_frustrations=tuple(_parse_str_list_any(row[5])),
            expertise_tier=str(row[6]) if row[6] is not None else None,
            language_baseline=str(row[7]) if row[7] is not None else None,
            typical_time_budget_minutes=int(row[8]) if row[8] is not None else None,
            wiki_slug=str(row[9]) if row[9] is not None else None,
        )
        for row in rows
        if row[0]
    )


def _load_editorial_rules(
    con: duckdb.DuckDBPyConnection,
    tables: set[str],
) -> tuple[EditorialRuleRecord, ...]:
    if "editorial_rules" not in tables:
        return ()
    rows = con.execute(
        """
        SELECT
            id,
            pref_label,
            definition,
            editorial_directive,
            illustrative_before_after,
            evidence_grade,
            wiki_slug
        FROM editorial_rules
        ORDER BY id
        """
    ).fetchall()
    return tuple(
        EditorialRuleRecord(
            id=str(row[0]),
            pref_label=str(row[1] or ""),
            definition=str(row[2] or ""),
            editorial_directive=str(row[3] or ""),
            illustrative_before_after=str(row[4] or ""),
            evidence_grade=str(row[5]) if row[5] is not None else None,
            wiki_slug=str(row[6]) if row[6] is not None else None,
        )
        for row in rows
        if row[0]
    )


def _attach_craft_summaries(
    entities: list[dict[str, Any]],
    *,
    pedagogical_rules: tuple[PedagogicalEffectRuleRecord, ...],
    bridges: tuple[CrossDomainBridgeRecord, ...],
    norms: tuple[EmpiricalNormRecord, ...],
    detectors: tuple[DiagnosticDetectorRecord, ...],
    editorial_rules: tuple[EditorialRuleRecord, ...],
) -> None:
    notes_by_id: dict[str, list[str]] = defaultdict(list)

    for rule in pedagogical_rules:
        summary = f"[{rule.effect_polarity}] {rule.adapted_prescription}".strip()
        if summary:
            notes_by_id[rule.id].append(summary)
            if rule.governed_principle_or_method:
                notes_by_id[rule.governed_principle_or_method].append(
                    f"{rule.pref_label}: {rule.adapted_prescription}"
                )

    for bridge in bridges:
        if bridge.actionable_guidance:
            notes_by_id[bridge.id].append(bridge.actionable_guidance)
            if bridge.source_concept:
                notes_by_id[bridge.source_concept].append(
                    f"{bridge.pref_label}: {bridge.actionable_guidance}"
                )
            if bridge.target_concept:
                notes_by_id[bridge.target_concept].append(
                    f"{bridge.pref_label}: {bridge.actionable_guidance}"
                )

    for norm in norms:
        band_desc = _format_norm_bands(norm.bands)
        if band_desc:
            notes_by_id[norm.id].append(band_desc)
            if norm.measured_subject:
                notes_by_id[norm.measured_subject].append(f"{norm.pref_label}: {band_desc}")

    for detector in detectors:
        if detector.remediation_instruction:
            notes_by_id[detector.id].append(detector.remediation_instruction)
            for concept_id in detector.linked_concepts:
                notes_by_id[concept_id].append(
                    f"{detector.pref_label}: {detector.remediation_instruction}"
                )

    for ed_rule in editorial_rules:
        if ed_rule.editorial_directive:
            notes_by_id[ed_rule.id].append(ed_rule.editorial_directive)

    for entity in entities:
        curie = str(entity["canonical_id"])
        parts = notes_by_id.get(curie)
        if parts:
            entity["craft_summary"] = " ".join(parts[:4])[:_CRAFT_CHAR_LIMIT]


def _format_norm_bands(bands: tuple[NormBandRecord, ...]) -> str:
    for band in bands:
        if "optimal" in band.zone or "target" in band.zone:
            bounds = _format_bounds(band.lower_bound, band.upper_bound)
            return f"{band.zone} {bounds}".strip()
    if bands:
        first = bands[0]
        return f"{first.zone} {_format_bounds(first.lower_bound, first.upper_bound)}".strip()
    return ""


def _format_bounds(lower: float | None, upper: float | None) -> str:
    if lower is not None and upper is not None:
        return f"{lower:g}–{upper:g}"
    if lower is not None:
        return f"≥{lower:g}"
    if upper is not None:
        return f"≤{upper:g}"
    return ""


def _load_relations(
    con: duckdb.DuckDBPyConnection,
    *,
    tables: set[str],
    entities: list[dict[str, Any]],
    entity_ids: set[str],
    bridges: tuple[CrossDomainBridgeRecord, ...],
    source_rel: str,
    issues: list[OntologyValidationIssue],
) -> list[RelationRecord]:
    relations: list[RelationRecord] = []
    seen_edges: set[tuple[str, str, str]] = set()

    def add_edge(source_id: str, rel_type: str, target_id: str, origin_path: str) -> None:
        if not source_id or not target_id or source_id == target_id:
            return
        if target_id not in entity_ids:
            if target_id.startswith("coe:C"):
                issues.append(
                    OntologyValidationIssue(
                        issue_kind="dangling_relation_target",
                        source_file_rel_path=source_rel,
                        path=origin_path,
                        message=f"Target CURIE '{target_id}' is absent from all_entities.",
                    )
                )
            return
        edge_sig = (source_id, rel_type, target_id)
        if edge_sig in seen_edges:
            return
        seen_edges.add(edge_sig)
        relations.append(
            RelationRecord(
                relation_type=rel_type,
                origin_kind="central",
                origin_path=origin_path,
                source_file_rel_path=source_rel,
                source_node_id=source_id,
                source_node_type="entity",
                source_entity_id=source_id,
                target_node_id=target_id,
                target_node_type="entity",
                target_entity_id=target_id,
                target_file_rel_path=None,
                metadata={},
            )
        )

    # Top-level replaced_by on all_entities
    for entity in entities:
        src = str(entity["canonical_id"])
        rep = entity.get("replaced_by")
        if isinstance(rep, str) and rep:
            add_edge(src, "replaced_by", rep, f"all_entities.{src}.replaced_by")

    # SKOS broader across any collection table carrying 'broader'
    for table in sorted(tables - _META_TABLES):
        cols = _table_columns(con, table)
        if "id" not in cols or "broader" not in cols:
            continue
        rows = con.execute(
            f"SELECT id, broader FROM {table} WHERE broader IS NOT NULL ORDER BY id"
        ).fetchall()
        for src_raw, targets_raw in rows:
            src = str(src_raw or "")
            for tgt in _parse_str_list_any(targets_raw):
                add_edge(src, "broader", tgt, f"{table}.{src}.broader")

    # Scalar foreign-key slots
    for table, slot in _SCALAR_REL_SLOTS:
        if table not in tables:
            continue
        cols = _table_columns(con, table)
        if "id" not in cols or slot not in cols:
            continue
        rows = con.execute(
            f"SELECT id, {slot} FROM {table} WHERE {slot} IS NOT NULL ORDER BY id"
        ).fetchall()
        for src_raw, tgt_raw in rows:
            add_edge(str(src_raw or ""), slot, str(tgt_raw or ""), f"{table}.{src_raw}.{slot}")

    # Array foreign-key slots
    for table, slot in _ARRAY_REL_SLOTS:
        if table not in tables:
            continue
        cols = _table_columns(con, table)
        if "id" not in cols or slot not in cols:
            continue
        rows = con.execute(
            f"SELECT id, {slot} FROM {table} WHERE {slot} IS NOT NULL ORDER BY id"
        ).fetchall()
        for src_raw, targets_raw in rows:
            src = str(src_raw or "")
            for tgt in _parse_str_list_any(targets_raw):
                add_edge(src, slot, tgt, f"{table}.{src}.{slot}")

    # Direct semantic shortcut across CrossDomainBridge (source_concept -> target_concept)
    for bridge in bridges:
        rel_name = bridge.bridge_kind or "bridges_to"
        add_edge(
            bridge.source_concept,
            rel_name,
            bridge.target_concept,
            f"cross_domain_bridges.{bridge.id}.{rel_name}",
        )

    # Direct competitive edge (subject_offering -> competitor_offering)
    if "competitive_positions" in tables:
        cols = _table_columns(con, "competitive_positions")
        if {"id", "subject_offering", "competitor_offering"}.issubset(cols):
            rows = con.execute(
                """
                SELECT id, subject_offering, competitor_offering
                FROM competitive_positions
                WHERE subject_offering IS NOT NULL AND competitor_offering IS NOT NULL
                ORDER BY id
                """
            ).fetchall()
            for row_id, subj, comp in rows:
                add_edge(
                    str(subj),
                    "competes_with",
                    str(comp),
                    f"competitive_positions.{row_id}.competes_with",
                )

    return relations


def _compute_snapshot_hash(
    entities: list[dict[str, Any]],
    terms: tuple[OntologyTermRecord, ...],
    relations: list[RelationRecord],
    termset_hash: str,
) -> str:
    entity_sig = [
        f"{e['canonical_id']}|{e['label']}|{e['entity_kind']}|{e.get('status')}|{e.get('wiki_slug')}"
        for e in entities
    ]
    term_sig = [f"{t.resource_id}|{t.written_form}|{t.term_kind}" for t in terms]
    rel_sig = [f"{r.source_node_id}|{r.relation_type}|{r.target_node_id}" for r in relations]
    return blake3_hex("\n".join(entity_sig), "\n".join(term_sig), "\n".join(rel_sig), termset_hash)


def _build_metadata_records(
    entities: list[dict[str, Any]],
    *,
    source_rel: str,
) -> list[OntologyMetadataRecord]:
    return [
        OntologyMetadataRecord(
            record_kind="entity",
            source_file_rel_path=source_rel,
            entity_id=str(entity["canonical_id"]),
            payload={
                "label": entity["label"],
                "entity_kind": entity["entity_kind"],
                "domain": entity["domain"],
                "wiki_slug": entity.get("wiki_slug"),
                "status": entity.get("status"),
            },
        )
        for entity in entities
    ]


def _build_node_records(
    entities: list[dict[str, Any]],
    *,
    source_rel: str,
) -> list[OntologyNodeRecord]:
    return [
        OntologyNodeRecord(
            node_id=str(entity["canonical_id"]),
            node_type="entity",
            source_file_rel_path=source_rel,
            entity_id=str(entity["canonical_id"]),
            label=str(entity["label"]),
            metadata={
                "entity_kind": entity["entity_kind"],
                "domain": entity["domain"],
                "wiki_slug": entity.get("wiki_slug"),
                "status": entity.get("status"),
            },
        )
        for entity in entities
    ]


def _parse_str_list_json(raw: Any) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, list | tuple):
        return [str(x).strip() for x in raw if str(x).strip()]
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if isinstance(x, str) and x.strip()]
    return []


def _parse_str_list_any(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return _parse_str_list_json(raw)
    if isinstance(raw, list | tuple):
        return [str(item).strip() for item in raw if item is not None and str(item).strip()]
    return []


def _parse_json_obj_list(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    items: list[Any]
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            return []
        items = decoded if isinstance(decoded, list) else []
    elif isinstance(raw, list | tuple):
        items = list(raw)
    else:
        return []
    out: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            out.append(item)
        elif isinstance(item, str):
            try:
                parsed = json.loads(item)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                out.append(parsed)
    return out


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        cleaned = value.strip()
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(cleaned)
    return unique
