"""Project a Central Configs distribution and library into runtime ontology records.

Domain modules name meaning. Library modules state how to recognise, measure,
and judge that meaning. This module is the consumer projection: it does not
edit vendored files, and it does not treat the legacy ``entity_types`` tree
as semantic authority.
"""

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from lxd.ontology.graph import RelationRecord, build_graph
from lxd.ontology.loader.entities import build_node_records, extract_metadata_records
from lxd.ontology.loader.sources import coverage_report_for_sources, load_sources
from lxd.ontology.loader.sources import snapshot_hash as compute_snapshot_hash
from lxd.ontology.loader.types import (
    AnchorConstraint,
    OntologyLoadResult,
    OntologySource,
    OntologyValidationIssue,
    RecognitionPattern,
)
from lxd.ontology.matcher import canonical_matcher_term_records, matcher_termset_hash
from lxd.ontology.normalization import normalize_match_text

_CENTRAL_GLOBS: tuple[str, ...] = (
    "domains/**/*.yaml",
    "manifests/*.yaml",
    "distribution.yaml",
)
_LIBRARY_GLOBS: tuple[str, ...] = ("data/**/*.yaml", "distribution.yaml")
_COLLECTION_KINDS: dict[str, str] = {
    "entity_collections": "entity_collection",
    "taxonomies": "taxonomy",
    "scoring_models": "scoring_model",
    "organisations": "organisation",
    "business_units": "business_unit",
    "product_families": "product_family",
    "applications": "application",
    "platforms": "platform",
    "editions": "edition",
    "surfaces": "surface",
    "add_ons": "add_on",
    "services": "service",
    "managed_services": "managed_service",
    "ai_models": "ai_model",
    "agents": "agent",
    "capabilities": "capability",
    "standards": "standard",
    "frameworks": "framework",
    "framework_stages": "framework_stage",
    "industry_categories": "industry_category",
    "practices": "practice",
}
_CHILD_KINDS: dict[str, str] = {
    "entities": "entity",
    "concepts": "concept",
    "criteria": "criterion",
    "dimensions": "scoring_dimension",
    "stages": "stage",
}
_PORTFOLIO_SLOTS: tuple[str, ...] = (
    "unit_of",
    "vendor",
    "families",
    "part_of",
    "built_on",
    "edition_of",
    "surface_of",
    "add_on_to",
    "capability_of",
    "available_in",
    "orchestrated_by",
    "powered_by",
    "conforms_to",
    "implements_category",
    "supports_practice",
    "serves_stage",
    "stage_of",
    "succeeded_by",
)
_LIBRARY_SLOTS: tuple[str, ...] = (
    "thresholds",
    "scoring_models",
    "principle_evidence",
    "prescriptions",
    "recognition_calibrations",
    "profiles",
    "layout_specs",
)
_SHORT_TERM_TYPES: frozenset[str] = frozenset(
    {"initialism", "acronym", "abbreviation", "product_code", "codename", "short_form"}
)
_REGEX_META: re.Pattern[str] = re.compile(r"[\\[\](){}|+?*]")
_CRAFT_CHAR_LIMIT = 600
_PROMPT_ENTITY_LIMIT = 5
_PROMPT_CHAR_LIMIT = 1200


def is_central_distribution(root: Path) -> bool:
    """Return whether ``root`` is a staged Central Configs distribution."""
    return (root / "distribution.yaml").is_file() and (root / "domains").is_dir()


def load_central_ontology(
    root: Path,
    library_root: Path | None,
    ignore_names: list[str],
) -> OntologyLoadResult:
    """Load a Central distribution and optional library into runtime artifacts.

    Args:
        root: Vendored ``central-configs`` distribution root.
        library_root: Vendored ``central-library`` root, when present.
        ignore_names: Filenames skipped while reading YAML.

    Returns:
        Ontology load result whose entity ids are ``coe:`` CURIEs.
    """
    sources = load_sources(root, list(_CENTRAL_GLOBS), ignore_names)
    library_sources: list[OntologySource] = []
    if library_root is not None and library_root.is_dir():
        library_sources = [
            OntologySource(
                file_path=source.file_path,
                file_rel_path=f"library/{source.file_rel_path}",
                blake3_hash=source.blake3_hash,
                data=source.data,
            )
            for source in load_sources(library_root, list(_LIBRARY_GLOBS), ignore_names)
        ]
    all_sources = [*sources, *library_sources]
    issues: list[OntologyValidationIssue] = []
    entities, relations = _project_resources(sources, issues)
    patterns, suppressed, anchors = _apply_library(entities, library_sources, issues)
    matcher_records = canonical_matcher_term_records(entities)
    return OntologyLoadResult(
        sources=all_sources,
        entity_definitions=entities,
        matcher_records=matcher_records,
        matcher_termset_hash=matcher_termset_hash(matcher_records),
        snapshot_hash=compute_snapshot_hash(all_sources),
        relation_records=relations,
        metadata_records=extract_metadata_records(all_sources, entities),
        coverage_report=coverage_report_for_sources(all_sources),
        validation_issues=issues,
        graph=build_graph(build_node_records(all_sources, entities, relations), relations),
        recognition_patterns=tuple(patterns),
        suppressed_terms=suppressed,
        anchor_constraints=tuple(anchors),
    )


def format_ontology_context(
    entity_definitions: list[dict[str, Any]],
    entity_ids: list[str],
) -> str:
    """Format definitions and library craft for matched entities.

    Args:
        entity_definitions: Projected entity records.
        entity_ids: Entity ids matched for the current question.

    Returns:
        Prompt block, or an empty string when no matched entity carries text.
    """
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
        lines.append(f"- **{label}**: {' '.join(parts)}")
    if not lines:
        return ""
    block = "## Ontology context\n\n" + "\n".join(lines) + "\n"
    if len(block) <= _PROMPT_CHAR_LIMIT:
        return block
    return block[: _PROMPT_CHAR_LIMIT - 1].rstrip() + "…\n"


def _project_resources(
    sources: list[OntologySource],
    issues: list[OntologyValidationIssue],
) -> tuple[list[dict[str, Any]], list[RelationRecord]]:
    found: list[tuple[str, str, str, dict[str, Any]]] = []
    for source in sources:
        data = source.data
        if not isinstance(data, dict):
            continue
        raw_domain = data.get("domain")
        domain = raw_domain if isinstance(raw_domain, str) else ""
        for key, kind in _COLLECTION_KINDS.items():
            items = data.get(key)
            if not isinstance(items, list):
                continue
            for item in items:
                _collect_resource(
                    item,
                    kind=kind,
                    source_rel=source.file_rel_path,
                    domain=domain,
                    found=found,
                )
    entities: list[dict[str, Any]] = []
    relations: list[RelationRecord] = []
    seen: set[str] = set()
    for kind, source_rel, domain, resource in found:
        resource_id = resource.get("id")
        if not isinstance(resource_id, str) or not resource_id:
            continue
        if resource_id in seen:
            issues.append(
                OntologyValidationIssue(
                    issue_kind="duplicate_resource",
                    source_file_rel_path=source_rel,
                    path=resource_id,
                    message=f"Duplicate ontology identifier '{resource_id}'.",
                )
            )
            continue
        seen.add(resource_id)
        entities.append(_entity_record(resource, kind=kind, domain=domain, source_rel=source_rel))
        relations.extend(_resource_relations(resource, source_rel=source_rel, issues=issues))
    return entities, relations


def _collect_resource(
    node: Any,
    *,
    kind: str,
    source_rel: str,
    domain: str,
    found: list[tuple[str, str, str, dict[str, Any]]],
) -> None:
    if not isinstance(node, dict):
        return
    if isinstance(node.get("id"), str):
        found.append((kind, source_rel, domain, node))
    for child_key, child_kind in _CHILD_KINDS.items():
        children = node.get(child_key)
        if not isinstance(children, list):
            continue
        for child in children:
            _collect_resource(
                child,
                kind=child_kind,
                source_rel=source_rel,
                domain=domain,
                found=found,
            )


def _entity_record(
    resource: dict[str, Any],
    *,
    kind: str,
    domain: str,
    source_rel: str,
) -> dict[str, Any]:
    resource_id = resource["id"]
    label = resource.get("label")
    label_text = (
        label.strip() if isinstance(label, str) and label.strip() else _local_name(resource_id)
    )
    description = resource.get("description")
    home = resource.get("home_domain")
    return {
        "canonical_id": resource_id,
        "label": label_text,
        "description": description.strip() if isinstance(description, str) else "",
        "aliases": _alias_names(resource, label_text),
        "indicators": [],
        "entity_kind": kind,
        "entity_type": kind,
        "domain": home if isinstance(home, str) and home else domain,
        "family": "",
        "source_file_rel_path": source_rel,
        "source_meta_id": None,
        "status": resource.get("status"),
        "craft_summary": "",
        "confidence_threshold": None,
        "ner_label": None,
    }


def _alias_names(resource: dict[str, Any], label: str) -> list[str]:
    names: list[str] = [label]
    for term in resource.get("terms") or []:
        if not isinstance(term, dict):
            continue
        literal = term.get("literal")
        if not isinstance(literal, str) or not literal.strip():
            continue
        term_type = term.get("term_type")
        normalized = normalize_match_text(literal)
        if len(normalized) < 3 and term_type not in _SHORT_TERM_TYPES:
            continue
        if len(normalized) < 2:
            continue
        names.append(literal.strip())
    for example in resource.get("examples") or []:
        if isinstance(example, str) and 3 <= len(example.strip()) <= 80:
            names.append(example.strip())
    return _dedupe(names)


def _resource_relations(
    resource: dict[str, Any],
    *,
    source_rel: str,
    issues: list[OntologyValidationIssue],
) -> list[RelationRecord]:
    resource_id = resource.get("id")
    if not isinstance(resource_id, str):
        return []
    records: list[RelationRecord] = []
    edges = resource.get("edges")
    if isinstance(edges, list):
        for index, edge in enumerate(edges):
            if not isinstance(edge, dict):
                issues.append(
                    OntologyValidationIssue(
                        issue_kind="invalid_edge",
                        source_file_rel_path=source_rel,
                        path=f"{resource_id}.edges[{index}]",
                        message="Semantic edges must be mappings.",
                    )
                )
                continue
            predicate = edge.get("predicate")
            object_id = edge.get("object_id")
            if not isinstance(predicate, str) or not isinstance(object_id, str):
                issues.append(
                    OntologyValidationIssue(
                        issue_kind="invalid_edge",
                        source_file_rel_path=source_rel,
                        path=f"{resource_id}.edges[{index}]",
                        message="Semantic edges require predicate and object_id.",
                    )
                )
                continue
            records.append(
                _relation(
                    source_id=resource_id,
                    relation_type=_local_predicate(predicate),
                    target_id=object_id,
                    source_rel=source_rel,
                    origin_path=f"{resource_id}.edges[{index}]",
                    metadata={
                        key: value
                        for key, value in edge.items()
                        if key not in {"predicate", "object_id", "subject_id"}
                    },
                )
            )
    for slot in ("broader", "narrower"):
        targets = resource.get(slot)
        if isinstance(targets, list):
            for target in targets:
                if isinstance(target, str):
                    records.append(
                        _relation(
                            source_id=resource_id,
                            relation_type=slot,
                            target_id=target,
                            source_rel=source_rel,
                            origin_path=f"{resource_id}.{slot}",
                            metadata={},
                        )
                    )
    in_scheme = resource.get("in_scheme")
    if isinstance(in_scheme, str):
        records.append(
            _relation(
                source_id=resource_id,
                relation_type="in_scheme",
                target_id=in_scheme,
                source_rel=source_rel,
                origin_path=f"{resource_id}.in_scheme",
                metadata={},
            )
        )
    for slot in _PORTFOLIO_SLOTS:
        records.extend(
            _slot_relations(resource_id, slot, resource.get(slot), source_rel=source_rel)
        )
    return records


def _slot_relations(
    source_id: str,
    slot: str,
    value: Any,
    *,
    source_rel: str,
) -> list[RelationRecord]:
    targets = value if isinstance(value, list) else [value]
    records: list[RelationRecord] = []
    for target in targets:
        if isinstance(target, str) and target:
            records.append(
                _relation(
                    source_id=source_id,
                    relation_type=slot,
                    target_id=target,
                    source_rel=source_rel,
                    origin_path=f"{source_id}.{slot}",
                    metadata={},
                )
            )
    return records


def _relation(
    *,
    source_id: str,
    relation_type: str,
    target_id: str,
    source_rel: str,
    origin_path: str,
    metadata: dict[str, Any],
) -> RelationRecord:
    return RelationRecord(
        relation_type=relation_type,
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
        metadata=metadata,
    )


def _apply_library(
    entities: list[dict[str, Any]],
    library_sources: list[OntologySource],
    issues: list[OntologyValidationIssue],
) -> tuple[list[RecognitionPattern], dict[str, frozenset[str]], list[AnchorConstraint]]:
    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in library_sources:
        data = source.data
        if not isinstance(data, dict):
            continue
        for slot in _LIBRARY_SLOTS:
            block = data.get(slot)
            if not isinstance(block, dict):
                continue
            for record_id, record in block.items():
                if not isinstance(record, dict):
                    continue
                stamped = {**record, "id": record.get("id", record_id), "record_kind": slot}
                for subject in _about_ids(stamped):
                    by_subject[subject].append(stamped)
    patterns: list[RecognitionPattern] = []
    suppressed: dict[str, frozenset[str]] = {}
    anchors: list[AnchorConstraint] = []
    for entity in entities:
        records = by_subject.get(entity["canonical_id"], [])
        if not records:
            continue
        entity["craft_summary"] = _craft_summary(records)
        _apply_recognition(
            entity, records, source_patterns=patterns, anchors=anchors, issues=issues
        )
        negatives = _negative_surfaces(records)
        if negatives:
            suppressed[entity["canonical_id"]] = frozenset(negatives)
    return patterns, suppressed, anchors


def _apply_recognition(
    entity: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    source_patterns: list[RecognitionPattern],
    anchors: list[AnchorConstraint],
    issues: list[OntologyValidationIssue],
) -> None:
    indicators: list[str] = []
    confidences: list[float] = []
    for record in records:
        if record.get("record_kind") != "recognition_calibrations":
            continue
        if record.get("status") == "retired":
            continue
        ner_label = record.get("ner_label")
        if isinstance(ner_label, str) and ner_label.strip():
            entity["ner_label"] = ner_label.strip()
        confidence = record.get("confidence_threshold")
        if isinstance(confidence, int | float):
            confidences.append(float(confidence))
        for surface in record.get("surfaces") or []:
            if isinstance(surface, str) and len(normalize_match_text(surface)) >= 3:
                indicators.append(surface.strip())
        for raw in record.get("recognition_patterns") or []:
            if not isinstance(raw, str) or not raw.strip():
                continue
            if _REGEX_META.search(raw) is None:
                if len(normalize_match_text(raw)) >= 3:
                    indicators.append(raw.strip())
                continue
            try:
                compiled = re.compile(raw, re.IGNORECASE)
            except re.error as exc:
                issues.append(
                    OntologyValidationIssue(
                        issue_kind="invalid_recognition_pattern",
                        source_file_rel_path=str(entity["source_file_rel_path"]),
                        path=str(record.get("id")),
                        message=f"Could not compile recognition pattern: {exc}",
                    )
                )
                continue
            source_patterns.append(
                RecognitionPattern(entity_id=entity["canonical_id"], compiled=compiled)
            )
        anchor_values = record.get("context_anchors")
        if isinstance(anchor_values, list) and anchor_values:
            normalized_anchors = frozenset(
                normalize_match_text(item)
                for item in anchor_values
                if isinstance(item, str) and normalize_match_text(item)
            )
            if normalized_anchors:
                term = record.get("term")
                surface = (
                    normalize_match_text(term) if isinstance(term, str) and term.strip() else None
                )
                window = record.get("anchor_window_tokens")
                anchors.append(
                    AnchorConstraint(
                        entity_id=entity["canonical_id"],
                        surface=surface,
                        anchors=normalized_anchors,
                        window_tokens=window if isinstance(window, int) and window > 0 else 12,
                    )
                )
    if indicators:
        entity["indicators"] = _dedupe([*entity.get("indicators", []), *indicators])
    if confidences:
        entity["confidence_threshold"] = min(confidences)


def _negative_surfaces(records: list[dict[str, Any]]) -> list[str]:
    negatives: list[str] = []
    for record in records:
        if record.get("record_kind") != "recognition_calibrations":
            continue
        for surface in record.get("negative_surfaces") or []:
            if isinstance(surface, str):
                normalized = normalize_match_text(surface)
                if normalized:
                    negatives.append(normalized)
    return negatives


def _about_ids(record: dict[str, Any]) -> list[str]:
    about = record.get("about")
    if isinstance(about, str):
        return [about]
    if isinstance(about, list):
        return [item for item in about if isinstance(item, str) and item]
    return []


def _craft_summary(records: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for record in records:
        if record.get("status") == "retired":
            continue
        kind = record.get("record_kind")
        if kind == "prescriptions":
            statement = record.get("statement")
            if isinstance(statement, str) and statement.strip():
                parts.append(statement.strip())
        elif kind == "principle_evidence":
            direction = record.get("direction")
            if isinstance(direction, str) and direction.strip():
                effect = record.get("effect_size")
                metric = record.get("effect_metric")
                suffix = ""
                if isinstance(effect, int | float):
                    if isinstance(metric, str) and metric.strip():
                        suffix = f" (effect {effect} {metric.strip()})"
                    else:
                        suffix = f" (effect {effect})"
                parts.append(direction.strip() + suffix)
        elif kind == "thresholds":
            phrase = _threshold_phrase(record)
            if phrase:
                parts.append(phrase)
        elif kind == "scoring_models":
            rule = record.get("pass_rule")
            if isinstance(rule, str) and rule.strip():
                parts.append(rule.strip())
        elif kind == "profiles":
            facet = record.get("facet")
            values = record.get("facet_values")
            if isinstance(facet, str) and isinstance(values, list) and values:
                rendered = ", ".join(str(value) for value in values[:4])
                parts.append(f"{facet}: {rendered}")
        if len(parts) >= 6:
            break
    return " ".join(parts)[:_CRAFT_CHAR_LIMIT]


def _threshold_phrase(record: dict[str, Any]) -> str:
    quantity = record.get("quantity")
    unit = record.get("unit")
    bound = record.get("bound_kind")
    if not isinstance(quantity, str) or not isinstance(unit, str) or not isinstance(bound, str):
        return ""
    value = record.get("target_value")
    if value is None:
        value = record.get("maximum_value")
    if value is None:
        value = record.get("minimum_value")
    if not isinstance(value, int | float):
        return ""
    return f"{quantity} {bound} {value} {unit}"


def _local_predicate(predicate: str) -> str:
    if predicate.startswith("coe:"):
        return predicate.removeprefix("coe:")
    return predicate


def _local_name(resource_id: str) -> str:
    return resource_id.rsplit("/", 1)[-1].replace("_", " ")


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(value)
    return unique
