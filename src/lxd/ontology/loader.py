"""Load ontology data and validate source metadata."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from blake3 import blake3

from lxd.domain.ids import blake3_hex
from lxd.ontology.graph import OntologyNodeRecord, RelationRecord, build_graph
from lxd.ontology.inventory import OntologyCoverageReport, build_coverage_report, discover_key_paths
from lxd.ontology.matcher import (
    MatcherTermRecord,
    canonical_matcher_term_records,
    matcher_termset_hash,
)


@dataclass(frozen=True)
class OntologySource:
    """Loaded ontology source file and parsed payload."""
    file_path: Path
    file_rel_path: str
    blake3_hash: str
    data: Any


@dataclass(frozen=True)
class OntologyMetadataRecord:
    """Ontology metadata row derived from file or entity payload."""
    record_kind: str
    source_file_rel_path: str
    entity_id: str | None
    payload: dict[str, Any]


@dataclass(frozen=True)
class OntologyValidationIssue:
    """Validation issue found during ontology loading."""
    issue_kind: str
    source_file_rel_path: str
    path: str
    message: str


@dataclass(frozen=True)
class OntologyLoadResult:
    """All artifacts produced by ontology loading."""
    sources: list[OntologySource]
    entity_definitions: list[dict[str, Any]]
    matcher_records: list[MatcherTermRecord]
    matcher_termset_hash: str
    snapshot_hash: str
    relation_records: list[RelationRecord]
    metadata_records: list[OntologyMetadataRecord]
    coverage_report: OntologyCoverageReport
    validation_issues: list[OntologyValidationIssue]
    graph: Any


@dataclass(frozen=True)
class _RelationSchema:
    file_relation_types: dict[str, dict[str, Any]]
    entity_relation_types: dict[str, dict[str, Any]]
    entity_relation_weights: set[str]


class _IncludeLoader(yaml.SafeLoader):
    pass


def _include_constructor(loader: _IncludeLoader, node: yaml.nodes.Node) -> Any:
    if not isinstance(node, yaml.ScalarNode):
        raise TypeError("!include expects a scalar path")
    include_path = Path(loader.name).parent / loader.construct_scalar(node)
    with include_path.open("r", encoding="utf-8") as handle:
        child_loader = _IncludeLoader(handle)
        child_loader.name = str(include_path)
        try:
            return child_loader.get_single_data()
        finally:
            child_loader.dispose()


_IncludeLoader.add_constructor("!include", _include_constructor)


def load_ontology(
    root: Path, include_globs: list[str], ignore_names: list[str]
) -> OntologyLoadResult:
    """Load ontology sources and derive runtime artifacts.

    Args:
        root: Ontology root directory.
        include_globs: Glob patterns selecting ontology files.
        ignore_names: Filenames to ignore while loading.

    Returns:
        Loaded ontology artifacts and derived indexes.
    """
    sources = _load_sources(root, include_globs, ignore_names)
    coverage_report = _coverage_report_for_sources(sources)
    entity_definitions = _extract_entity_definitions(sources)
    matcher_records = canonical_matcher_term_records(entity_definitions)
    snapshot_hash = _snapshot_hash(sources)
    metadata_records = _extract_metadata_records(sources, entity_definitions)
    relation_schema = _extract_relation_schema(sources)
    relation_records, validation_issues = _extract_relations(sources, entity_definitions, relation_schema)
    graph = build_graph(
        _build_node_records(sources, entity_definitions, relation_records),
        relation_records,
    )
    return OntologyLoadResult(
        sources=sources,
        entity_definitions=entity_definitions,
        matcher_records=matcher_records,
        matcher_termset_hash=matcher_termset_hash(matcher_records),
        snapshot_hash=snapshot_hash,
        relation_records=relation_records,
        metadata_records=metadata_records,
        coverage_report=coverage_report,
        validation_issues=validation_issues,
        graph=graph,
    )


def _load_sources(
    root: Path, include_globs: list[str], ignore_names: list[str]
) -> list[OntologySource]:
    seen: set[Path] = set()
    collected: list[OntologySource] = []
    for pattern in include_globs:
        for path in sorted(root.glob(pattern)):
            if not path.is_file() or path.name in ignore_names or path in seen:
                continue
            seen.add(path)
            collected.append(
                OntologySource(
                    file_path=path,
                    file_rel_path=str(path.relative_to(root)),
                    blake3_hash=_file_hash(path),
                    data=_load_yaml_with_includes(path),
                )
            )
    return collected


def _load_yaml_with_includes(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        loader = _IncludeLoader(handle)
        loader.name = str(path)
        try:
            return loader.get_single_data()
        finally:
            loader.dispose()


def _file_hash(path: Path) -> str:
    hasher = blake3()
    with path.open("rb") as handle:
        while chunk := handle.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def _snapshot_hash(sources: list[OntologySource]) -> str:
    payload = "\n".join(
        json.dumps(
            {
                "file_rel_path": source.file_rel_path,
                "data": _canonicalize_for_hashing(source.data),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        for source in sorted(sources, key=lambda item: item.file_rel_path)
    )
    return blake3_hex(payload)


def _canonicalize_for_hashing(value: Any) -> Any:
    if isinstance(value, dict):
        items = [
            {
                "key": _canonicalize_key(key),
                "value": _canonicalize_for_hashing(child),
            }
            for key, child in sorted(value.items(), key=_mapping_item_sort_key)
        ]
        return {"__type__": "mapping", "items": items}
    if isinstance(value, list):
        return [_canonicalize_for_hashing(item) for item in value]
    return value


def _canonicalize_key(key: Any) -> str:
    if isinstance(key, str):
        return f"str:{key}"
    if isinstance(key, bool):
        return f"bool:{str(key).lower()}"
    if key is None:
        return "none:null"
    if isinstance(key, int):
        return f"int:{key}"
    if isinstance(key, float):
        return f"float:{key!r}"
    return f"{type(key).__name__}:{key!r}"


def _mapping_item_sort_key(item: tuple[Any, Any]) -> str:
    return _canonicalize_key(item[0])


def _coverage_report_for_sources(sources: list[OntologySource]) -> OntologyCoverageReport:
    path_counts: dict[str, int] = defaultdict(int)
    for source in sources:
        discovered = discover_key_paths(source.data)
        for path, count in discovered.items():
            path_counts[path] += count
    return build_coverage_report(path_counts)


def _extract_entity_definitions(sources: list[OntologySource]) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    for source in sources:
        data = source.data
        if not isinstance(data, dict):
            continue
        entity_types = data.get("entity_types")
        if not isinstance(entity_types, dict):
            continue
        source_meta = data.get("_meta") if isinstance(data.get("_meta"), dict) else {}
        source_meta_id = source_meta.get("id") if isinstance(source_meta, dict) else None
        for entity_id, payload in entity_types.items():
            if not isinstance(payload, dict):
                continue
            merged = {
                "canonical_id": entity_id,
                **payload,
                "source_file_rel_path": source.file_rel_path,
                "source_meta_id": source_meta_id,
            }
            entities.append(merged)
    return entities


def _extract_metadata_records(
    sources: list[OntologySource], entity_definitions: list[dict[str, Any]]
) -> list[OntologyMetadataRecord]:
    records: list[OntologyMetadataRecord] = []
    for source in sources:
        if isinstance(source.data, dict):
            records.append(
                OntologyMetadataRecord(
                    record_kind="file",
                    source_file_rel_path=source.file_rel_path,
                    entity_id=None,
                    payload=dict(source.data),
                )
            )
    for entity in entity_definitions:
        payload = {
            key: value
            for key, value in entity.items()
            if key not in {"source_file_rel_path", "source_meta_id"}
        }
        records.append(
            OntologyMetadataRecord(
                record_kind="entity",
                source_file_rel_path=_coerce_required_str(entity, "source_file_rel_path"),
                entity_id=_coerce_required_str(entity, "canonical_id"),
                payload=payload,
            )
        )
    return records


def _extract_relation_schema(sources: list[OntologySource]) -> _RelationSchema:
    file_relation_types: dict[str, dict[str, Any]] = {}
    entity_relation_types: dict[str, dict[str, Any]] = {}
    entity_relation_weights: set[str] = set()
    for source in sources:
        if not isinstance(source.data, dict):
            continue
        file_relationships = source.data.get("file_relationships")
        if isinstance(file_relationships, dict):
            for relation_type, payload in file_relationships.items():
                if isinstance(payload, dict) and isinstance(relation_type, str):
                    file_relation_types[relation_type] = dict(payload)
        entity_relations = source.data.get("entity_relations")
        if isinstance(entity_relations, dict):
            for relation_type, payload in entity_relations.items():
                if isinstance(payload, dict) and isinstance(relation_type, str):
                    entity_relation_types[relation_type] = dict(payload)
        relation_weights = source.data.get("entity_relation_weights")
        if isinstance(relation_weights, dict):
            for weight_name in relation_weights:
                if isinstance(weight_name, str):
                    entity_relation_weights.add(weight_name)
    return _RelationSchema(
        file_relation_types=file_relation_types,
        entity_relation_types=entity_relation_types,
        entity_relation_weights=entity_relation_weights,
    )


def _extract_relations(
    sources: list[OntologySource],
    entity_definitions: list[dict[str, Any]],
    relation_schema: _RelationSchema,
) -> tuple[list[RelationRecord], list[OntologyValidationIssue]]:
    relations: list[RelationRecord] = []
    issues: list[OntologyValidationIssue] = []
    valid_entity_ids = {entity["canonical_id"] for entity in entity_definitions}
    entity_target_index = _build_entity_target_index(entity_definitions)
    sources_by_rel_path = {source.file_rel_path: source for source in sources}
    sources_by_meta_id = {
        meta_id: source
        for source in sources
        for meta_id in [_source_meta_id(source)]
        if meta_id is not None
    }
    top_level_key_index = _build_top_level_key_index(sources)
    for source in sources:
        if not isinstance(source.data, dict):
            continue
        relations.extend(
            _extract_file_relationships(
                source=source,
                relation_schema=relation_schema,
                sources_by_rel_path=sources_by_rel_path,
                sources_by_meta_id=sources_by_meta_id,
                issues=issues,
            )
        )
        entity_types = source.data.get("entity_types")
        if not isinstance(entity_types, dict):
            continue
        for entity_id, payload in entity_types.items():
            if entity_id not in valid_entity_ids or not isinstance(payload, dict):
                continue
            relations.extend(
                _extract_entity_relationships(
                    source=source,
                    entity_id=entity_id,
                    payload=payload,
                    valid_entity_ids=valid_entity_ids,
                    entity_target_index=entity_target_index,
                    relation_schema=relation_schema,
                    sources_by_meta_id=sources_by_meta_id,
                    top_level_key_index=top_level_key_index,
                    issues=issues,
                )
            )
    return relations, issues


def _extract_file_relationships(
    *,
    source: OntologySource,
    relation_schema: _RelationSchema,
    sources_by_rel_path: dict[str, OntologySource],
    sources_by_meta_id: dict[str, OntologySource],
    issues: list[OntologyValidationIssue],
) -> list[RelationRecord]:
    source_meta = source.data.get("_meta")
    if not isinstance(source_meta, dict):
        return []
    relationships = source_meta.get("relationships")
    if not isinstance(relationships, list):
        return []
    records: list[RelationRecord] = []
    source_node_id = _file_node_id(source.file_rel_path)
    for index, item in enumerate(relationships):
        origin_path = "_meta.relationships.*"
        if not isinstance(item, dict):
            issues.append(
                OntologyValidationIssue(
                    issue_kind="invalid_file_relationship",
                    source_file_rel_path=source.file_rel_path,
                    path=origin_path,
                    message="Expected _meta.relationships entries to be mappings.",
                )
            )
            continue
        target = item.get("target")
        relation_type = item.get("type")
        if not isinstance(target, str) or not isinstance(relation_type, str):
            issues.append(
                OntologyValidationIssue(
                    issue_kind="invalid_file_relationship",
                    source_file_rel_path=source.file_rel_path,
                    path=f"{origin_path}[{index}]",
                    message="File relationship entries must include string target and type.",
                )
            )
            continue
        if relation_schema.file_relation_types and relation_type not in relation_schema.file_relation_types:
            issues.append(
                OntologyValidationIssue(
                    issue_kind="unknown_file_relation_type",
                    source_file_rel_path=source.file_rel_path,
                    path=f"{origin_path}[{index}].type",
                    message=f"Undeclared file relation type '{relation_type}'.",
                )
            )
        resolved_target = _resolve_relation_target_file(
            source=source,
            target=target,
            sources_by_rel_path=sources_by_rel_path,
            sources_by_meta_id=sources_by_meta_id,
        )
        target_node_id = (
            _file_node_id(resolved_target.file_rel_path)
            if resolved_target is not None
            else _external_file_node_id(target)
        )
        records.append(
            RelationRecord(
                relation_type=relation_type,
                origin_kind="file_meta",
                origin_path=origin_path,
                source_file_rel_path=source.file_rel_path,
                source_node_id=source_node_id,
                source_node_type="ontology_file",
                source_entity_id=None,
                target_node_id=target_node_id,
                target_node_type="ontology_file" if resolved_target is not None else "external_file",
                target_entity_id=None,
                target_file_rel_path=resolved_target.file_rel_path if resolved_target is not None else None,
                metadata={key: value for key, value in item.items() if key not in {"target", "type"}},
            )
        )
    return records


def _extract_entity_relationships(
    *,
    source: OntologySource,
    entity_id: str,
    payload: dict[str, Any],
    valid_entity_ids: set[str],
    entity_target_index: dict[str, str],
    relation_schema: _RelationSchema,
    sources_by_meta_id: dict[str, OntologySource],
    top_level_key_index: dict[str, list[OntologySource]],
    issues: list[OntologyValidationIssue],
) -> list[RelationRecord]:
    records: list[RelationRecord] = []
    source_node_id = entity_id
    parent_entity = payload.get("parent_entity")
    if parent_entity is None:
        pass
    elif isinstance(parent_entity, str):
        records.append(
            _entity_relation_record(
                source=source,
                entity_id=entity_id,
                relation_type="parent_entity",
                origin_path=f"entity_types.{entity_id}.parent_entity",
                target_name=parent_entity,
                target_file_hint=None,
                valid_entity_ids=valid_entity_ids,
                entity_target_index=entity_target_index,
                metadata={},
                issues=issues,
            )
        )
    else:
        issues.append(
            OntologyValidationIssue(
                issue_kind="invalid_parent_entity",
                source_file_rel_path=source.file_rel_path,
                path=f"entity_types.{entity_id}.parent_entity",
                message="parent_entity must be null or a string.",
            )
        )
    relates_to = payload.get("relates_to")
    if relates_to is not None:
        if not isinstance(relates_to, list):
            issues.append(
                OntologyValidationIssue(
                    issue_kind="invalid_relates_to",
                    source_file_rel_path=source.file_rel_path,
                    path=f"entity_types.{entity_id}.relates_to",
                    message="relates_to must be a list.",
                )
            )
        else:
            for item in relates_to:
                if isinstance(item, str):
                    relation_type = "relates_to"
                    metadata: dict[str, Any] = {}
                    target_name = item
                    target_file_hint = None
                elif isinstance(item, dict):
                    target_name = item.get("target")
                    relation_type = item.get("relation", "relates_to")
                    target_file_hint = item.get("target_file")
                    metadata = {
                        key: value
                        for key, value in item.items()
                        if key not in {"target", "relation", "target_file"}
                    }
                else:
                    issues.append(
                        OntologyValidationIssue(
                            issue_kind="invalid_entity_relation",
                            source_file_rel_path=source.file_rel_path,
                            path=f"entity_types.{entity_id}.relates_to.*",
                            message="Entity relates_to entries must be strings or mappings.",
                        )
                    )
                    continue
                if not isinstance(target_name, str) or not isinstance(relation_type, str):
                    issues.append(
                        OntologyValidationIssue(
                            issue_kind="invalid_entity_relation",
                            source_file_rel_path=source.file_rel_path,
                            path=f"entity_types.{entity_id}.relates_to.*",
                            message="Entity relations require string target and relation fields.",
                        )
                    )
                    continue
                _validate_entity_relation_schema(
                    relation_schema=relation_schema,
                    source_file_rel_path=source.file_rel_path,
                    entity_id=entity_id,
                    relation_type=relation_type,
                    metadata=metadata,
                    issues=issues,
                )
                records.append(
                    _entity_relation_record(
                        source=source,
                        entity_id=entity_id,
                        relation_type=relation_type,
                        origin_path=f"entity_types.{entity_id}.relates_to.*",
                        target_name=target_name,
                        target_file_hint=target_file_hint if isinstance(target_file_hint, str) else None,
                        valid_entity_ids=valid_entity_ids,
                        entity_target_index=entity_target_index,
                        metadata=metadata,
                        issues=issues,
                    )
                )
    taxonomy_mapping = payload.get("taxonomy_mapping")
    if taxonomy_mapping is not None:
        records.extend(
            _extract_taxonomy_mapping_relations(
                source=source,
                entity_id=entity_id,
                taxonomy_mapping=taxonomy_mapping,
                sources_by_meta_id=sources_by_meta_id,
                issues=issues,
            )
        )
    maps_to_taxonomy_types = payload.get("maps_to_taxonomy_types")
    if maps_to_taxonomy_types is not None:
        records.extend(
            _extract_taxonomy_type_relations(
                source=source,
                entity_id=entity_id,
                payload=payload,
                maps_to_taxonomy_types=maps_to_taxonomy_types,
                issues=issues,
            )
        )
    taxonomy_reference = payload.get("taxonomy_reference")
    if taxonomy_reference is not None:
        records.extend(
            _extract_taxonomy_reference_relations(
                source=source,
                entity_id=entity_id,
                taxonomy_reference=taxonomy_reference,
                validate_against_taxonomy=payload.get("validate_against_taxonomy"),
                sources_by_meta_id=sources_by_meta_id,
                top_level_key_index=top_level_key_index,
                issues=issues,
            )
        )
    return [record for record in records if record.source_node_id == source_node_id]


def _validate_entity_relation_schema(
    *,
    relation_schema: _RelationSchema,
    source_file_rel_path: str,
    entity_id: str,
    relation_type: str,
    metadata: dict[str, Any],
    issues: list[OntologyValidationIssue],
) -> None:
    if relation_schema.entity_relation_types and relation_type not in relation_schema.entity_relation_types:
        issues.append(
            OntologyValidationIssue(
                issue_kind="unknown_entity_relation_type",
                source_file_rel_path=source_file_rel_path,
                path=f"entity_types.{entity_id}.relates_to.*.relation",
                message=f"Undeclared entity relation type '{relation_type}'.",
            )
        )
    weight = metadata.get("weight")
    if (
        relation_schema.entity_relation_weights
        and weight is not None
        and isinstance(weight, str)
        and weight not in relation_schema.entity_relation_weights
    ):
        issues.append(
            OntologyValidationIssue(
                issue_kind="unknown_entity_relation_weight",
                source_file_rel_path=source_file_rel_path,
                path=f"entity_types.{entity_id}.relates_to.*.weight",
                message=f"Undeclared entity relation weight '{weight}'.",
            )
        )


def _entity_relation_record(
    *,
    source: OntologySource,
    entity_id: str,
    relation_type: str,
    origin_path: str,
    target_name: str,
    target_file_hint: str | None,
    valid_entity_ids: set[str],
    entity_target_index: dict[str, str],
    metadata: dict[str, Any],
    issues: list[OntologyValidationIssue],
) -> RelationRecord:
    target_entity_id = _resolve_entity_target(
        target_name,
        valid_entity_ids=valid_entity_ids,
        entity_target_index=entity_target_index,
    )
    if target_entity_id is None:
        issues.append(
            OntologyValidationIssue(
                issue_kind="unresolved_entity_relation_target",
                source_file_rel_path=source.file_rel_path,
                path=origin_path,
                message=f"Unresolved entity relation target '{target_name}'.",
            )
        )
    return RelationRecord(
        relation_type=relation_type,
        origin_kind="entity",
        origin_path=origin_path,
        source_file_rel_path=source.file_rel_path,
        source_node_id=entity_id,
        source_node_type="entity",
        source_entity_id=entity_id,
        target_node_id=target_entity_id or _unresolved_entity_node_id(target_name),
        target_node_type="entity" if target_entity_id is not None else "unresolved_entity",
        target_entity_id=target_entity_id,
        target_file_rel_path=target_file_hint,
        metadata=metadata,
    )


def _extract_taxonomy_mapping_relations(
    *,
    source: OntologySource,
    entity_id: str,
    taxonomy_mapping: Any,
    sources_by_meta_id: dict[str, OntologySource],
    issues: list[OntologyValidationIssue],
) -> list[RelationRecord]:
    if not isinstance(taxonomy_mapping, list):
        issues.append(
            OntologyValidationIssue(
                issue_kind="invalid_taxonomy_mapping",
                source_file_rel_path=source.file_rel_path,
                path=f"entity_types.{entity_id}.taxonomy_mapping",
                message="taxonomy_mapping must be a list.",
            )
        )
        return []
    records: list[RelationRecord] = []
    for item in taxonomy_mapping:
        if not isinstance(item, dict):
            issues.append(
                OntologyValidationIssue(
                    issue_kind="invalid_taxonomy_mapping",
                    source_file_rel_path=source.file_rel_path,
                    path=f"entity_types.{entity_id}.taxonomy_mapping.*",
                    message="taxonomy_mapping entries must be mappings.",
                )
            )
            continue
        taxonomy_id = item.get("taxonomy")
        dimension = item.get("dimension")
        values = item.get("values")
        if not isinstance(taxonomy_id, str) or not isinstance(dimension, str) or not isinstance(values, list):
            issues.append(
                OntologyValidationIssue(
                    issue_kind="invalid_taxonomy_mapping",
                    source_file_rel_path=source.file_rel_path,
                    path=f"entity_types.{entity_id}.taxonomy_mapping.*",
                    message="taxonomy_mapping requires string taxonomy/dimension and list values.",
                )
            )
            continue
        taxonomy_source = sources_by_meta_id.get(taxonomy_id)
        for value in values:
            if not isinstance(value, str):
                continue
            records.append(
                RelationRecord(
                    relation_type="maps_to_taxonomy_value",
                    origin_kind="taxonomy_mapping",
                    origin_path=f"entity_types.{entity_id}.taxonomy_mapping.*",
                    source_file_rel_path=source.file_rel_path,
                    source_node_id=entity_id,
                    source_node_type="entity",
                    source_entity_id=entity_id,
                    target_node_id=_taxonomy_value_node_id(taxonomy_id, dimension, value),
                    target_node_type="taxonomy_value",
                    target_entity_id=None,
                    target_file_rel_path=taxonomy_source.file_rel_path if taxonomy_source else None,
                    metadata={
                        "taxonomy": taxonomy_id,
                        "dimension": dimension,
                        "value": value,
                        **{key: child for key, child in item.items() if key not in {"taxonomy", "dimension", "values"}},
                    },
                )
            )
    return records


def _extract_taxonomy_type_relations(
    *,
    source: OntologySource,
    entity_id: str,
    payload: dict[str, Any],
    maps_to_taxonomy_types: Any,
    issues: list[OntologyValidationIssue],
) -> list[RelationRecord]:
    if not isinstance(maps_to_taxonomy_types, list):
        issues.append(
            OntologyValidationIssue(
                issue_kind="invalid_maps_to_taxonomy_types",
                source_file_rel_path=source.file_rel_path,
                path=f"entity_types.{entity_id}.maps_to_taxonomy_types",
                message="maps_to_taxonomy_types must be a list.",
            )
        )
        return []
    taxonomy_id = _infer_taxonomy_id(payload)
    records: list[RelationRecord] = []
    for value in maps_to_taxonomy_types:
        if not isinstance(value, str):
            continue
        records.append(
            RelationRecord(
                relation_type="maps_to_taxonomy_type",
                origin_kind="taxonomy_type_mapping",
                origin_path=f"entity_types.{entity_id}.maps_to_taxonomy_types.*",
                source_file_rel_path=source.file_rel_path,
                source_node_id=entity_id,
                source_node_type="entity",
                source_entity_id=entity_id,
                target_node_id=_taxonomy_type_node_id(taxonomy_id or "unknown_taxonomy", value),
                target_node_type="taxonomy_type",
                target_entity_id=None,
                target_file_rel_path=None,
                metadata={"taxonomy": taxonomy_id, "value": value},
            )
        )
    return records


def _extract_taxonomy_reference_relations(
    *,
    source: OntologySource,
    entity_id: str,
    taxonomy_reference: Any,
    validate_against_taxonomy: Any,
    sources_by_meta_id: dict[str, OntologySource],
    top_level_key_index: dict[str, list[OntologySource]],
    issues: list[OntologyValidationIssue],
) -> list[RelationRecord]:
    if not isinstance(taxonomy_reference, str):
        issues.append(
            OntologyValidationIssue(
                issue_kind="invalid_taxonomy_reference",
                source_file_rel_path=source.file_rel_path,
                path=f"entity_types.{entity_id}.taxonomy_reference",
                message="taxonomy_reference must be a string.",
            )
        )
        return []
    resolved_source = _resolve_named_reference_source(
        taxonomy_reference,
        sources_by_meta_id=sources_by_meta_id,
        top_level_key_index=top_level_key_index,
    )
    target_node_id = (
        _file_node_id(resolved_source.file_rel_path)
        if resolved_source is not None
        else _taxonomy_reference_node_id(taxonomy_reference)
    )
    target_node_type = "ontology_file" if resolved_source is not None else "taxonomy_reference"
    target_file_rel_path = resolved_source.file_rel_path if resolved_source is not None else None
    records = [
        RelationRecord(
            relation_type="references_taxonomy",
            origin_kind="taxonomy_reference",
            origin_path=f"entity_types.{entity_id}.taxonomy_reference",
            source_file_rel_path=source.file_rel_path,
            source_node_id=entity_id,
            source_node_type="entity",
            source_entity_id=entity_id,
            target_node_id=target_node_id,
            target_node_type=target_node_type,
            target_entity_id=None,
            target_file_rel_path=target_file_rel_path,
            metadata={"taxonomy_reference": taxonomy_reference},
        )
    ]
    if validate_against_taxonomy is True:
        records.append(
            RelationRecord(
                relation_type="validated_against_taxonomy",
                origin_kind="taxonomy_reference",
                origin_path=f"entity_types.{entity_id}.validate_against_taxonomy",
                source_file_rel_path=source.file_rel_path,
                source_node_id=entity_id,
                source_node_type="entity",
                source_entity_id=entity_id,
                target_node_id=target_node_id,
                target_node_type=target_node_type,
                target_entity_id=None,
                target_file_rel_path=target_file_rel_path,
                metadata={"taxonomy_reference": taxonomy_reference},
            )
        )
    elif validate_against_taxonomy not in {None, False}:
        issues.append(
            OntologyValidationIssue(
                issue_kind="invalid_validate_against_taxonomy",
                source_file_rel_path=source.file_rel_path,
                path=f"entity_types.{entity_id}.validate_against_taxonomy",
                message="validate_against_taxonomy must be boolean when present.",
            )
        )
    return records


def _build_node_records(
    sources: list[OntologySource],
    entity_definitions: list[dict[str, Any]],
    relations: list[RelationRecord],
) -> list[OntologyNodeRecord]:
    nodes: dict[str, OntologyNodeRecord] = {}
    for source in sources:
        source_meta = source.data.get("_meta") if isinstance(source.data, dict) else None
        label = None
        metadata: dict[str, Any] = {"file_rel_path": source.file_rel_path}
        if isinstance(source_meta, dict):
            title = source_meta.get("title")
            meta_id = source_meta.get("id")
            label = title if isinstance(title, str) else meta_id if isinstance(meta_id, str) else source.file_rel_path
            metadata.update(
                {
                    "meta_id": meta_id,
                    "purpose": source_meta.get("purpose"),
                    "domain": source_meta.get("domain"),
                    "domain_type": source_meta.get("domain_type"),
                }
            )
        nodes[_file_node_id(source.file_rel_path)] = OntologyNodeRecord(
            node_id=_file_node_id(source.file_rel_path),
            node_type="ontology_file",
            source_file_rel_path=source.file_rel_path,
            entity_id=None,
            label=label or source.file_rel_path,
            metadata=metadata,
        )
    for entity in entity_definitions:
        canonical_id = _coerce_required_str(entity, "canonical_id")
        label = entity.get("label")
        nodes[canonical_id] = OntologyNodeRecord(
            node_id=canonical_id,
            node_type="entity",
            source_file_rel_path=_coerce_required_str(entity, "source_file_rel_path"),
            entity_id=canonical_id,
            label=label if isinstance(label, str) else canonical_id,
            metadata={
                "entity_kind": entity.get("entity_kind"),
                "family": entity.get("family"),
                "source_meta_id": entity.get("source_meta_id"),
            },
        )
    for relation in relations:
        if relation.target_node_id in nodes:
            continue
        metadata = dict(relation.metadata)
        label = relation.target_node_id
        if relation.target_node_type in {"taxonomy_value", "taxonomy_type"}:
            label = str(metadata.get("value") or relation.target_node_id)
        elif relation.target_node_type == "taxonomy_reference":
            label = str(metadata.get("taxonomy_reference") or relation.target_node_id)
        elif relation.target_node_type == "external_file":
            label = relation.target_node_id.removeprefix("external_file:")
        elif relation.target_node_type == "unresolved_entity":
            label = relation.target_node_id.removeprefix("unresolved_entity:")
        nodes[relation.target_node_id] = OntologyNodeRecord(
            node_id=relation.target_node_id,
            node_type=relation.target_node_type,
            source_file_rel_path=relation.target_file_rel_path,
            entity_id=relation.target_entity_id,
            label=label,
            metadata=metadata,
        )
    return list(nodes.values())


def _build_top_level_key_index(sources: list[OntologySource]) -> dict[str, list[OntologySource]]:
    index: dict[str, list[OntologySource]] = defaultdict(list)
    for source in sources:
        if not isinstance(source.data, dict):
            continue
        for key in source.data:
            if isinstance(key, str):
                index[key].append(source)
    return index


def _build_entity_target_index(entity_definitions: list[dict[str, Any]]) -> dict[str, str]:
    index: dict[str, str] = {}
    folded_candidates: dict[str, set[str]] = defaultdict(set)
    for entity in entity_definitions:
        canonical_id = _coerce_required_str(entity, "canonical_id")
        for candidate in (
            canonical_id,
            entity.get("gliner_label"),
            entity.get("label"),
        ):
            if not isinstance(candidate, str) or not candidate.strip():
                continue
            index.setdefault(candidate, canonical_id)
            folded_candidates[candidate.casefold()].add(canonical_id)
    for folded, canonical_ids in folded_candidates.items():
        if len(canonical_ids) == 1:
            index.setdefault(folded, next(iter(canonical_ids)))
    return index


def _resolve_entity_target(
    target_name: str,
    *,
    valid_entity_ids: set[str],
    entity_target_index: dict[str, str],
) -> str | None:
    if target_name in valid_entity_ids:
        return target_name
    if target_name in entity_target_index:
        return entity_target_index[target_name]
    return entity_target_index.get(target_name.casefold())


def _resolve_relation_target_file(
    *,
    source: OntologySource,
    target: str,
    sources_by_rel_path: dict[str, OntologySource],
    sources_by_meta_id: dict[str, OntologySource],
) -> OntologySource | None:
    if target in sources_by_meta_id:
        return sources_by_meta_id[target]
    normalized = _normalize_rel_path(target)
    if normalized in sources_by_rel_path:
        return sources_by_rel_path[normalized]
    source_dir = PurePosixPath(source.file_rel_path).parent
    relative = _normalize_rel_path(str(source_dir / target))
    return sources_by_rel_path.get(relative)


def _resolve_named_reference_source(
    reference_name: str,
    *,
    sources_by_meta_id: dict[str, OntologySource],
    top_level_key_index: dict[str, list[OntologySource]],
) -> OntologySource | None:
    if reference_name in sources_by_meta_id:
        return sources_by_meta_id[reference_name]
    candidates = top_level_key_index.get(reference_name, [])
    if len(candidates) == 1:
        return candidates[0]
    return None


def _source_meta_id(source: OntologySource) -> str | None:
    if not isinstance(source.data, dict):
        return None
    source_meta = source.data.get("_meta")
    if not isinstance(source_meta, dict):
        return None
    meta_id = source_meta.get("id")
    return meta_id if isinstance(meta_id, str) and meta_id.strip() else None


def _infer_taxonomy_id(payload: dict[str, Any]) -> str | None:
    taxonomy_mapping = payload.get("taxonomy_mapping")
    if isinstance(taxonomy_mapping, list):
        for item in taxonomy_mapping:
            if isinstance(item, dict):
                taxonomy_id = item.get("taxonomy")
                if isinstance(taxonomy_id, str) and taxonomy_id.strip():
                    return taxonomy_id
    source_meta_id = payload.get("source_meta_id")
    if isinstance(source_meta_id, str) and source_meta_id.endswith("_entities"):
        return source_meta_id.removesuffix("_entities") + "_taxonomy"
    return None


def _normalize_rel_path(value: str) -> str:
    return str(PurePosixPath(value))


def _file_node_id(file_rel_path: str) -> str:
    return f"file:{file_rel_path}"


def _external_file_node_id(target: str) -> str:
    return f"external_file:{target}"


def _unresolved_entity_node_id(entity_name: str) -> str:
    return f"unresolved_entity:{entity_name}"


def _taxonomy_value_node_id(taxonomy_id: str, dimension: str, value: str) -> str:
    return f"taxonomy_value:{taxonomy_id}:{dimension}:{value}"


def _taxonomy_type_node_id(taxonomy_id: str, value: str) -> str:
    return f"taxonomy_type:{taxonomy_id}:{value}"


def _taxonomy_reference_node_id(reference_name: str) -> str:
    return f"taxonomy_reference:{reference_name}"


def _coerce_required_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing required string field: {key}")
    return value
