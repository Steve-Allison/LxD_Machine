"""Entity definition extraction, metadata records, and node building."""

from typing import Any

from lxd.ontology.graph import OntologyNodeRecord, RelationRecord
from lxd.ontology.loader.types import OntologyMetadataRecord, OntologySource


def extract_entity_definitions(sources: list[OntologySource]) -> list[dict[str, Any]]:
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


def extract_metadata_records(
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
                source_file_rel_path=coerce_required_str(entity, "source_file_rel_path"),
                entity_id=coerce_required_str(entity, "canonical_id"),
                payload=payload,
            )
        )
    return records


def build_node_records(
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
            label = (
                title
                if isinstance(title, str)
                else meta_id
                if isinstance(meta_id, str)
                else source.file_rel_path
            )
            metadata.update(
                {
                    "meta_id": meta_id,
                    "purpose": source_meta.get("purpose"),
                    "domain": source_meta.get("domain"),
                    "domain_type": source_meta.get("domain_type"),
                }
            )
        nodes[file_node_id(source.file_rel_path)] = OntologyNodeRecord(
            node_id=file_node_id(source.file_rel_path),
            node_type="ontology_file",
            source_file_rel_path=source.file_rel_path,
            entity_id=None,
            label=label or source.file_rel_path,
            metadata=metadata,
        )
    for entity in entity_definitions:
        canonical_id = coerce_required_str(entity, "canonical_id")
        label = entity.get("label")
        nodes[canonical_id] = OntologyNodeRecord(
            node_id=canonical_id,
            node_type="entity",
            source_file_rel_path=coerce_required_str(entity, "source_file_rel_path"),
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


def coerce_required_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing required string field: {key}")
    return value


def file_node_id(file_rel_path: str) -> str:
    return f"file:{file_rel_path}"


def external_file_node_id(target: str) -> str:
    return f"external_file:{target}"


def unresolved_entity_node_id(entity_name: str) -> str:
    return f"unresolved_entity:{entity_name}"


def taxonomy_value_node_id(taxonomy_id: str, dimension: str, value: str) -> str:
    return f"taxonomy_value:{taxonomy_id}:{dimension}:{value}"


def taxonomy_type_node_id(taxonomy_id: str, value: str) -> str:
    return f"taxonomy_type:{taxonomy_id}:{value}"


def taxonomy_reference_node_id(reference_name: str) -> str:
    return f"taxonomy_reference:{reference_name}"
