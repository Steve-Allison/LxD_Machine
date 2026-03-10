from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx

from lxd.domain.ids import make_graph_edge_key


@dataclass(frozen=True)
class OntologyNodeRecord:
    node_id: str
    node_type: str
    source_file_rel_path: str | None
    entity_id: str | None
    label: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RelationRecord:
    relation_type: str
    origin_kind: str
    origin_path: str
    source_file_rel_path: str
    source_node_id: str
    source_node_type: str
    source_entity_id: str | None
    target_node_id: str
    target_node_type: str
    target_entity_id: str | None
    target_file_rel_path: str | None
    metadata: dict[str, Any]


OntologyGraph = nx.MultiDiGraph


def build_graph(
    nodes: list[OntologyNodeRecord],
    relations: list[RelationRecord],
) -> OntologyGraph:
    graph = nx.MultiDiGraph()
    for node in nodes:
        graph.add_node(
            node.node_id,
            node_type=node.node_type,
            source_file_rel_path=node.source_file_rel_path,
            entity_id=node.entity_id,
            label=node.label,
            metadata=node.metadata,
        )
    for relation in relations:
        key = make_graph_edge_key(
            relation.origin_kind,
            relation.source_file_rel_path,
            relation.source_node_id,
            relation.relation_type,
            relation.target_node_id,
        )
        if relation.source_node_id not in graph:
            graph.add_node(
                relation.source_node_id,
                node_type=relation.source_node_type,
                source_file_rel_path=relation.source_file_rel_path,
                entity_id=relation.source_entity_id,
                label=relation.source_node_id,
                metadata={},
            )
        if relation.target_node_id not in graph:
            graph.add_node(
                relation.target_node_id,
                node_type=relation.target_node_type,
                source_file_rel_path=relation.target_file_rel_path,
                entity_id=relation.target_entity_id,
                label=relation.target_node_id,
                metadata=dict(relation.metadata),
            )
        graph.add_edge(
            relation.source_node_id,
            relation.target_node_id,
            key=key,
            relation_type=relation.relation_type,
            origin_kind=relation.origin_kind,
            origin_path=relation.origin_path,
            source_file_rel_path=relation.source_file_rel_path,
            source_node_id=relation.source_node_id,
            source_node_type=relation.source_node_type,
            source_entity_id=relation.source_entity_id,
            target_node_id=relation.target_node_id,
            target_node_type=relation.target_node_type,
            target_entity_id=relation.target_entity_id,
            target_file_rel_path=relation.target_file_rel_path,
            relation_metadata=relation.metadata,
        )
    return graph


def direct_neighbors(graph: OntologyGraph, entity_id: str) -> list[dict[str, Any]]:
    neighbors: list[dict[str, Any]] = []
    for _, target, key, data in graph.out_edges(entity_id, keys=True, data=True):
        neighbor_attrs = graph.nodes[target]
        record = {
            "direction": "out",
            "edge_key": key,
            "neighbor_node_id": target,
            "neighbor_node_type": neighbor_attrs.get("node_type"),
            "neighbor_entity_id": neighbor_attrs.get("entity_id"),
            "neighbor_label": neighbor_attrs.get("label"),
            "neighbor_metadata": neighbor_attrs.get("metadata"),
        }
        record.update(data)
        neighbors.append(record)
    for source, _, key, data in graph.in_edges(entity_id, keys=True, data=True):
        neighbor_attrs = graph.nodes[source]
        record = {
            "direction": "in",
            "edge_key": key,
            "neighbor_node_id": source,
            "neighbor_node_type": neighbor_attrs.get("node_type"),
            "neighbor_entity_id": neighbor_attrs.get("entity_id"),
            "neighbor_label": neighbor_attrs.get("label"),
            "neighbor_metadata": neighbor_attrs.get("metadata"),
        }
        record.update(data)
        neighbors.append(record)
    return neighbors
