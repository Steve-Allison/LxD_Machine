"""YAML source loading, hashing, and coverage for ontology files."""

import json
from collections import defaultdict
from operator import attrgetter
from pathlib import Path
from typing import Any

import yaml
from blake3 import blake3

from lxd.domain.ids import blake3_hex
from lxd.ontology.inventory import OntologyCoverageReport, build_coverage_report, discover_key_paths
from lxd.ontology.loader.types import OntologySource


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


def load_sources(
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


def snapshot_hash(sources: list[OntologySource]) -> str:
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
        for source in sorted(sources, key=attrgetter("file_rel_path"))
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


def coverage_report_for_sources(sources: list[OntologySource]) -> OntologyCoverageReport:
    path_counts: dict[str, int] = defaultdict(int)
    for source in sources:
        discovered = discover_key_paths(source.data)
        for path, count in discovered.items():
            path_counts[path] += count
    return build_coverage_report(path_counts)
