from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from lxd.settings.models import RuntimeConfig


def resolve_repo_root(cwd: Path | None = None) -> Path:
    current = (cwd or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pixi.toml").exists() and (candidate / "Plans").exists():
            return candidate
    raise FileNotFoundError("Could not locate repo root containing pixi.toml and Plans/")


def load_runtime_config(
    repo_root: Path,
    *,
    profile: str | None = None,
    config_path: Path | None = None,
) -> tuple[RuntimeConfig, Path]:
    resolved_config_path = _resolve_config_path(
        repo_root=repo_root,
        profile=profile,
        config_path=config_path,
    )
    raw = _load_yaml(resolved_config_path)
    _resolve_paths_section(raw, base_dir=resolved_config_path.parent)
    _resolve_reranker_section(raw, base_dir=resolved_config_path.parent)
    return RuntimeConfig.model_validate(raw), resolved_config_path


def _resolve_config_path(
    *,
    repo_root: Path,
    profile: str | None,
    config_path: Path | None,
) -> Path:
    if profile is not None and config_path is not None:
        raise ValueError("Specify either profile or config_path, not both.")
    if config_path is not None:
        resolved = config_path.expanduser()
        if not resolved.is_absolute():
            resolved = (repo_root / resolved).resolve()
        return _require_existing_config(resolved)
    if profile is not None:
        return _require_existing_config(repo_root / f"config.{profile}.yaml")
    return _require_existing_config(repo_root / "config.yaml")


def _require_existing_config(config_path: Path) -> Path:
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing runtime config file: {config_path}")
    return config_path


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping at top level of {path}")
    return loaded


def _resolve_paths_section(raw: dict[str, Any], *, base_dir: Path) -> None:
    paths = raw.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("Runtime config must define a top-level 'paths' mapping.")
    for key in ("corpus_path", "ontology_path", "data_path"):
        value = paths.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Runtime config paths.{key} must be a non-empty string.")
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = (base_dir / candidate).resolve()
        paths[key] = candidate


def _resolve_reranker_section(raw: dict[str, Any], *, base_dir: Path) -> None:
    reranker = raw.get("reranker")
    if not isinstance(reranker, dict):
        return
    launch = reranker.get("launch")
    if not isinstance(launch, dict):
        return
    model_path = launch.get("model_path")
    if model_path is None:
        return
    if not isinstance(model_path, str) or not model_path.strip():
        raise ValueError("Runtime config reranker.launch.model_path must be a non-empty string.")
    candidate = Path(model_path).expanduser()
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    launch["model_path"] = candidate
