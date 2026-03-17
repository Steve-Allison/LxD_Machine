from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from lxd.observability.logging import configure_logging
from lxd.settings.loader import load_runtime_config, resolve_repo_root
from lxd.settings.models import RuntimeConfig


@dataclass(frozen=True)
class AppContext:
    repo_root: Path
    config: RuntimeConfig
    config_path: Path


def bootstrap_app(
    cwd: Path | None = None,
    *,
    profile: str | None = None,
    config_path: Path | None = None,
) -> AppContext:
    repo_root = resolve_repo_root(cwd)
    load_dotenv(repo_root / ".env", override=False)
    config, resolved_config_path = load_runtime_config(
        repo_root,
        profile=profile,
        config_path=config_path,
    )
    configure_logging(config.logging.level, config.logging.format)
    return AppContext(repo_root=repo_root, config=config, config_path=resolved_config_path)
