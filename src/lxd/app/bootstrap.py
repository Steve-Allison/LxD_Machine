"""Bootstrap application dependencies and runtime services."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from lxd.observability.logging import configure_logging
from lxd.settings.loader import load_runtime_config, resolve_repo_root
from lxd.settings.models import RuntimeConfig


@dataclass(frozen=True)
class AppContext:
    """Hold resolved runtime context for CLI and MCP entrypoints.

    Attributes:
        repo_root: Repository root discovered from the working directory.
        config: Validated runtime configuration.
        config_path: Absolute path to the config file used to build `config`.
    """
    repo_root: Path
    config: RuntimeConfig
    config_path: Path


def bootstrap_app(
    cwd: Path | None = None,
    *,
    profile: str | None = None,
    config_path: Path | None = None,
) -> AppContext:
    """Resolve runtime config and initialize process-wide logging.

    Args:
        cwd: Starting directory used when resolving the repository root.
        profile: Optional profile name that maps to `config.<profile>.yaml`.
        config_path: Optional explicit path to a runtime config file.

    Returns:
        Immutable application context containing repo root and validated config.

    Raises:
        FileNotFoundError: If repo root or config file cannot be resolved.
        ValueError: If both `profile` and `config_path` are provided, or config validation fails.

    Side Effects:
        Reads `.env` and runtime config files from disk; configures global logging.
    """
    repo_root = resolve_repo_root(cwd)
    load_dotenv(repo_root / ".env", override=False)
    config, resolved_config_path = load_runtime_config(
        repo_root,
        profile=profile,
        config_path=config_path,
    )
    configure_logging(config.logging.level, config.logging.format)
    return AppContext(repo_root=repo_root, config=config, config_path=resolved_config_path)
