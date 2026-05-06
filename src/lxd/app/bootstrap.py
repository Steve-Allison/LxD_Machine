"""Bootstrap application dependencies and runtime services."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import structlog
from dotenv import load_dotenv

from lxd.domain.ids import blake3_hex
from lxd.observability.logging import configure_logging
from lxd.settings.loader import load_runtime_config, resolve_repo_root
from lxd.settings.models import RuntimeConfig

_log = structlog.get_logger(__name__)

_CONFIG_LOCK_FILENAME = "config.lock"


@dataclass(frozen=True, slots=True)
class AppContext:
    """Hold resolved runtime context for CLI and MCP entrypoints.

    Attributes:
        repo_root: Repository root discovered from the working directory.
        config: Validated runtime configuration.
        config_path: Absolute path to the config file used to build `config`.
        config_digest: Blake3 digest of the resolved config, used to detect drift.
    """

    repo_root: Path
    config: RuntimeConfig
    config_path: Path
    config_digest: str


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
    configure_logging(
        config.logging.level,
        config.logging.format,
        sample_rate=config.logging.sample_rate,
        sampled_event_names=frozenset(config.logging.sampled_event_names),
    )
    digest = compute_config_digest(config)
    reconcile_config_lock(config.paths.data_path, digest=digest)
    return AppContext(
        repo_root=repo_root,
        config=config,
        config_path=resolved_config_path,
        config_digest=digest,
    )


def compute_config_digest(config: RuntimeConfig) -> str:
    """Return a stable Blake3 digest of the resolved configuration.

    The digest is derived from the JSON-normalised Pydantic dump with
    sorted keys. Path values are serialised as strings via
    ``model_dump(mode="json")`` so the digest stays stable across
    platforms.

    Args:
        config: Fully resolved runtime configuration.

    Returns:
        Hex-encoded Blake3 digest suitable for comparison or logging.
    """
    payload = json.dumps(
        config.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return blake3_hex(payload)


def reconcile_config_lock(data_path: Path, *, digest: str) -> None:
    """Compare ``digest`` against ``<data_path>/config.lock`` and log drift.

    Behaviour:
        * If ``data_path`` does not yet exist, no lock is written (first-run
          ingestion is expected to create and seed the directory).
        * If the lock file is missing, it is created with the current digest.
        * If the stored digest differs, a ``config.lock.mismatch`` warning is
          logged. The caller remains responsible for deciding whether a
          mismatch is fatal (migrations, ingestion, etc.).

    Args:
        data_path: Directory that owns runtime state (SQLite, LanceDB, lock).
        digest: Digest produced by :func:`compute_config_digest`.

    Side Effects:
        Creates or reads ``<data_path>/config.lock`` and may emit warnings.
    """
    if not data_path.exists():
        return
    lock_path = data_path / _CONFIG_LOCK_FILENAME
    if not lock_path.exists():
        lock_path.write_text(digest + "\n", encoding="utf-8")
        _log.info("config.lock.initialised", path=str(lock_path))
        return
    stored = lock_path.read_text(encoding="utf-8").strip()
    if stored != digest:
        _log.warning(
            "config.lock.mismatch",
            path=str(lock_path),
            stored_digest=stored,
            current_digest=digest,
        )
