"""Tests for tenancy and Central_Configs v5.0.0 DuckDB configuration validation."""

from pathlib import Path

import pytest

from lxd.ontology.loader import load_configured_ontology, load_ontology
from lxd.settings.loader import load_runtime_config, resolve_repo_root
from lxd.settings.models import PathsConfig, TenancyConfig


def test_tenancy_config_defaults_to_single_tenant() -> None:
    """Omitting corpus_id yields the default tenant."""
    config = TenancyConfig()
    assert config.corpus_id == "default"


@pytest.mark.parametrize(
    "invalid",
    [
        "",
        "-leading-dash",
        "UPPERCASE",
        "has spaces",
        "a" * 64,
        "bad chars!",
        "has.dot",
    ],
)
def test_tenancy_config_rejects_invalid_ids(invalid: str) -> None:
    with pytest.raises(ValueError):
        TenancyConfig(corpus_id=invalid)


def test_runtime_config_loads_with_default_tenancy_and_central_configs_path() -> None:
    """Default config resolves tenancy and Central_Configs v5.0.0 DuckDB path."""
    repo_root = resolve_repo_root()
    config, _ = load_runtime_config(repo_root)
    assert config.tenancy.corpus_id == "default"
    assert config.paths.central_configs_path == (repo_root / ".." / "Central_Configs").resolve()
    assert config.paths.central_duckdb_path == (
        config.paths.central_configs_path / "dist" / "build" / "central.duckdb"
    )


def test_paths_config_central_duckdb_path_accepts_direct_duckdb_file(tmp_path: Path) -> None:
    db_file = tmp_path / "custom.duckdb"
    db_file.write_bytes(b"placeholder")
    paths = PathsConfig(
        corpus_path=tmp_path / "corpus",
        central_configs_path=db_file,
        data_path=tmp_path / "data",
    )
    assert paths.central_duckdb_path == db_file


def test_load_ontology_raises_when_duckdb_store_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"central\.duckdb"):
        load_ontology(tmp_path / "missing-central")


def test_load_configured_ontology_loads_live_central_configs() -> None:
    config, _ = load_runtime_config(resolve_repo_root())
    if not config.paths.central_duckdb_path.is_file():
        pytest.skip("Central_Configs v5.0.0 central.duckdb not built locally")
    loaded = load_configured_ontology(config)
    assert len(loaded.entity_definitions) >= 564
    assert all(e["canonical_id"].startswith("coe:C") for e in loaded.entity_definitions)
