"""Tests for tenancy + ontology-schema config validation.

Covers:
    * :class:`TenancyConfig` validation (corpus_id shape).
    * :class:`OntologyFileModel` validation on representative payloads.
"""

import pytest

from lxd.ontology.schema_models import OntologyFileModel, validate_ontology_file
from lxd.settings.loader import load_runtime_config, resolve_repo_root
from lxd.settings.models import TenancyConfig


def test_tenancy_config_defaults_to_single_tenant() -> None:
    """Omitting ``corpus_id`` yields the ``default`` tenant."""
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


def test_runtime_config_loads_with_default_tenancy() -> None:
    """Configs without an explicit `tenancy:` block load with the default tenant."""
    config, _ = load_runtime_config(resolve_repo_root())
    assert config.tenancy.corpus_id == "default"


def test_validate_ontology_file_accepts_minimal_payload() -> None:
    """A simple entity-types payload validates without errors."""
    payload = {
        "_meta": {"id": "demo", "title": "Demo"},
        "entity_types": {
            "ENT_A": {"label": "A", "entity_kind": "concept"},
        },
    }
    model = validate_ontology_file(payload)
    assert isinstance(model, OntologyFileModel)
    assert model.meta is not None
    assert model.meta.id == "demo"
    assert model.entity_types is not None
    assert "ENT_A" in model.entity_types


def test_validate_ontology_file_rejects_non_mapping() -> None:
    with pytest.raises(ValueError, match="mapping"):
        validate_ontology_file(["not", "a", "dict"])


def test_validate_ontology_file_allows_unknown_top_level_keys() -> None:
    """Extras are permitted because ontology shapes evolve quickly."""
    payload = {"unknown_section": {"foo": "bar"}}
    model = validate_ontology_file(payload)
    assert model.entity_types is None
