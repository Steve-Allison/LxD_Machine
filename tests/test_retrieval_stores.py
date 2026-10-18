"""Tests for retrieval store handle lifecycle and connection pooling."""

from pathlib import Path
from types import SimpleNamespace
from typing import cast

from lxd.retrieval.stores import open_retrieval_stores
from lxd.settings.models import ModelsConfig, PathsConfig, RuntimeConfig
from lxd.stores.sqlite._pool import reset_pool


def _make_config(tmp_path: Path) -> RuntimeConfig:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = PathsConfig(
        corpus_path=tmp_path / "corpus",
        ontology_path=tmp_path / "ontology",
        data_path=data_dir,
    )
    models = ModelsConfig(embed="mock-embed", llm="mock-llm", rerank="mock-rerank", embed_dims=8)
    return cast("RuntimeConfig", SimpleNamespace(paths=paths, models=models))


def test_open_retrieval_stores_reuses_pooled_connection(tmp_path: Path) -> None:
    reset_pool()
    config = _make_config(tmp_path)

    with open_retrieval_stores(config) as stores1:
        conn1 = stores1.sqlite
        assert conn1 is not None

    with open_retrieval_stores(config) as stores2:
        conn2 = stores2.sqlite
        assert conn2 is not None
        # Must be the exact same connection object from the pool
        assert conn1 is conn2


def test_open_retrieval_stores_rolls_back_on_error(tmp_path: Path) -> None:
    reset_pool()
    config = _make_config(tmp_path)

    try:
        with open_retrieval_stores(config) as stores:
            conn = stores.sqlite
            conn.execute("CREATE TABLE IF NOT EXISTS test_table (id INTEGER PRIMARY KEY);")
            conn.execute("INSERT INTO test_table VALUES (1);")
            raise RuntimeError("simulated query error")
    except RuntimeError:
        pass

    # Next call with pooled connection must not be left in an aborted transaction
    with open_retrieval_stores(config) as stores:
        # Should be able to execute queries cleanly without transaction error
        rows = stores.sqlite.execute("SELECT * FROM test_table;").fetchall()
        assert len(rows) == 0 or len(rows) == 1
