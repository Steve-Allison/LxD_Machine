"""Tests for the per-thread SQLite connection pool."""

import threading
from collections.abc import Generator
from pathlib import Path

import pytest

from lxd.stores.sqlite._pool import pooled_connection, reset_pool
from lxd.stores.sqlite.connection import build_store_paths


@pytest.fixture(autouse=True)
def _reset_pool_around_each_test() -> Generator[None]:
    """Ensure each test sees a fresh pool — order-of-test independence."""
    reset_pool()
    yield
    reset_pool()


def test_pooled_connection_reuses_connection_across_calls_in_same_thread(
    tmp_path: Path,
) -> None:
    """100 sequential calls in one thread share a single underlying connection."""
    store_paths = build_store_paths(tmp_path)

    seen_ids: set[int] = set()
    for _ in range(100):
        with pooled_connection(store_paths.sqlite_path) as connection:
            seen_ids.add(id(connection))

    assert len(seen_ids) == 1, (
        f"Expected one connection identity across 100 calls, saw {len(seen_ids)}."
    )


def test_pooled_connection_does_not_share_across_threads(tmp_path: Path) -> None:
    """Different threads must get distinct connection objects (sqlite check_same_thread)."""
    store_paths = build_store_paths(tmp_path)

    seen_ids: list[int] = []
    barrier = threading.Barrier(3)
    lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        with pooled_connection(store_paths.sqlite_path) as connection, lock:
            seen_ids.append(id(connection))

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(seen_ids) == 3
    assert len(set(seen_ids)) == 3, (
        "Expected three distinct connection objects across three threads, "
        f"saw {len(set(seen_ids))} distinct."
    )


def test_pooled_connection_rolls_back_open_transaction_on_exception(
    tmp_path: Path,
) -> None:
    """An exception inside the with-block rolls back any in-flight transaction."""
    store_paths = build_store_paths(tmp_path)

    # Prime the schema and insert a known row through a fresh pool entry.
    with pooled_connection(store_paths.sqlite_path) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS pool_test (k TEXT PRIMARY KEY);")
        connection.execute("INSERT INTO pool_test (k) VALUES ('committed');")
        connection.commit()

    # Now: start a write inside a with-block that raises before commit.
    with pytest.raises(RuntimeError), pooled_connection(store_paths.sqlite_path) as connection:
        connection.execute("INSERT INTO pool_test (k) VALUES ('rolled_back');")
        raise RuntimeError("simulated tool failure")

    # The pooled connection survived; the rolled-back row is absent.
    with pooled_connection(store_paths.sqlite_path) as connection:
        rows = {row[0] for row in connection.execute("SELECT k FROM pool_test;")}
    assert rows == {"committed"}, (
        f"Pool should have rolled back the in-flight insert; saw rows={rows}."
    )
