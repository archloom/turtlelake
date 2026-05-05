"""Multi-process write contract.

The pitch ('open a directory, no daemon, no network') leaves an open
question: what happens when two processes open the same directory and
both write? These tests pin the contract:

1. Concurrent appends from spawn-mode workers all succeed; Lance's
   optimistic concurrency on the manifest serializes the writes into a
   linear version history, no rows lost.
2. The fork start method is *not* safe with Lance — Lance prints a
   warning and behavior is undefined. Tests use spawn explicitly.
3. A reader opening the dataset while another process appends sees
   either the pre-write or post-write state, never a partial fragment.
4. compact() can run concurrently with readers; it produces a new
   version that subsequent opens see, while in-flight opens stay
   pinned to whatever version they grabbed.

These tests are slow-ish (process spawn overhead) but bounded —
each runs a few small workers and joins with a timeout.
"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import pytest

from turtlelake import Dataset

# Required: spawn (default on macOS, opt-in on Linux). Lance is
# explicitly not fork-safe — see the UserWarning emitted by `lance`
# at import time when running under fork.
SPAWN = mp.get_context("spawn")


def _seed(path: Path) -> None:
    """Bootstrap a dataset in the parent process before fanning out
    workers. Each worker appends; nobody starts from empty."""
    ds = Dataset.open(path)
    from pyoxigraph import Literal, NamedNode, Quad

    pred = NamedNode("http://www.w3.org/2000/01/rdf-schema#label")
    ds._append_quads(
        [Quad(NamedNode("https://ex/seed"), pred, Literal("seed"))],
        batch_size=10,
    )


def _drain(queue) -> list[tuple]:
    out = []
    while not queue.empty():
        out.append(queue.get_nowait())
    return out


def test_two_writers_both_succeed_and_no_rows_lost(tmp_path):
    """Append 50 quads from each of 2 processes. Final count must
    equal seed (1) + 2 × 50 = 101. Lance serializes via optimistic
    concurrency; one writer may see a retry inside Lance, but neither
    fails."""
    from tests._concurrent_helpers import append_triples

    path = tmp_path / "kg"
    _seed(path)
    queue = SPAWN.Queue()
    procs = [
        SPAWN.Process(target=append_triples, args=(str(path), f"w{i}", 50, queue))
        for i in range(2)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0, f"worker exited {p.exitcode}"

    results = _drain(queue)
    assert len(results) == 2
    assert all(r[1] == "ok" for r in results), results

    ds = Dataset.open(path)
    assert ds.count() == 1 + 2 * 50


def test_concurrent_embedding_writes_succeed(tmp_path):
    """Same contract for the embeddings dataset. Two processes write
    different IRIs; both must persist."""
    from tests._concurrent_helpers import append_embeddings

    path = tmp_path / "kg"
    _seed(path)
    queue = SPAWN.Queue()
    procs = [
        SPAWN.Process(target=append_embeddings, args=(str(path), f"w{i}", 30, 4, queue))
        for i in range(2)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0

    results = _drain(queue)
    oks = [r for r in results if r[1] == "ok"]
    assert len(oks) == 2, f"expected both writers to succeed; got {results}"

    ds = Dataset.open(path)
    assert ds.embedding_count() == 2 * 30


def test_reader_during_write_sees_a_consistent_snapshot(tmp_path):
    """Open + count from one process while another is appending. The
    reader must see either the pre- or post-write count, never a
    partial fragment / torn manifest."""
    from tests._concurrent_helpers import append_triples, read_count

    path = tmp_path / "kg"
    _seed(path)
    queue = SPAWN.Queue()

    writer = SPAWN.Process(target=append_triples, args=(str(path), "wA", 200, queue))
    reader = SPAWN.Process(target=read_count, args=(str(path), queue))
    writer.start()
    reader.start()
    for p in (writer, reader):
        p.join(timeout=60)
        assert p.exitcode == 0

    results = _drain(queue)
    by_label = {r[0]: r for r in results}
    assert by_label["reader"][1] == "ok"
    # The reader's count is pre-write (just the seed) or post-write
    # (1 + 200) — both are valid linearizations.
    reader_count = int(by_label["reader"][2])
    assert reader_count in (1, 1 + 200)


@pytest.mark.parametrize("n_writers", [3])
def test_n_concurrent_writers_no_lost_updates(tmp_path, n_writers):
    """N>2 stress check. Even with 3 writers contending, every row
    written must land in the final state."""
    from tests._concurrent_helpers import append_triples

    path = tmp_path / "kg"
    _seed(path)
    queue = SPAWN.Queue()
    rows_each = 30
    procs = [
        SPAWN.Process(
            target=append_triples, args=(str(path), f"w{i}", rows_each, queue)
        )
        for i in range(n_writers)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)
        assert p.exitcode == 0

    results = _drain(queue)
    assert all(r[1] == "ok" for r in results), results

    ds = Dataset.open(path)
    assert ds.count() == 1 + n_writers * rows_each


def test_compact_concurrent_with_reader_does_not_corrupt(tmp_path):
    """compact() rewrites fragments on a new Lance version. A reader
    opened before the compact must continue to read its own version
    without error (Lance keeps the old fragments live until cleanup).
    """
    from tests._concurrent_helpers import append_triples, compact_dataset

    path = tmp_path / "kg"
    _seed(path)

    # Build up a few fragments so compact has work to do.
    queue = SPAWN.Queue()
    for i in range(3):
        p = SPAWN.Process(
            target=append_triples, args=(str(path), f"setup{i}", 20, queue)
        )
        p.start()
        p.join(timeout=60)
        assert p.exitcode == 0
    _drain(queue)

    # Now race a compactor against a parent-process reader.
    compactor = SPAWN.Process(target=compact_dataset, args=(str(path), queue))
    compactor.start()

    # Parent reads its own handle while compaction runs.
    ds = Dataset.open(path)
    rows_before_join = ds.count()
    assert rows_before_join >= 1 + 3 * 20

    compactor.join(timeout=60)
    assert compactor.exitcode == 0

    # After compaction, a fresh open still sees every row.
    fresh = Dataset.open(path)
    assert fresh.count() == rows_before_join
