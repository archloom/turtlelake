"""Regression tests for the next-wave speed-ups:

  - Incremental cache update on insert / insert_turtle (no rebuild)
  - pre_warm=True forces materialization on open

Every test must verify *correctness first*, performance second.
"""

import time

from pyoxigraph import Literal, NamedNode, Quad

from turtlelake import Dataset


# ── Incremental cache updates ────────────────────────────────


def test_insert_turtle_extends_warm_cache_in_place(tmp_path):
    """If the cache is warm, insert_turtle should push the new quad to it,
    NOT invalidate. Verifiable because next query sees the new fact
    without a big re-materialization latency spike."""
    kg = Dataset.open(tmp_path / "kg")
    # Seed 5k triples — enough that a rebuild is measurable.
    lines = ['@prefix ex: <https://ex.org/> .']
    for i in range(5000):
        lines.append(f'ex:d{i} ex:k "v{i}" .')
    ttl_seed = tmp_path / "seed.ttl"
    ttl_seed.write_text("\n".join(lines), encoding="utf-8")
    kg.ingest_ttl(ttl_seed)
    kg.query("SELECT ?s WHERE { ?s ?p ?o } LIMIT 1")  # warm the cache

    t0 = time.perf_counter()
    kg.insert_turtle('<https://ex.org/new> <https://ex.org/p> "x" .')
    rows = kg.query(
        "SELECT ?o WHERE { <https://ex.org/new> ?p ?o }"
    )
    elapsed = time.perf_counter() - t0

    # Correctness: the new fact IS visible.
    assert len(rows) == 1
    assert rows[0]["o"]["value"] == "x"
    # Performance: one insert + query on a warm cache over 5k triples
    # should be under 200 ms. Before the fix, this was ~150-200 ms of
    # rebuild. This test catches a regression that reintroduces invalidation.
    assert elapsed < 0.5, (
        f"insert+read on warm cache took {elapsed:.3f}s; "
        "likely regressed to full rebuild"
    )


def test_insert_quads_also_extends_cache(tmp_path):
    kg = Dataset.open(tmp_path / "kg")
    kg.insert_turtle('<https://ex.org/a> <https://ex.org/p> "1" .')
    kg.query("ASK { ?s ?p ?o }")  # warm

    kg.insert(
        [Quad(
            NamedNode("https://ex.org/b"),
            NamedNode("https://ex.org/p"),
            Literal("2"),
        )]
    )
    rows = kg.query("SELECT ?o WHERE { ?s ?p ?o } ORDER BY ?o")
    assert [r["o"]["value"] for r in rows] == ["1", "2"]


def test_insert_before_any_query_does_not_spuriously_warm(tmp_path):
    """If no cache existed before the insert, we don't create one —
    the next read path materializes lazily as usual."""
    kg = Dataset.open(tmp_path / "kg")
    kg.insert_turtle('<https://ex.org/a> <https://ex.org/p> "1" .')
    # _cached_engine must still be None until a read triggers it.
    assert kg._cached_engine is None


# ── pre_warm=True ────────────────────────────────────────────


def test_pre_warm_open_populates_cache(tmp_path):
    # First populate the dataset
    warm = Dataset.open(tmp_path / "kg")
    warm.insert_turtle(
        '<https://ex.org/a> <https://ex.org/p> "v" .'
    )

    # Re-open with pre_warm=True → the cache should be populated before
    # any read.
    pre = Dataset.open(tmp_path / "kg", pre_warm=True)
    assert pre._cached_engine is not None
    assert pre._cached_engine_version == pre._lance.version


def test_pre_warm_on_empty_dataset_is_noop(tmp_path):
    """pre_warm should not crash when the dataset has no data yet."""
    kg = Dataset.open(tmp_path / "kg", pre_warm=True)
    assert kg._cached_engine is None  # nothing to warm


def test_pre_warm_first_query_latency_is_small(tmp_path):
    # Seed 5k triples for a measurable materialization cost.
    lines = ['@prefix ex: <https://ex.org/> .']
    for i in range(5000):
        lines.append(f'ex:d{i} ex:k "v{i}" .')
    ttl_seed = tmp_path / "seed.ttl"
    ttl_seed.write_text("\n".join(lines), encoding="utf-8")
    Dataset.open(tmp_path / "kg").ingest_ttl(ttl_seed)

    pre = Dataset.open(tmp_path / "kg", pre_warm=True)
    t0 = time.perf_counter()
    pre.query("SELECT ?s WHERE { ?s ?p ?o } LIMIT 1")
    elapsed = time.perf_counter() - t0
    # The materialization happened during open(); the first query
    # should be at steady-state latency.
    assert elapsed < 0.05, f"first query after pre_warm took {elapsed:.3f}s"


# ── Sanity: rollback still invalidates correctly ─────────────


def test_rollback_invalidates_cache(tmp_path):
    """Regression: rollback must not leave a stale cache behind even
    though the cache is now actively extended on inserts."""
    kg = Dataset.open(tmp_path / "kg")
    kg.insert_turtle('<https://ex.org/a> <https://ex.org/p> "v1" .')
    kg.checkpoint("pre")
    kg.query("ASK { ?s ?p ?o }")  # warm with v1 visible
    kg.insert_turtle('<https://ex.org/a> <https://ex.org/p> "v2" .')  # extends cache
    kg.rollback("pre")
    # Cache must reflect the rolled-back state — no "v2"
    objs = {r["o"]["value"] for r in kg.query("SELECT ?o WHERE { ?s ?p ?o }")}
    assert objs == {"v1"}


# ── Sanity: ingest_ttl still invalidates (bulk path) ─────────


def test_ingest_ttl_still_triggers_rebuild_on_next_query(tmp_path):
    """Bulk ingests don't incrementally extend (they could hold gigabytes
    in memory). They still invalidate so the next query rebuilds from
    the fresh Lance version. This test just pins that contract."""
    kg = Dataset.open(tmp_path / "kg")
    ttl1 = tmp_path / "a.ttl"
    ttl1.write_text('<https://ex.org/a> <https://ex.org/p> "1" .', encoding="utf-8")
    kg.ingest_ttl(ttl1)
    kg.query("ASK { ?s ?p ?o }")  # warm

    ttl2 = tmp_path / "b.ttl"
    ttl2.write_text('<https://ex.org/b> <https://ex.org/p> "2" .', encoding="utf-8")
    kg.ingest_ttl(ttl2)
    # After the bulk ingest, the cache version is stale; next query triggers
    # a rebuild. Correctness: both quads visible.
    rows = kg.query("SELECT ?o WHERE { ?s ?p ?o } ORDER BY ?o")
    values = sorted(r["o"]["value"] for r in rows)
    assert values == ["1", "2"]
