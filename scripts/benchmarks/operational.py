"""Operational benchmarks — capabilities competitors structurally
cannot replicate at this exact intersection (embedded + graph +
vectors + atomic versioning).

Each benchmark answers a yes/no or numeric question that's irrelevant
to a pure-vector or pure-graph system but central to turtlelake's
positioning. Numbers are intentionally simple: open() latency in
milliseconds, "yes the rollback was atomic" as a pass/fail, etc.

Where /dev/shm exists (Linux tmpfs, RAM-backed), we use it as the
default benchmark root so disk I/O isn't a confound. Override with
TURTLELAKE_BENCHMARK_ROOT if you want to test on a real disk.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from pyoxigraph import Literal, NamedNode, Quad

from turtlelake import Dataset


def _bench_root() -> Path:
    """Pick a base directory for benchmark datasets. Prefers /dev/shm
    on Linux (RAM-backed tmpfs) so I/O latency doesn't dominate the
    operational measurements; the tests really exercise CPU + memory
    paths."""
    override = os.environ.get("TURTLELAKE_BENCHMARK_ROOT")
    if override:
        return Path(override)
    shm = Path("/dev/shm")
    if shm.exists() and os.access(shm, os.W_OK):
        return shm
    return Path(tempfile.gettempdir())


# ── 1. open-to-first-query latency ──────────────────────────────────


def bench_open_to_first_query(rows: int = 1_000) -> dict:
    """How long from `Dataset.open(path)` to the first non-trivial
    query returning. The "embedded, no daemon" pitch is a latency
    promise; this puts a number on it."""
    root = _bench_root() / f"tlk-bench-open-{os.getpid()}"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    try:
        # Build a small dataset once.
        ds = Dataset.open(root)
        label = NamedNode("http://www.w3.org/2000/01/rdf-schema#label")
        quads = [
            Quad(NamedNode(f"https://ex/{i}"), label, Literal(f"e{i}"))
            for i in range(rows)
        ]
        ds._append_quads(quads, batch_size=10_000)
        del ds

        # Cold open + first query (entity).
        t0 = time.perf_counter()
        kg = Dataset.open(root)
        # `entity` is the canonical agent first call. SPARQL would
        # also work but `entity` doesn't pay SPARQL's parser cost.
        kg.entity("https://ex/0")
        cold_ms = (time.perf_counter() - t0) * 1000

        # Warm open (process state cached, OS page cache hot).
        warm_ms = []
        for _ in range(5):
            t0 = time.perf_counter()
            kg2 = Dataset.open(root)
            kg2.entity("https://ex/0")
            warm_ms.append((time.perf_counter() - t0) * 1000)

        return {
            "rows": rows,
            "cold_open_to_first_query_ms": round(cold_ms, 2),
            "warm_open_to_first_query_ms": round(sum(warm_ms) / len(warm_ms), 2),
            "warm_runs": len(warm_ms),
        }
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ── 2. Update-visibility latency ────────────────────────────────────


def bench_update_visibility() -> dict:
    """Time from `insert_turtle(...)` returning to the new triple being
    visible in `query(...)`. In a two-store stack this includes a
    sync hop; embedded turtlelake should be sub-millisecond."""
    root = _bench_root() / f"tlk-bench-vis-{os.getpid()}"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    try:
        ds = Dataset.open(root)
        ds.insert_turtle(
            "@prefix ex: <https://ex/> . ex:bootstrap ex:p ex:o ."
        )

        latencies: list[float] = []
        for i in range(20):
            iri = f"https://ex/marker{i}"
            t0 = time.perf_counter()
            ds.insert_turtle(f"<{iri}> <https://ex/p> <https://ex/o> .")
            # Query for it.
            rows = ds.query(
                "SELECT ?o WHERE { <%s> <https://ex/p> ?o }" % iri
            )
            latencies.append((time.perf_counter() - t0) * 1000)
            assert rows, "newly-inserted triple not visible"

        return {
            "writes": len(latencies),
            "median_ms": round(sorted(latencies)[len(latencies) // 2], 3),
            "p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 3),
            "max_ms": round(max(latencies), 3),
        }
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ── 3. Reproducibility @ tag ────────────────────────────────────────


def bench_reproducibility_at_tag() -> dict:
    """Same query at the same tag returns byte-identical results.
    Pass/fail. The standard ANN systems can't pass this because their
    indexes have non-deterministic insertion-order effects; turtlelake
    can because Lance versions are content-addressed."""
    root = _bench_root() / f"tlk-bench-repro-{os.getpid()}"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    try:
        # Set up a dataset, write some embeddings, tag it.
        ds = Dataset.open(root)
        label = NamedNode("http://www.w3.org/2000/01/rdf-schema#label")
        ds._append_quads(
            [
                Quad(NamedNode(f"https://ex/{i}"), label, Literal(f"e{i}"))
                for i in range(50)
            ],
            batch_size=100,
        )
        import numpy as np

        rng = np.random.default_rng(7)
        vecs = rng.standard_normal((50, 16)).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        ds.embed(
            [f"https://ex/{i}" for i in range(50)],
            vecs.tolist(),
            model_id="repro:m1",
        )
        ds.checkpoint("v1")

        # Mutate after the tag.
        ds.embed(["https://ex/extra"], [[1.0] * 16], model_id="repro:m1")

        # Re-open at v1 from N independent handles; each must return
        # the same hits in the same order.
        runs = []
        for _ in range(5):
            kg = Dataset.open(root, tag="v1")
            hits = kg.vector_search(vecs[0].tolist(), k=5)
            runs.append(tuple((h["iri"], round(h["distance"], 6)) for h in hits))

        all_match = all(r == runs[0] for r in runs)
        return {
            "runs": len(runs),
            "byte_identical": bool(all_match),
            "first_result_set": list(runs[0]),
        }
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ── 4. Crash-recovery convergence ───────────────────────────────────


def bench_crash_recovery() -> dict:
    """Kill a process mid-checkpoint, reopen → does the dataset
    converge to a consistent state?

    We run the writer in a subprocess so we can deliver SIGKILL
    deterministically. The subprocess's only job: write one quad,
    write the WAL pending_checkpoint marker, create the triples tag,
    then deliberately crash before tagging the embeddings dataset.
    Reopening must reconcile the partial state via
    `_recover_pending_checkpoint`.
    """
    root = _bench_root() / f"tlk-bench-crash-{os.getpid()}"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    helper = Path(__file__).parent / "_crash_helper.py"
    try:
        result = subprocess.run(
            [sys.executable, str(helper), str(root)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        crashed = result.returncode != 0
        # Reopen — _recover_pending_checkpoint must converge state.
        kg = Dataset.open(root)
        triples_has = "v1" in kg.tags()
        emb_has = (
            kg._embeddings is not None
            and "v1" in list(kg._embeddings.tags.list())
        )
        manifest = kg._read_manifest()
        marker_cleared = "pending_checkpoint" not in manifest

        return {
            "subprocess_crashed_intentionally": crashed,
            "triples_tagged_after_recovery": triples_has,
            "embeddings_tagged_after_recovery": emb_has,
            "wal_marker_cleared": marker_cleared,
            "pair_consistent": triples_has and emb_has,
        }
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ── 5. In-RAM speedup ───────────────────────────────────────────────


def bench_in_memory_speedup(rows: int = 50_000, dim: int = 64) -> dict:
    """Compare per-query latency on the canonical Lance path vs the
    in-RAM cache. The hypothesis: at small batches the cache is
    multiple-x faster because we skip per-call Lance overhead."""
    root = _bench_root() / f"tlk-bench-ram-{os.getpid()}"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    try:
        import numpy as np

        ds = Dataset.open(root)
        rng = np.random.default_rng(0)
        vecs = rng.standard_normal((rows, dim)).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        ds.embed(
            [f"v/{i}" for i in range(rows)],
            vecs.tolist(),
            model_id="ram:m1",
        )

        q = vecs[0].tolist()
        N = 200
        # Lance path
        t0 = time.perf_counter()
        for _ in range(N):
            ds.vector_search(q, k=10, in_memory=False)
        lance_t = time.perf_counter() - t0

        # Warm the cache and run again.
        ds.preload_vectors()
        t0 = time.perf_counter()
        for _ in range(N):
            ds.vector_search(q, k=10, in_memory=True)
        mem_t = time.perf_counter() - t0

        # Batch in-memory.
        qs = [vecs[i].tolist() for i in range(N)]
        t0 = time.perf_counter()
        ds.vector_search_batch(qs, k=10, in_memory=True)
        batch_t = time.perf_counter() - t0

        return {
            "rows": rows,
            "dim": dim,
            "queries": N,
            "lance_qps": round(N / lance_t, 1),
            "in_memory_qps": round(N / mem_t, 1),
            "in_memory_speedup_vs_lance": round(lance_t / mem_t, 2),
            "in_memory_batch_qps": round(N / batch_t, 1),
            "in_memory_batch_speedup_vs_lance": round(lance_t / batch_t, 2),
        }
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ── runner ──────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--only",
        choices=["open", "visibility", "repro", "crash", "ram", "all"],
        default="all",
    )
    ap.add_argument("--ram-rows", type=int, default=50_000)
    args = ap.parse_args()

    out: dict = {
        "benchmark_root": str(_bench_root()),
        "benchmark_root_is_tmpfs": _bench_root() == Path("/dev/shm"),
    }

    if args.only in ("open", "all"):
        print("running: open-to-first-query latency...", file=sys.stderr)
        out["open_to_first_query"] = bench_open_to_first_query()
    if args.only in ("visibility", "all"):
        print("running: update-visibility latency...", file=sys.stderr)
        out["update_visibility"] = bench_update_visibility()
    if args.only in ("repro", "all"):
        print("running: reproducibility at tag...", file=sys.stderr)
        out["reproducibility_at_tag"] = bench_reproducibility_at_tag()
    if args.only in ("crash", "all"):
        print("running: crash-recovery convergence...", file=sys.stderr)
        out["crash_recovery"] = bench_crash_recovery()
    if args.only in ("ram", "all"):
        print(f"running: in-RAM speedup ({args.ram_rows} rows)...", file=sys.stderr)
        out["in_memory_speedup"] = bench_in_memory_speedup(rows=args.ram_rows)

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
