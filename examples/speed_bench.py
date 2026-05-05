"""Speed benchmark: turtlelake's cached engine vs pure pyoxigraph.

Scenario: an agent browses a graph. It issues many SPARQL queries (or
entity() calls) between rare writes. That's the hot path.

We time three variants:

  (a) pyoxigraph direct         — the floor; no turtlelake overhead
  (b) turtlelake (cached)       — first query pays materialization,
                                    subsequent queries on the same
                                    version are free
  (c) turtlelake (no cache)     — fresh engine every query; this is
                                    what the project did pre-v0.0.2

Run (POSIX):   .venv/bin/python examples/speed_bench.py
Run (Windows): .venv\Scripts\python.exe examples\speed_bench.py
"""

from __future__ import annotations

import io as _io
import tempfile
import time
from pathlib import Path

from pyoxigraph import RdfFormat, Store

from turtlelake import Dataset
from turtlelake.engine import SparqlEngine

# Synthetic graph: 5000 devices * 3 predicates each = 15k triples.
N_DEVICES = 5000


def synth_ttl() -> str:
    lines = ["@prefix ex: <https://ex.org/> .", ""]
    for i in range(N_DEVICES):
        lines.append(
            f'ex:d{i} a ex:Device ; '
            f'ex:label "device-{i}" ; '
            f'ex:family ex:F{i % 10} .'
        )
    return "\n".join(lines)


QUERY = """
    PREFIX ex: <https://ex.org/>
    SELECT ?label WHERE {
        ?d a ex:Device ; ex:label ?label ; ex:family ex:F3 .
    }
    LIMIT 10
"""


# 100 distinct SPARQL queries — each asks about a *different* family.
# Proves the cache accelerates ALL queries on the same version, not just
# repeats of the same string.
def distinct_queries(n: int) -> list[str]:
    return [
        f"""
        PREFIX ex: <https://ex.org/>
        SELECT ?label WHERE {{
            ?d a ex:Device ; ex:label ?label ; ex:family ex:F{i % 10} .
        }}
        LIMIT 10
        """
        for i in range(n)
    ]


def bench(name: str, fn, n_queries: int) -> float:
    t0 = time.perf_counter()
    for _ in range(n_queries):
        fn()
    elapsed = time.perf_counter() - t0
    print(f"  {name:35s} {n_queries:4d} queries  | {elapsed*1000:7.1f} ms  "
          f"({elapsed*1000/n_queries:6.2f} ms/query)")
    return elapsed


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpd:
        tmp = Path(tmpd)
        ttl = tmp / "big.ttl"
        ttl.write_text(synth_ttl(), encoding="utf-8")

        print(f"\nSynthetic corpus: {N_DEVICES} devices x 3 predicates "
              f"({N_DEVICES * 3} triples)\n")

        n = 100  # query count in the hot loop

        # (a) pure pyoxigraph — the floor
        ox = Store(path=str(tmp / "ox"))
        ox.load(
            _io.BytesIO(ttl.read_bytes()), format=RdfFormat.TURTLE
        )
        pyox_elapsed = bench(
            "pyoxigraph (direct)",
            lambda: list(ox.query(QUERY)),
            n,
        )

        # (b) turtlelake with engine cache ON (the production path).
        # The first call eats materialization; subsequent calls are fast.
        kg = Dataset.open(tmp / "tl")
        kg.ingest_ttl(ttl)

        # Warm the cache once so we time the steady-state hot path.
        warm_t0 = time.perf_counter()
        kg.query(QUERY)
        warm_ms = (time.perf_counter() - warm_t0) * 1000
        print(f"  turtlelake (first query - cold)      warm-up    | {warm_ms:7.1f} ms")

        cached_elapsed = bench(
            "turtlelake (cached, same query ×N)",
            lambda: kg.query(QUERY),
            n,
        )

        # The scary bit: if someone thought "cache" meant "result cache",
        # they might expect a totally different query to pay the full
        # materialization cost again. It doesn't — the engine cache holds
        # the data, so any question runs fast.
        queries_mixed = distinct_queries(n)
        _iter = iter(queries_mixed)
        distinct_elapsed = bench(
            "turtlelake (cached, N DIFFERENT queries)",
            lambda: kg.query(next(_iter)),
            n,
        )

        # (c) turtlelake WITHOUT the cache — rebuild engine every call.
        # Simulates the pre-v0.0.2 behavior.
        def uncached() -> None:
            engine = SparqlEngine.from_lance(kg._lance)
            engine.query(QUERY)

        uncached_elapsed = bench(
            "turtlelake (rebuild every query)",
            uncached,
            n,
        )

        # (d) turtlelake entity() over the cached engine: one of the most
        # common agent calls. No pyoxigraph direct equivalent (no hop expand
        # built in), so we just report.
        entity_iri = "https://ex.org/d42"
        entity_elapsed = bench(
            "turtlelake.entity(iri, hops=1)",
            lambda: kg.entity(entity_iri, hops=1),
            n,
        )

        # (e) write-heavy agent loop: insert a small fact, then read.
        # Before the incremental cache, every insert invalidated the engine,
        # so the next read paid full re-materialization. Now the insert
        # pushes into the cache and the read is free.
        def write_then_read(i: int) -> None:
            kg.insert_turtle(
                f'<https://ex.org/new{i}> <https://ex.org/p> "v{i}" .'
            )
            kg.query(QUERY)

        t0 = time.perf_counter()
        for i in range(20):
            write_then_read(i)
        write_loop_ms = (time.perf_counter() - t0) * 1000
        print(
            f"  write+read loop (20 iters)               "
            f"·  {write_loop_ms:7.1f} ms  "
            f"({write_loop_ms / 20:6.2f} ms/iter)"
        )

        # (f) pre_warm on open: materialization cost paid at open time.
        pre_warm_t0 = time.perf_counter()
        kg_warm = Dataset.open(tmp / "tl", pre_warm=True)
        pre_warm_open_ms = (time.perf_counter() - pre_warm_t0) * 1000
        first_t0 = time.perf_counter()
        kg_warm.query(QUERY)
        first_query_warm_ms = (time.perf_counter() - first_t0) * 1000
        print(
            f"  pre_warm=True open+first query           "
            f"·  open {pre_warm_open_ms:7.1f} ms  "
            f"/  first query {first_query_warm_ms:.2f} ms"
        )

        print()
        pyox_per = pyox_elapsed / n * 1000
        cached_per = cached_elapsed / n * 1000
        distinct_per = distinct_elapsed / n * 1000
        print(f"  same-query per call:                  {cached_per:.2f} ms")
        print(f"  DIFFERENT queries per call:           {distinct_per:.2f} ms "
              f"← not a result cache, an engine cache")
        print(f"  pyoxigraph direct (reference):        {pyox_per:.2f} ms")
        print(f"  cached speedup vs rebuild-every-call: "
              f"{uncached_elapsed / cached_elapsed:6.1f}×")
        print(f"  entity() is a hot-path primitive:     "
              f"{entity_elapsed*1000/n:6.2f} ms/call")
        print(f"  one-time warm-up cost (materialize):  "
              f"{warm_ms:6.1f} ms  (amortized across every query on the version)")


if __name__ == "__main__":
    main()
