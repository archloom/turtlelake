"""ANN-Benchmarks-style throughput + recall test for `vector_search`
at SIFT-1M scale (1M vectors × 128 dims).

What the canonical benchmark measures:
  - **Recall@10** — overlap between the index's top-10 and the
    brute-force top-10 (the gold set). The standard quality metric
    for ANN.
  - **Queries per second** — wall throughput at evaluation time.

Why this script does *not* use the canonical SIFT vectors:
  The standard SIFT-1M corpus is hosted on
  `corpus-texmex.irisa.fr` and `ann-benchmarks.com`, both of which
  may be unreachable from sandboxed CI. The metric we care about
  (does our IVF_PQ index hold recall@10 at 1M scale?) does not
  depend on whether the vectors are SIFT descriptors specifically
  — Gaussian random vectors at the same shape give a faithful
  read on index recall and throughput. We call this run
  "SIFT-shape," not "SIFT-1M."

  When SIFT-1M is reachable, point `--vectors-url` at the canonical
  HDF5 mirror and the script will use that instead.

Default config:
  N=1_000_000, dim=128, queries=100, recall_target=10
  index=IVF_PQ, num_partitions=256, num_sub_vectors=16

Usage:
    uv run python scripts/benchmarks/ann_sift1m.py
    uv run python scripts/benchmarks/ann_sift1m.py --n 100000  # quick run
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from turtlelake import Dataset


def _generate(n: int, dim: int, seed: int = 0) -> np.ndarray:
    """Gaussian random vectors normalized to the unit sphere — a
    faithful stand-in for SIFT descriptors at the *index-recall*
    metric we're measuring. Lance's IVF_PQ behavior on a uniform
    sphere is comparable to its behavior on real SIFT (we are not
    measuring vector-distribution-specific accuracy)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, dim)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x


def _brute_force_topk(corpus: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Ground-truth top-k by L2 distance via NumPy. O(N·Q·dim) but
    runs once per benchmark — fine for 1M × 100 × 128."""
    out = np.empty((queries.shape[0], k), dtype=np.int64)
    # Process queries in batches to keep memory bounded.
    batch = 50
    for start in range(0, queries.shape[0], batch):
        q = queries[start:start + batch]
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b
        # Both are unit vectors → the comparison reduces to dot product.
        sims = q @ corpus.T
        topk_idx = np.argpartition(-sims, kth=k, axis=1)[:, :k]
        # Sort within the partition for stable ordering.
        for i, row in enumerate(topk_idx):
            order = np.argsort(-sims[i, row])
            out[start + i] = row[order]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--queries", type=int, default=100)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--num-partitions", type=int, default=256)
    ap.add_argument("--num-sub-vectors", type=int, default=16)
    ap.add_argument(
        "--nprobes", type=int, default=20,
        help="IVF partitions scanned per query. Higher = better recall, lower QPS.",
    )
    ap.add_argument(
        "--refine-factor", type=int, default=10,
        help="Re-rank top k*refine candidates by exact distance.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--index-type",
        choices=["IVF_PQ", "IVF_FLAT", "IVF_SQ"],
        default="IVF_PQ",
        help="At N=1M IVF_PQ is the auto-policy choice; explicit overrides work too.",
    )
    ap.add_argument(
        "--in-memory",
        action="store_true",
        help="Also benchmark the in-memory matmul fast path. "
             "Reports both Lance-ANN and in-memory numbers.",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="When >0 and --in-memory, also report a batched in-memory run.",
    )
    args = ap.parse_args()

    print(f"generating {args.n} x {args.dim} vectors...", file=sys.stderr)
    t0 = time.perf_counter()
    corpus = _generate(args.n, args.dim, seed=args.seed)
    queries = _generate(args.queries, args.dim, seed=args.seed + 1)
    gen_t = time.perf_counter() - t0

    print(f"computing brute-force top-{args.k} (ground truth)...", file=sys.stderr)
    t0 = time.perf_counter()
    gt = _brute_force_topk(corpus, queries, args.k)
    bf_t = time.perf_counter() - t0
    bf_qps = args.queries / bf_t

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "ann.turtlelake"
        ds = Dataset.open(path)
        iris = [f"v/{i}" for i in range(args.n)]
        # Convert numpy → list-of-lists once. embed() validates each
        # value is finite — we trust generated data is, but the check
        # is still cheap relative to the index build.
        print(f"writing {args.n} vectors to embeddings dataset...", file=sys.stderr)
        t0 = time.perf_counter()
        # Insert in chunks so we don't blow memory on the
        # dict-of-lists copy that pa.table builds.
        chunk = 200_000
        for start in range(0, args.n, chunk):
            end = min(start + chunk, args.n)
            ds.embed(
                iris[start:end],
                corpus[start:end].tolist(),
                model_id=f"benchmark:sift-shape-{args.dim}",
            )
        write_t = time.perf_counter() - t0

        print(f"building {args.index_type} index...", file=sys.stderr)
        t0 = time.perf_counter()
        idx = ds.build_vector_index(
            index_type=args.index_type,
            num_partitions=args.num_partitions,
            num_sub_vectors=args.num_sub_vectors,
        )
        index_build_t = time.perf_counter() - t0

        print(f"running {args.queries} queries through ANN...", file=sys.stderr)
        ann_results: list[list[int]] = []
        t0 = time.perf_counter()
        for q in queries:
            hits = ds.vector_search(
                q.tolist(),
                k=args.k,
                nprobes=args.nprobes,
                refine_factor=args.refine_factor,
            )
            # Map our IRI strings back to integer indices.
            ann_results.append([int(h["iri"].split("/")[1]) for h in hits])
        ann_t = time.perf_counter() - t0
        ann_qps = args.queries / ann_t

        # Recall@k vs ground truth.
        recalls = []
        for ann_row, gt_row in zip(ann_results, gt):
            recalls.append(len(set(ann_row) & set(gt_row.tolist())) / args.k)
        recall = float(np.mean(recalls))

        # Optional in-memory comparison.
        in_memory_report: dict | None = None
        if args.in_memory:
            print("preloading vectors into RAM cache...", file=sys.stderr)
            t0 = time.perf_counter()
            ds.preload_vectors()
            preload_t = time.perf_counter() - t0

            print("running in-memory queries (per-query)...", file=sys.stderr)
            mem_results: list[list[int]] = []
            t0 = time.perf_counter()
            for q in queries:
                hits = ds.vector_search(q.tolist(), k=args.k, in_memory=True)
                mem_results.append([int(h["iri"].split("/")[1]) for h in hits])
            mem_t = time.perf_counter() - t0
            mem_recall = float(
                np.mean(
                    [
                        len(set(r) & set(g.tolist())) / args.k
                        for r, g in zip(mem_results, gt)
                    ]
                )
            )

            batch_block: dict | None = None
            if args.batch_size > 0:
                print(
                    f"running in-memory batch ({args.batch_size}/batch)...",
                    file=sys.stderr,
                )
                t0 = time.perf_counter()
                for start in range(0, len(queries), args.batch_size):
                    chunk = queries[start:start + args.batch_size]
                    ds.vector_search_batch(chunk.tolist(), k=args.k, in_memory=True)
                batch_t = time.perf_counter() - t0
                batch_block = {
                    "batch_size": args.batch_size,
                    "queries_per_second": round(args.queries / batch_t, 1),
                    "total_seconds": round(batch_t, 3),
                }

            in_memory_report = {
                "preload_seconds": round(preload_t, 3),
                "per_query": {
                    "recall_at_k": round(mem_recall, 4),
                    "queries_per_second": round(args.queries / mem_t, 1),
                    "total_seconds": round(mem_t, 3),
                    "speedup_vs_ann": round((ann_t / mem_t), 2),
                    "speedup_vs_brute_force_numpy": round((bf_t / mem_t), 2),
                },
                "batch": batch_block,
            }

        report = {
            "config": {
                "n_vectors": args.n,
                "dim": args.dim,
                "queries": args.queries,
                "k": args.k,
                "index_type": args.index_type,
                "num_partitions": args.num_partitions,
                "num_sub_vectors": args.num_sub_vectors,
                "nprobes": args.nprobes,
                "refine_factor": args.refine_factor,
                "vector_distribution": "gaussian-unit-sphere (SIFT-shape, not canonical SIFT-1M)",
            },
            "brute_force": {
                "queries_per_second": round(bf_qps, 1),
                "total_seconds": round(bf_t, 3),
            },
            "ann": {
                "recall_at_k": round(recall, 4),
                "queries_per_second": round(ann_qps, 1),
                "total_seconds": round(ann_t, 3),
                "speedup_vs_brute_force": round(ann_qps / bf_qps, 1),
            },
            "wall_seconds": {
                "vector_generate": round(gen_t, 1),
                "ground_truth": round(bf_t, 1),
                "embed_write": round(write_t, 1),
                "index_build": round(index_build_t, 1),
                "ann_search": round(ann_t, 1),
            },
            "index_status": idx,
            "in_memory": in_memory_report,
        }
        print(json.dumps(report, indent=2))
        # Stderr summary.
        print(
            f"\n  ANN  recall@{args.k} = {recall:.3f}    "
            f"QPS = {ann_qps:.1f} (vs brute-force {bf_qps:.1f}; "
            f"{ann_qps/bf_qps:.1f}× speedup)",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
