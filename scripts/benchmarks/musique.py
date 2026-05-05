"""MuSiQue multi-hop QA benchmark for turtlelake's GraphRAG.

Compares **flat vector search** vs **`graph_rag` with title-mention
edges** on the HippoRAG-sampled MuSiQue subset (1000 questions,
11656 corpus paragraphs).

How it works:

  1. Download the dataset from the HippoRAG repo (CC-BY-4.0).
  2. Build a single global KG over all corpus paragraphs:
       - each paragraph → entity with `rdfs:label = title`
                                      `skos:definition = text`
       - add a `mentions` edge from paragraph A to paragraph B
         whenever A's text mentions B's title (case-insensitive).
       This is the standard multi-hop retrieval graph used by
       HippoRAG and other GraphRAG systems.
  3. Embed every paragraph (sentence-transformers if reachable,
     LSA otherwise -- see `_common.build_embedder`).
  4. For each question:
       flat   = vector_search(q, k=K)
       graph  = graph_rag(q, k=K_seed, hops=H)  → flatten neighbors
  5. Score: paragraph-level recall@k against the supporting-paragraph
     gold set (the set of corpus indices flagged `is_supporting=True`
     in MuSiQue's per-question paragraph list).

The metric is **supporting-paragraph recall@k**, the same metric
HippoRAG reports. We do not score answer-string correctness -- that
needs an LLM and conflates retrieval with generation. Retrieval is
what turtlelake controls; we measure that.

Usage:
    uv run python scripts/benchmarks/musique.py
    uv run python scripts/benchmarks/musique.py --max-questions 100
    uv run python scripts/benchmarks/musique.py --hops 2 --k-seed 3
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
from pathlib import Path

from pyoxigraph import Literal, NamedNode, Quad

from turtlelake import Dataset

# Local import -- the runner script lives in scripts/benchmarks/.
sys.path.insert(0, str(Path(__file__).parent))
from _common import (  # noqa: E402  (sys.path mutation above is required)
    build_embedder,
    expand_graph_rag,
    http_download,
    recall_at_k,
)


MUSIQUE_QUESTIONS_URL = (
    "https://raw.githubusercontent.com/OSU-NLP-Group/HippoRAG/main/"
    "reproduce/dataset/musique.json"
)
MUSIQUE_CORPUS_URL = (
    "https://raw.githubusercontent.com/OSU-NLP-Group/HippoRAG/main/"
    "reproduce/dataset/musique_corpus.json"
)

NS = "https://benchmark.turtlelake/musique/"
LABEL = NamedNode("http://www.w3.org/2000/01/rdf-schema#label")
DEFINITION = NamedNode("http://www.w3.org/2004/02/skos/core#definition")
MENTIONS = NamedNode(NS + "mentions")
PARA_CLASS = NamedNode(NS + "Paragraph")
TYPE = NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")


def _para_iri(idx: int) -> str:
    return f"{NS}para/{idx}"


def _build_mentions_edges(corpus: list[dict], title_to_idx: dict[str, int]) -> list[tuple[int, int]]:
    """Title-mention edges: A → B if A's text mentions B's title.

    Fast path: compile a single alternation regex over all titles
    (Python's re module collapses this to a trie internally). One
    `findall` per paragraph instead of N-titles passes. ~100× faster
    than the per-title regex loop on the 11k-paragraph MuSiQue
    corpus."""
    titles = [(t, i) for t, i in title_to_idx.items() if len(t) >= 3]
    if not titles:
        return []
    # Sort by length descending so the trie prefers longer matches
    # (avoids "Spain" matching inside "Spain national football team").
    titles.sort(key=lambda ti: -len(ti[0]))
    title_lookup = {t.lower(): i for t, i in titles}
    pattern = re.compile(
        r"\b(?:" + "|".join(re.escape(t) for t, _ in titles) + r")\b",
        re.IGNORECASE,
    )

    edges: list[tuple[int, int]] = []
    for a_idx, item in enumerate(corpus):
        text = item.get("text", "")
        if not text:
            continue
        seen: set[int] = set()
        for m in pattern.finditer(text):
            b_idx = title_lookup.get(m.group(0).lower())
            if b_idx is None or b_idx == a_idx or b_idx in seen:
                continue
            seen.add(b_idx)
            edges.append((a_idx, b_idx))
    return edges


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-questions", type=int, default=200,
                    help="0 means every question in the dataset")
    ap.add_argument("--k", type=int, default=10, help="recall@k")
    ap.add_argument("--k-seed", type=int, default=3,
                    help="seed-set size for graph_rag")
    ap.add_argument("--hops", type=int, default=1, help="hops for graph_rag")
    ap.add_argument("--embedding-prefer", choices=["auto", "st", "lsa"], default="auto")
    ap.add_argument("--embedding-dim", type=int, default=128)
    ap.add_argument("--st-model", default=None,
                    help="HF sentence-transformers model id "
                         "(default: all-MiniLM-L6-v2)")
    ap.add_argument("--st-query-prefix", default="",
                    help='E5/BGE prefix for query strings, e.g. "query: "')
    ap.add_argument("--st-passage-prefix", default="",
                    help='E5/BGE prefix for passage strings, e.g. "passage: "')
    ap.add_argument("--out", type=str, default="-",
                    help="JSON output path (default stdout)")
    ap.add_argument("--methods", default="flat,graph,hybrid,ppr",
                    help="comma-separated subset of "
                         "{flat,graph,hybrid,ppr}; default runs all")
    ap.add_argument("--in-memory", default="auto",
                    choices=["auto", "true", "false"])
    args = ap.parse_args()

    print("downloading MuSiQue dataset...", file=sys.stderr)
    questions = json.loads(http_download(MUSIQUE_QUESTIONS_URL).read_text())
    corpus = json.loads(http_download(MUSIQUE_CORPUS_URL).read_text())
    print(f"  {len(questions)} questions, {len(corpus)} corpus paragraphs",
          file=sys.stderr)

    title_to_idx: dict[str, int] = {}
    for i, item in enumerate(corpus):
        title = item.get("title")
        if title and title not in title_to_idx:
            title_to_idx[title] = i

    # Build the gold mapping. MuSiQue's supporting-paragraph annotation
    # is per-question and refers to the question's own 20 paragraphs
    # (via `idx` and `is_supporting`). We resolve those by title to the
    # corresponding corpus index.
    gold_by_q: dict[str, set[int]] = {}
    for q in questions:
        gold = set()
        for p in q.get("paragraphs", []):
            if p.get("is_supporting"):
                idx = title_to_idx.get(p["title"])
                if idx is not None:
                    gold.add(idx)
        gold_by_q[q["id"]] = gold

    # Skip questions whose supporting paragraphs aren't all in the
    # corpus (rare; happens when titles are truncated/normalized
    # differently). They'd score 0/0 = NaN otherwise.
    valid_qs = [q for q in questions if gold_by_q[q["id"]]]
    if args.max_questions and args.max_questions > 0:
        valid_qs = valid_qs[: args.max_questions]
    print(f"  scoring {len(valid_qs)} questions", file=sys.stderr)

    # Embed paragraphs once. Question embeddings reuse the same model.
    texts = [f"{c['title']}. {c['text']}" for c in corpus]
    print("  fitting embedder...", file=sys.stderr)
    t0 = time.perf_counter()
    embed_fn, model_id, emb_dim = build_embedder(
        texts,
        dim=args.embedding_dim,
        prefer=args.embedding_prefer,
        st_model=args.st_model,
        st_query_prefix=args.st_query_prefix,
        st_passage_prefix=args.st_passage_prefix,
    )
    print(f"  embedder: {model_id} (dim={emb_dim}) in {time.perf_counter()-t0:.1f}s",
          file=sys.stderr)

    print("  embedding paragraphs...", file=sys.stderr)
    t0 = time.perf_counter()
    para_vecs = embed_fn(texts)
    print(f"  embedded {len(para_vecs)} paragraphs in {time.perf_counter()-t0:.1f}s",
          file=sys.stderr)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "musique.turtlelake"
        ds = Dataset.open(path)

        # Ingest paragraphs as RDF entities.
        print("  ingesting RDF...", file=sys.stderr)
        t0 = time.perf_counter()
        quads: list[Quad] = []
        for i, item in enumerate(corpus):
            iri = NamedNode(_para_iri(i))
            quads.append(Quad(iri, TYPE, PARA_CLASS))
            if item.get("title"):
                quads.append(Quad(iri, LABEL, Literal(item["title"])))
            if item.get("text"):
                # Keep the literal short so pyoxigraph doesn't drag.
                quads.append(Quad(iri, DEFINITION, Literal(item["text"][:5000])))
        edges = _build_mentions_edges(corpus, title_to_idx)
        for a_idx, b_idx in edges:
            quads.append(
                Quad(
                    NamedNode(_para_iri(a_idx)),
                    MENTIONS,
                    NamedNode(_para_iri(b_idx)),
                )
            )
        ds._append_quads(quads, batch_size=100_000)
        print(f"    {len(quads)} quads, {len(edges)} mention edges in "
              f"{time.perf_counter()-t0:.1f}s", file=sys.stderr)

        # Embed.
        iris = [_para_iri(i) for i in range(len(corpus))]
        ds.embed(iris, para_vecs, model_id=model_id)
        # Pre-warm the in-memory cache so vector_search bypasses Lance --         # for 11k vectors this is ~10× faster per query and the gain
        # carries through hybrid + ppr (both call vector_search).
        ds.preload_vectors()
        # Build a BM25 index over labels + definitions for the hybrid
        # path; cheap (under a second) and reused across all questions.
        ds.preload_text_index()
        # Build an index (auto-policy will pick IVF_FLAT here).
        idx_status = ds.build_vector_index()
        print(f"  index: {idx_status['action']} ({idx_status.get('reason', 'auto')})",
              file=sys.stderr)

        # Score across the requested method set.
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
        valid_methods = {"flat", "graph", "hybrid", "ppr"}
        bad = [m for m in methods if m not in valid_methods]
        if bad:
            raise SystemExit(f"unknown methods: {bad}")
        recalls: dict[str, list[float]] = {m: [] for m in methods}
        durations: dict[str, float] = {m: 0.0 for m in methods}

        # Per-hop-class buckets so we can report which question shapes
        # benefit from graph expansion.
        by_hop: dict[str, dict[str, list[float]]] = {}

        print("  scoring questions...", file=sys.stderr)
        t_total = time.perf_counter()
        for nq, q in enumerate(valid_qs):
            q_vec = embed_fn(q["question"])
            gold = {_para_iri(i) for i in gold_by_q[q["id"]]}
            hop_class = q["id"].split("__")[0]
            slot = by_hop.setdefault(hop_class, {m: [] for m in methods})

            for method in methods:
                t0 = time.perf_counter()
                if method == "flat":
                    out_iris = [
                        h["iri"]
                        for h in ds.vector_search(q_vec, k=args.k, model_id=model_id)
                    ]
                    score_k = args.k
                elif method == "graph":
                    graph_out = ds.graph_rag(
                        q_vec, k=args.k_seed, hops=args.hops, model_id=model_id
                    )
                    out_iris = expand_graph_rag(graph_out)
                    score_k = len(out_iris) or args.k
                elif method == "hybrid":
                    out_iris = [
                        h["iri"]
                        for h in ds.hybrid_search(
                            q["question"], q_vec, k=args.k, model_id=model_id
                        )
                    ]
                    score_k = args.k
                elif method == "ppr":
                    out_iris = [
                        h["iri"]
                        for h in ds.graph_rag_ppr(
                            q_vec,
                            k=args.k,
                            seed_k=args.k_seed,
                            damping=0.5,
                            iterations=20,
                            model_id=model_id,
                            edge_predicates=[str(MENTIONS)],
                        )
                    ]
                    score_k = args.k
                else:
                    raise AssertionError(method)  # unreachable
                durations[method] += time.perf_counter() - t0
                r = recall_at_k(out_iris, gold, score_k)
                recalls[method].append(r)
                slot[method].append(r)

            if (nq + 1) % 50 == 0:
                done = nq + 1
                line = "  ".join(
                    f"{m}={sum(recalls[m])/done:.3f}" for m in methods
                )
                print(f"    {done}/{len(valid_qs)}  {line}", file=sys.stderr)

        n = len(valid_qs)
        report = {
            "config": {
                "questions_scored": n,
                "k_flat": args.k,
                "k_seed_graph": args.k_seed,
                "hops": args.hops,
                "embedding": model_id,
                "embedding_dim": emb_dim,
                "corpus_size": len(corpus),
                "mention_edges": len(edges),
                "in_memory": args.in_memory,
                "methods": methods,
            },
            "results": {
                m: {
                    "recall": sum(recalls[m]) / n,
                    "total_seconds": round(durations[m], 3),
                    "queries_per_second": (
                        round(n / durations[m], 1) if durations[m] else None
                    ),
                }
                for m in methods
            },
            "by_hop_class": {
                hop: {
                    "n": len(next(iter(slots.values()))),
                    **{
                        f"{m}_recall": sum(slots[m]) / len(slots[m])
                        for m in methods
                    },
                }
                for hop, slots in sorted(by_hop.items())
            },
            "wall_seconds": round(time.perf_counter() - t_total, 1),
        }

        text = json.dumps(report, indent=2)
        if args.out == "-":
            print(text)
        else:
            Path(args.out).write_text(text)
            print(f"wrote {args.out}", file=sys.stderr)
        # Brief summary to stderr.
        lines = [
            f"  {m:<8} recall = {report['results'][m]['recall']:.3f}    "
            f"QPS = {report['results'][m]['queries_per_second']}"
            for m in methods
        ]
        print("\n" + "\n".join(lines), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
