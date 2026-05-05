"""Reproducible benchmark: flat vector search vs `graph_rag` on a
synthetic multi-hop QA dataset.

Why synthetic. The goal is to measure the *retrieval-shape* difference
between (a) raw vector search and (b) vector search + structural
expansion. That difference shows up most clearly when the answer
entity is a few edges away from any vector-similar entity. A
controlled synthetic graph is the cleanest way to isolate the effect;
absolute numbers depend on the embedding model and corpus, not on the
shape we're studying.

What it builds:
  - A synthetic graph of `n_topics` topics; each topic has
    `papers_per_topic` papers; each paper has `authors_per_paper`
    authors. Edges: topic --has_paper--> paper --written_by--> author.
  - A descriptive label per node, embedded via a small but real
    transformation (we use a deterministic hash projection by default
    so the harness runs offline; pass --use-sentence-transformers to
    swap in all-MiniLM-L6-v2).

What it scores:
  - "What are the authors who write about <topic>?" — a 2-hop
    question. Answer: the set of authors whose papers are tagged
    with that topic.
  - For each question:
      flat   = vector_search(query_embedding, k=K)
      graph  = graph_rag(query_embedding, k=K_seed, hops=2) — collect
               every IRI mentioned in any expanded entity
  - Hit-rate = |{author IRIs in result} ∩ {gold authors}| /
               |{gold authors}|
  - Mean Reciprocal Rank for the strongest gold author found.

Usage:
    uv run python scripts/benchmark_graphrag.py
    uv run python scripts/benchmark_graphrag.py --topics 30 --papers 6
    uv run python scripts/benchmark_graphrag.py --use-sentence-transformers

Output is a single JSON line + a human-readable summary on stderr,
so you can pipe to `jq` from a CI run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import struct
import sys
import tempfile
import time
from pathlib import Path
from typing import Sequence

from pyoxigraph import Literal, NamedNode, Quad

from turtlelake import Dataset

NS = "https://benchmark.turtlelake/"
LABEL = NamedNode("http://www.w3.org/2000/01/rdf-schema#label")
TYPE = NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
HAS_PAPER = NamedNode(NS + "hasPaper")
WRITTEN_BY = NamedNode(NS + "writtenBy")
ABOUT = NamedNode(NS + "about")
TOPIC_CLASS = NamedNode(NS + "Topic")
PAPER_CLASS = NamedNode(NS + "Paper")
AUTHOR_CLASS = NamedNode(NS + "Author")


# ── synthetic embedding (deterministic, no deps) ────────────────────


def _hash_embed(text: str, dim: int = 64) -> list[float]:
    """Stable hash → float vector. Not semantic; exists so the harness
    runs without torch / a model server. Replace with a real model
    via `--use-sentence-transformers`."""
    out: list[float] = []
    seed = text.encode("utf-8")
    counter = 0
    while len(out) < dim:
        h = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
        for offset in range(0, len(h), 4):
            if len(out) >= dim:
                break
            (n,) = struct.unpack(">I", h[offset:offset + 4])
            out.append(n / 2**32 - 0.5)
        counter += 1
    return out


def _build_embedder(use_st: bool):
    if not use_st:
        return _hash_embed, 64, "demo:hash-64"
    from sentence_transformers import SentenceTransformer

    name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(name)
    dim = model.get_sentence_embedding_dimension()

    def _embed(text: str) -> list[float]:
        return model.encode([text], normalize_embeddings=True)[0].tolist()

    return _embed, dim, name


# ── synthetic dataset construction ─────────────────────────────────


def build_dataset(
    path: Path,
    *,
    n_topics: int,
    papers_per_topic: int,
    authors_per_paper: int,
    embed_fn,
    model_id: str,
) -> dict:
    """Build a synthetic graph and embed every node by its label.

    Returns the gold answer mapping `{topic_iri: set(author_iri)}` so
    the scoring step can check retrieval against ground truth.
    """
    random.seed(0)
    ds = Dataset.open(path)

    quads: list[Quad] = []
    iris: list[str] = []
    labels: list[str] = []
    gold: dict[str, set[str]] = {}

    for ti in range(n_topics):
        topic_iri = f"{NS}topic/{ti}"
        topic_label = f"Topic {ti}: {_topic_phrase(ti)}"
        quads.append(Quad(NamedNode(topic_iri), TYPE, TOPIC_CLASS))
        quads.append(Quad(NamedNode(topic_iri), LABEL, Literal(topic_label)))
        iris.append(topic_iri)
        labels.append(topic_label)
        gold[topic_iri] = set()

        for pi in range(papers_per_topic):
            paper_iri = f"{NS}paper/{ti}-{pi}"
            paper_label = f"Paper on {_topic_phrase(ti)} #{pi}"
            quads.append(Quad(NamedNode(paper_iri), TYPE, PAPER_CLASS))
            quads.append(Quad(NamedNode(paper_iri), LABEL, Literal(paper_label)))
            quads.append(Quad(NamedNode(paper_iri), ABOUT, NamedNode(topic_iri)))
            quads.append(
                Quad(NamedNode(topic_iri), HAS_PAPER, NamedNode(paper_iri))
            )
            iris.append(paper_iri)
            labels.append(paper_label)

            for ai in range(authors_per_paper):
                # Authors are unique per paper to make scoring crisp.
                author_iri = f"{NS}author/{ti}-{pi}-{ai}"
                author_label = _author_name(ti * 1000 + pi * 100 + ai)
                quads.append(Quad(NamedNode(author_iri), TYPE, AUTHOR_CLASS))
                quads.append(Quad(NamedNode(author_iri), LABEL, Literal(author_label)))
                quads.append(
                    Quad(NamedNode(paper_iri), WRITTEN_BY, NamedNode(author_iri))
                )
                iris.append(author_iri)
                labels.append(author_label)
                gold[topic_iri].add(author_iri)

    ds._append_quads(quads, batch_size=10_000)
    vectors = [embed_fn(label) for label in labels]
    ds.embed(iris, vectors, model_id=model_id)
    return gold


# ── retrieval methods ─────────────────────────────────────────────


def retrieve_flat(
    ds: Dataset, query_vec: Sequence[float], k: int, model_id: str
) -> list[str]:
    """Pure vector search: top-k IRIs by ANN distance, no expansion."""
    return [h["iri"] for h in ds.vector_search(query_vec, k=k, model_id=model_id)]


def retrieve_graph_rag(
    ds: Dataset,
    query_vec: Sequence[float],
    k_seed: int,
    hops: int,
    model_id: str,
) -> list[str]:
    """GraphRAG: top-k seeds, then walk every hop. Returns every IRI
    that appears in the expanded subgraphs. Order: seeds first, then
    neighbors in BFS order."""
    out = ds.graph_rag(query_vec, k=k_seed, hops=hops, model_id=model_id)
    seen: list[str] = []
    seen_set: set[str] = set()
    for hit in out["hits"]:
        if hit["iri"] not in seen_set:
            seen.append(hit["iri"])
            seen_set.add(hit["iri"])
    for iri, entity in out["entities"].items():
        # Include neighbors discovered during expansion.
        for edge in entity.get("outgoing", []):
            o = edge["object"]
            if o.get("type") == "iri" and o["value"] not in seen_set:
                seen.append(o["value"])
                seen_set.add(o["value"])
        for edge in entity.get("incoming", []):
            s = edge["subject"]
            if s not in seen_set:
                seen.append(s)
                seen_set.add(s)
        for nb_iri, nb in entity.get("neighbors", {}).items():
            if nb_iri not in seen_set:
                seen.append(nb_iri)
                seen_set.add(nb_iri)
            for edge in nb.get("outgoing", []):
                o = edge["object"]
                if o.get("type") == "iri" and o["value"] not in seen_set:
                    seen.append(o["value"])
                    seen_set.add(o["value"])
    return seen


# ── scoring ─────────────────────────────────────────────────────


def score_recall_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    """Fraction of gold IRIs that appear in the first `k` retrieved."""
    if not gold:
        return 0.0
    top = set(retrieved[:k])
    return len(top & gold) / len(gold)


def score_mrr(retrieved: list[str], gold: set[str]) -> float:
    """Reciprocal rank of the first gold IRI in retrieval. 0 if none."""
    for i, iri in enumerate(retrieved, start=1):
        if iri in gold:
            return 1.0 / i
    return 0.0


# ── label helpers (deterministic) ──────────────────────────────────


def _topic_phrase(i: int) -> str:
    seeds = [
        "graph databases", "vector retrieval", "knowledge graphs",
        "approximate nearest neighbors", "Arrow columnar formats",
        "RDF reasoning", "lakehouse architecture", "embeddings drift",
        "agent memory", "ontology alignment", "SHACL validation",
        "provenance tracking", "MCP servers", "SPARQL optimization",
        "lance versioning",
    ]
    return seeds[i % len(seeds)]


def _author_name(seed: int) -> str:
    first = ["Alex", "Bo", "Cam", "Dani", "Erin", "Fei", "Gus", "Hana", "Ivo", "Jay"]
    last = ["Park", "Singh", "Lopez", "Ng", "Costa", "Eze", "Volkov", "Tanaka", "Adel", "Sato"]
    return f"{first[seed % 10]} {last[(seed // 10) % 10]}"


# ── main ─────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", type=int, default=15)
    ap.add_argument("--papers", type=int, default=4)
    ap.add_argument("--authors", type=int, default=2)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--use-sentence-transformers", action="store_true")
    args = ap.parse_args()

    embed_fn, dim, model_id = _build_embedder(args.use_sentence_transformers)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "bench.turtlelake"
        gold = build_dataset(
            path,
            n_topics=args.topics,
            papers_per_topic=args.papers,
            authors_per_paper=args.authors,
            embed_fn=embed_fn,
            model_id=model_id,
        )
        ds = Dataset.open(path)

        flat_recalls, graph_recalls = [], []
        flat_mrrs, graph_mrrs = [], []
        flat_t, graph_t = 0.0, 0.0

        for topic_iri, gold_authors in gold.items():
            # Question: who wrote about this topic? The query embedding
            # is the TOPIC label itself, so flat search gets a fair
            # shot — it just retrieves the topic, not the authors. The
            # whole point: even when flat retrieves the right *anchor*
            # entity, the answer (the authors) is two hops away. Only
            # graph_rag traverses that structure.
            ti = int(topic_iri.rsplit("/", 1)[-1])
            query_label = f"Topic {ti}: {_topic_phrase(ti)}"
            q_vec = embed_fn(query_label)

            t0 = time.perf_counter()
            flat = retrieve_flat(ds, q_vec, k=args.k, model_id=model_id)
            flat_t += time.perf_counter() - t0

            t0 = time.perf_counter()
            graph = retrieve_graph_rag(
                ds, q_vec, k_seed=3, hops=2, model_id=model_id
            )
            graph_t += time.perf_counter() - t0

            flat_recalls.append(score_recall_at_k(flat, gold_authors, args.k))
            graph_recalls.append(score_recall_at_k(graph, gold_authors, len(graph)))
            flat_mrrs.append(score_mrr(flat, gold_authors))
            graph_mrrs.append(score_mrr(graph, gold_authors))

        n = len(gold)
        report = {
            "config": {
                "topics": args.topics,
                "papers_per_topic": args.papers,
                "authors_per_paper": args.authors,
                "k": args.k,
                "embedding": model_id,
                "embedding_dim": dim,
            },
            "flat": {
                "recall_at_k": sum(flat_recalls) / n,
                "mrr": sum(flat_mrrs) / n,
                "total_seconds": round(flat_t, 4),
            },
            "graph_rag": {
                "recall_at_expanded_set": sum(graph_recalls) / n,
                "mrr": sum(graph_mrrs) / n,
                "total_seconds": round(graph_t, 4),
            },
            "delta": {
                "recall_lift": (sum(graph_recalls) - sum(flat_recalls)) / n,
                "mrr_lift": (sum(graph_mrrs) - sum(flat_mrrs)) / n,
            },
        }

        print(json.dumps(report, indent=2))
        # Human-readable summary on stderr so JSON pipes stay clean.
        flat_r = report["flat"]["recall_at_k"]
        graph_r = report["graph_rag"]["recall_at_expanded_set"]
        flat_m = report["flat"]["mrr"]
        graph_m = report["graph_rag"]["mrr"]
        print(
            f"\n  flat   recall@{args.k} = {flat_r:.3f}    MRR = {flat_m:.3f}\n"
            f"  graph  recall_set    = {graph_r:.3f}    MRR = {graph_m:.3f}\n"
            f"  Δ recall: {report['delta']['recall_lift']:+.3f}   "
            f"Δ MRR: {report['delta']['mrr_lift']:+.3f}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
