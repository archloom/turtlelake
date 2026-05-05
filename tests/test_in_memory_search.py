"""Tests for the in-RAM vector cache and the methods that build on it.

Pin three contracts:
1. `preload_vectors()` populates a deterministic snapshot of the
   embeddings dataset.
2. `vector_search(in_memory=True)` returns the same top-k IRIs
   (modulo distance precision) as the Lance path.
3. The cache invalidates on the next `embed()` write.
"""

import numpy as np
import pytest

from turtlelake import Dataset


def _seed(ds: Dataset, n: int = 100, dim: int = 16) -> np.ndarray:
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    ds.embed(
        [f"i/{i}" for i in range(n)],
        vecs.tolist(),
        model_id="m1",
    )
    return vecs


def test_preload_reports_cache_state(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    _seed(ds, n=10, dim=8)
    info = ds.preload_vectors()
    assert info["rows"] == 10
    assert info["dim"] == 8
    assert info["bytes"] == 10 * 8 * 4
    assert ds.has_warm_cache()


def test_in_memory_matches_lance_top_k(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    vecs = _seed(ds, n=50, dim=16)
    ds.preload_vectors()
    q = vecs[3].tolist()
    lance_hits = [h["iri"] for h in ds.vector_search(q, k=5, in_memory=False)]
    mem_hits = [h["iri"] for h in ds.vector_search(q, k=5, in_memory=True)]
    # Top-1 must match exactly (it's the query itself, distance ~ 0).
    assert lance_hits[0] == mem_hits[0]
    # The full top-5 should overlap heavily (Lance brute-force is exact at this
    # scale, in-memory is exact, the distance metric matches).
    overlap = len(set(lance_hits) & set(mem_hits))
    assert overlap >= 4, f"lance={lance_hits} mem={mem_hits}"


def test_in_memory_strict_requires_warm_cache(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    _seed(ds, n=5, dim=4)
    with pytest.raises(RuntimeError, match="no warm cache"):
        ds.vector_search([1.0, 0.0, 0.0, 0.0], k=1, in_memory=True)


def test_cache_invalidates_on_write(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    _seed(ds, n=5, dim=4)
    ds.preload_vectors()
    assert ds.has_warm_cache()
    ds.embed(["new"], [[0.1, 0.2, 0.3, 0.4]], model_id="m1")
    assert not ds.has_warm_cache()


def test_in_memory_auto_falls_back_when_cold(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    _seed(ds, n=5, dim=4)
    # No preload — auto must use Lance path silently.
    hits = ds.vector_search([1.0, 0.0, 0.0, 0.0], k=2, in_memory="auto")
    assert len(hits) == 2


def test_in_memory_filters_by_model_id_when_cache_unfiltered(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    ds.embed(["a", "b"], [[1.0, 0.0], [0.0, 1.0]], model_id="m1")
    ds.embed(["c", "d"], [[1.0, 0.0], [0.0, 1.0]], model_id="m2")
    ds.preload_vectors()
    hits = ds.vector_search([1.0, 0.0], k=10, model_id="m1", in_memory=True)
    iris = {h["iri"] for h in hits}
    assert iris == {"a", "b"}


def test_batch_in_memory_matches_per_query(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    vecs = _seed(ds, n=30, dim=8)
    ds.preload_vectors()
    qs = vecs[:4].tolist()
    per = [ds.vector_search(q, k=3, in_memory=True) for q in qs]
    batch = ds.vector_search_batch(qs, k=3, in_memory=True)
    assert len(batch) == 4
    for p, b in zip(per, batch):
        assert [h["iri"] for h in p] == [h["iri"] for h in b]


def test_hybrid_search_requires_text_index(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    _seed(ds, n=5, dim=4)
    with pytest.raises(RuntimeError, match="text index"):
        ds.hybrid_search("q", [1.0] * 4, k=1)


def test_hybrid_search_fuses_bm25_and_vector(tmp_path):
    from pyoxigraph import Literal, NamedNode, Quad

    ds = Dataset.open(tmp_path / "kg")
    label = NamedNode("http://www.w3.org/2000/01/rdf-schema#label")
    defi = NamedNode("http://www.w3.org/2004/02/skos/core#definition")
    quads = []
    for i in range(20):
        s = NamedNode(f"https://ex/{i}")
        quads.append(Quad(s, label, Literal(f"entity {i}")))
        quads.append(
            Quad(s, defi, Literal(f"description of entity {i} topic{i % 4}"))
        )
    ds._append_quads(quads, batch_size=100)
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((20, 8)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    ds.embed(
        [f"https://ex/{i}" for i in range(20)],
        vecs.tolist(),
        model_id="m1",
    )
    ds.preload_text_index()

    out = ds.hybrid_search("entity 5 description", vecs[5].tolist(), k=3)
    assert out[0]["iri"] == "https://ex/5"
    assert "bm25" in out[0]["sources"] or "vector" in out[0]["sources"]


def test_graph_rag_ppr_returns_seed_dominated_top_k(tmp_path):
    """A cycle graph: each node points to the next. PPR seeded on
    node 0 should rank 0 (and its near successors) above distant
    nodes."""
    from pyoxigraph import NamedNode, Quad

    ds = Dataset.open(tmp_path / "kg")
    nxt = NamedNode("https://ex/next")
    quads = [
        Quad(
            NamedNode(f"https://ex/{i}"),
            nxt,
            NamedNode(f"https://ex/{(i + 1) % 50}"),
        )
        for i in range(50)
    ]
    ds._append_quads(quads, batch_size=100)
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((50, 8)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    ds.embed(
        [f"https://ex/{i}" for i in range(50)],
        vecs.tolist(),
        model_id="m1",
    )
    out = ds.graph_rag_ppr(vecs[0].tolist(), k=3, seed_k=1, damping=0.5)
    assert out[0]["iri"] == "https://ex/0"


def test_graph_rag_ppr_validates_damping(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    _seed(ds, n=5, dim=4)
    with pytest.raises(ValueError, match="damping"):
        ds.graph_rag_ppr([1.0] * 4, damping=1.5)


def test_tune_nprobes_returns_valid_dict(tmp_path):
    """tune_nprobes() should return a dict shape we can persist."""
    ds = Dataset.open(tmp_path / "kg")
    rng = np.random.default_rng(0)
    n = 300
    vecs = rng.standard_normal((n, 8)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    ds.embed(
        [f"v/{i}" for i in range(n)],
        vecs.tolist(),
        model_id="m1",
    )
    ds.build_vector_index(index_type="IVF_FLAT", num_partitions=4)
    out = ds.tune_nprobes(target_recall=0.5, sample_queries=10)
    assert "nprobes" in out
    assert "achieved_recall" in out
    assert 0.0 <= out["achieved_recall"] <= 1.0
