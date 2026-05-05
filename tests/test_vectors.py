"""Per-IRI vector layer: write, query, version, multi-model.

These tests lock the contract of the vector surface that the GraphRAG
repositioning depends on. If any of them fail, the agent-facing pitch
("semantic + structural retrieval in one artifact") regresses.
"""

import pytest

from turtlelake import Dataset


def _seed_triples(ds: Dataset) -> None:
    """A few quads so the entity-expansion path has data to chew on."""
    from pyoxigraph import Literal, NamedNode, Quad

    label = NamedNode("http://www.w3.org/2000/01/rdf-schema#label")
    s_a = NamedNode("https://ex.org/A")
    s_b = NamedNode("https://ex.org/B")
    ds._append_quads(
        [
            Quad(s_a, label, Literal("A")),
            Quad(s_b, label, Literal("B")),
        ],
        batch_size=10,
    )


def test_embed_creates_dataset_and_records_dim(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    _seed_triples(ds)
    n = ds.embed(
        ["https://ex.org/A", "https://ex.org/B"],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        model_id="test:m1",
    )
    assert n == 2
    assert ds.embedding_count() == 2
    assert ds.embedding_dim() == 3


def test_embed_rejects_dim_mismatch(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    with pytest.raises(ValueError, match="dim"):
        ds.embed(
            ["a", "b"],
            [[1.0, 0.0, 0.0], [1.0, 0.0]],  # different dims
            model_id="m",
        )


def test_embed_rejects_iri_vector_length_mismatch(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    with pytest.raises(ValueError, match="length mismatch"):
        ds.embed(["a"], [[1.0], [2.0]], model_id="m")


def test_vector_search_orders_by_distance(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    _seed_triples(ds)
    ds.embed(
        ["a", "b", "c"],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]],
        model_id="m1",
    )
    hits = ds.vector_search([1.0, 0.0, 0.0], k=3)
    iris_in_order = [h["iri"] for h in hits]
    assert iris_in_order[0] == "a"
    # Distances must be monotonically non-decreasing.
    distances = [h["distance"] for h in hits]
    assert distances == sorted(distances)


def test_vector_search_filters_by_model_id(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    _seed_triples(ds)
    ds.embed(["a"], [[1.0, 0.0, 0.0]], model_id="m1")
    ds.embed(["a"], [[0.0, 1.0, 0.0]], model_id="m2")
    only_m1 = ds.vector_search([1.0, 0.0, 0.0], k=10, model_id="m1")
    assert {h["model_id"] for h in only_m1} == {"m1"}


def test_vector_search_without_embeddings_raises(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    _seed_triples(ds)
    with pytest.raises(RuntimeError, match="No embeddings"):
        ds.vector_search([1.0, 0.0, 0.0], k=3)


def test_embed_appends_across_calls(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    ds.embed(["a"], [[1.0, 0.0]], model_id="m")
    ds.embed(["b"], [[0.0, 1.0]], model_id="m")
    assert ds.embedding_count() == 2


def test_embed_persists_across_reopen(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    ds.embed(["a", "b"], [[1.0, 0.0], [0.0, 1.0]], model_id="m")
    # Re-open without retaining the handle
    del ds
    again = Dataset.open(tmp_path / "kg")
    assert again.embedding_count() == 2
    assert again.embedding_dim() == 2
    hits = again.vector_search([1.0, 0.0], k=1)
    assert hits[0]["iri"] == "a"


def test_entity_with_similar_appends_neighbors(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    _seed_triples(ds)
    ds.embed(
        ["https://ex.org/A", "https://ex.org/B"],
        [[1.0, 0.0, 0.0], [0.99, 0.0, 0.0]],
        model_id="m",
    )
    out = ds.entity("https://ex.org/A", hops=1, similar=3)
    assert "similar" in out
    # Self-IRI must be filtered out of the similar list
    assert all(s["iri"] != "https://ex.org/A" for s in out["similar"])
    assert out["similar"][0]["iri"] == "https://ex.org/B"


def test_vector_search_accepts_ann_tunables(tmp_path):
    """`nprobes` and `refine_factor` are passed through to Lance for
    indexed search. At small scale (no index, brute-force scan) they
    are accepted as a no-op rather than rejected."""
    ds = Dataset.open(tmp_path / "kg")
    ds.embed(["a", "b"], [[1.0, 0.0], [0.0, 1.0]], model_id="m")
    hits = ds.vector_search([1.0, 0.0], k=2, nprobes=4, refine_factor=2)
    assert hits[0]["iri"] == "a"


def test_vector_search_rejects_invalid_nprobes(tmp_path):
    import pytest

    ds = Dataset.open(tmp_path / "kg")
    ds.embed(["a"], [[1.0]], model_id="m")
    with pytest.raises(ValueError, match="nprobes"):
        ds.vector_search([1.0], k=1, nprobes=0)


def test_vector_search_rejects_invalid_refine_factor(tmp_path):
    import pytest

    ds = Dataset.open(tmp_path / "kg")
    ds.embed(["a"], [[1.0]], model_id="m")
    with pytest.raises(ValueError, match="refine_factor"):
        ds.vector_search([1.0], k=1, refine_factor=0)


def test_entity_similar_omitted_when_no_embedding_for_iri(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    _seed_triples(ds)
    ds.embed(["https://ex.org/A"], [[1.0, 0.0]], model_id="m")
    out = ds.entity("https://ex.org/B", hops=1, similar=3)
    assert "similar" not in out  # B has no vector → field omitted, not empty
