"""GraphRAG retrieval: vector hits + structural expansion in one call.

This is the agent-facing reason the vector layer exists. Lock the shape
of `graph_rag()` and prove it composes vector_search with entity().
"""

from turtlelake import Dataset


def _seed(ds: Dataset) -> None:
    from pyoxigraph import Literal, NamedNode, Quad

    label = NamedNode("http://www.w3.org/2000/01/rdf-schema#label")
    family = NamedNode("https://ex.org/ont#family")
    a = NamedNode("https://ex.org/A")
    b = NamedNode("https://ex.org/B")
    fam = NamedNode("https://ex.org/family/X")
    ds._append_quads(
        [
            Quad(a, label, Literal("Device A")),
            Quad(a, family, fam),
            Quad(b, label, Literal("Device B")),
            Quad(b, family, fam),
            Quad(fam, label, Literal("Family X")),
        ],
        batch_size=10,
    )
    ds.embed(
        ["https://ex.org/A", "https://ex.org/B", "https://ex.org/family/X"],
        [[1.0, 0.0, 0.0], [0.95, 0.0, 0.0], [0.0, 1.0, 0.0]],
        model_id="m1",
    )


def test_graph_rag_returns_hits_and_entities(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    _seed(ds)
    out = ds.graph_rag([1.0, 0.0, 0.0], k=2, hops=1)
    assert set(out) == {"hits", "entities"}
    iris = [h["iri"] for h in out["hits"]]
    assert iris == ["https://ex.org/A", "https://ex.org/B"]
    # Each hit gets a structural expansion
    assert set(out["entities"]) == set(iris)
    a_entity = out["entities"]["https://ex.org/A"]
    predicates = {e["predicate"] for e in a_entity["outgoing"]}
    assert "https://ex.org/ont#family" in predicates


def test_graph_rag_respects_k(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    _seed(ds)
    out = ds.graph_rag([1.0, 0.0, 0.0], k=1, hops=1)
    assert len(out["hits"]) == 1
    assert len(out["entities"]) == 1


def test_graph_rag_two_hops_expands_neighbors(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    _seed(ds)
    out = ds.graph_rag([1.0, 0.0, 0.0], k=1, hops=2)
    a_entity = out["entities"]["https://ex.org/A"]
    assert "neighbors" in a_entity
    assert "https://ex.org/family/X" in a_entity["neighbors"]


def test_graph_rag_filters_by_model_id(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    _seed(ds)
    # Add an unrelated model — its vectors must not pollute the hits.
    ds.embed(["https://ex.org/A"], [[0.0, 0.0, 1.0]], model_id="m2")
    out = ds.graph_rag([1.0, 0.0, 0.0], k=2, hops=1, model_id="m1")
    assert all(h["model_id"] == "m1" for h in out["hits"])


def test_graph_rag_without_embeddings_raises(tmp_path):
    import pytest

    ds = Dataset.open(tmp_path / "kg")
    with pytest.raises(RuntimeError, match="No embeddings"):
        ds.graph_rag([1.0, 0.0, 0.0], k=3, hops=1)
