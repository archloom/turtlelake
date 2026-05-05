"""Checkpoint/rollback must operate atomically across the triples and
embeddings datasets. If we tag, write to both, then roll back, both
must come back to the pre-checkpoint state.
"""

from pyoxigraph import Literal, NamedNode, Quad

from turtlelake import Dataset


def _seed(ds: Dataset) -> None:
    label = NamedNode("http://www.w3.org/2000/01/rdf-schema#label")
    a = NamedNode("https://ex.org/A")
    ds._append_quads([Quad(a, label, Literal("A"))], batch_size=10)
    ds.embed(["https://ex.org/A"], [[1.0, 0.0]], model_id="m")


def test_checkpoint_tags_both_datasets(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    _seed(ds)
    ds.checkpoint("baseline")
    # Lance keeps tags on each dataset independently — assert both saw it.
    assert "baseline" in ds.tags()
    assert "baseline" in list(ds._embeddings.tags.list())


def test_rollback_restores_triples_and_embeddings(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    _seed(ds)
    triples_before = ds.count()
    embeddings_before = ds.embedding_count()
    ds.checkpoint("pre")

    # Speculative writes on BOTH datasets.
    label = NamedNode("http://www.w3.org/2000/01/rdf-schema#label")
    b = NamedNode("https://ex.org/B")
    ds._append_quads([Quad(b, label, Literal("B"))], batch_size=10)
    ds.embed(["https://ex.org/B"], [[0.0, 1.0]], model_id="m")
    assert ds.count() == triples_before + 1
    assert ds.embedding_count() == embeddings_before + 1

    ds.rollback("pre")
    assert ds.count() == triples_before
    assert ds.embedding_count() == embeddings_before
    # And vector_search no longer surfaces B's vector.
    iris = {h["iri"] for h in ds.vector_search([0.0, 1.0], k=10)}
    assert "https://ex.org/B" not in iris


def test_open_at_tag_pins_both_datasets(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    _seed(ds)
    ds.checkpoint("v1")
    # Mutate after the tag.
    label = NamedNode("http://www.w3.org/2000/01/rdf-schema#label")
    b = NamedNode("https://ex.org/B")
    ds._append_quads([Quad(b, label, Literal("B"))], batch_size=10)
    ds.embed(["https://ex.org/B"], [[0.0, 1.0]], model_id="m")

    pinned = Dataset.open(tmp_path / "kg", tag="v1")
    assert pinned.count() == 1  # Only A
    assert pinned.embedding_count() == 1
    assert pinned.vector_search([1.0, 0.0], k=10)[0]["iri"] == "https://ex.org/A"


def test_rollback_when_no_embeddings_dataset_still_works(tmp_path):
    """Backwards-compatible: pre-vector datasets must still rollback."""
    ds = Dataset.open(tmp_path / "kg")
    label = NamedNode("http://www.w3.org/2000/01/rdf-schema#label")
    a = NamedNode("https://ex.org/A")
    ds._append_quads([Quad(a, label, Literal("A"))], batch_size=10)
    ds.checkpoint("pre")
    b = NamedNode("https://ex.org/B")
    ds._append_quads([Quad(b, label, Literal("B"))], batch_size=10)
    ds.rollback("pre")
    assert ds.count() == 1
    assert ds._embeddings is None  # never created
