"""UC-5 reproducibility: same tag == same SPARQL result, regardless of
what was written after the tag.

Scenarios implemented: 5.1 (same tag same results), 5.2 (post-tag writes
do not affect tagged reads).
"""

from pyoxigraph import Literal, NamedNode, Quad

from turtlelake import Dataset


def _write(ds: Dataset, value: str) -> None:
    s = NamedNode("https://example.org/s")
    p = NamedNode("https://example.org/p")
    ds._append_quads([Quad(s, p, Literal(value))], batch_size=10)


def _objects(ds: Dataset) -> list[str]:
    rows = ds.query(
        "SELECT ?o WHERE { ?s ?p ?o } ORDER BY ?o"
    )
    return [r["o"]["value"] for r in rows]


def test_5_1_same_tag_same_result_across_opens(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    _write(ds, "v1")
    _write(ds, "v2")
    ds.tag("eval-v3")

    a = Dataset.open(tmp_path / "kg", tag="eval-v3")
    b = Dataset.open(tmp_path / "kg", tag="eval-v3")
    assert _objects(a) == _objects(b) == ["v1", "v2"]


def test_5_2_post_tag_writes_dont_affect_tagged_read(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    _write(ds, "v1")
    ds.tag("frozen")
    _write(ds, "v2")  # after the tag

    snapshot = Dataset.open(tmp_path / "kg", tag="frozen")
    assert _objects(snapshot) == ["v1"]
    # The live handle sees both.
    live = Dataset.open(tmp_path / "kg")
    assert _objects(live) == ["v1", "v2"]
