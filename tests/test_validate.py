"""UC-9 SHACL validation. Requires the optional `[shacl]` extra.

Scenarios implemented: 9.1 (violating data → non-empty report), 9.2
(conforming data → empty violations), 9.3 (missing pyshacl → actionable
error).
"""

import pytest
from pyoxigraph import Literal, NamedNode, Quad

from turtlelake import Dataset


@pytest.fixture
def device_shapes(tmp_path):
    shapes = tmp_path / "shapes.ttl"
    shapes.write_text(
        """
@prefix sh:   <http://www.w3.org/ns/shacl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix ex:   <https://example.org/> .

ex:DeviceShape a sh:NodeShape ;
    sh:targetClass ex:Device ;
    sh:property [
        sh:path rdfs:label ;
        sh:minCount 1 ;
        sh:message "Every Device must have an rdfs:label." ;
    ] .
""",
        encoding="utf-8",
    )
    return shapes


def _seed_device(ds: Dataset, *, with_label: bool) -> None:
    s = NamedNode("https://example.org/d1")
    a = NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
    dev = NamedNode("https://example.org/Device")
    quads = [Quad(s, a, dev)]
    if with_label:
        label = NamedNode("http://www.w3.org/2000/01/rdf-schema#label")
        quads.append(Quad(s, label, Literal("D1")))
    ds._append_quads(quads, batch_size=10)


def test_9_1_violating_data_produces_non_empty_report(tmp_path, device_shapes):
    pytest.importorskip("pyshacl")
    pytest.importorskip("rdflib")
    ds = Dataset.open(tmp_path / "kg")
    _seed_device(ds, with_label=False)
    report = ds.validate(device_shapes)
    assert report["conforms"] is False
    assert "Every Device must have an rdfs:label" in report["report_text"]


def test_9_2_conforming_data_reports_no_violations(tmp_path, device_shapes):
    pytest.importorskip("pyshacl")
    pytest.importorskip("rdflib")
    ds = Dataset.open(tmp_path / "kg")
    _seed_device(ds, with_label=True)
    report = ds.validate(device_shapes)
    assert report["conforms"] is True


def test_9_3_missing_pyshacl_raises_actionable_error(tmp_path, monkeypatch):
    """If pyshacl isn't installed, `validate` raises RuntimeError naming
    the extra to install. We simulate by making the imports fail."""
    import sys

    ds = Dataset.open(tmp_path / "kg")
    # Seed something so `query` inside validate has a store to work with.
    s = NamedNode("https://example.org/s")
    p = NamedNode("https://example.org/p")
    ds._append_quads([Quad(s, p, Literal("v"))], batch_size=10)

    # Block both imports so validate() hits its ImportError path.
    monkeypatch.setitem(sys.modules, "pyshacl", None)
    monkeypatch.setitem(sys.modules, "rdflib", None)
    with pytest.raises(RuntimeError, match=r"turtlelake\[shacl\]"):
        ds.validate("/nonexistent/shapes.ttl")
