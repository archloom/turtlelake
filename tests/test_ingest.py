"""Ingest + roundtrip: parse an in-memory TTL string, write to Lance, read back,
then prove SPARQL works on the reconstructed store."""

from pyoxigraph import BlankNode, Literal, NamedNode, Quad

from turtlelake import Dataset
from turtlelake.ingest import quads_to_record_batch
from turtlelake.schema import BNODE, IRI, LITERAL


def _sample_quads():
    s = NamedNode("https://example.org/s")
    p = NamedNode("https://example.org/p")
    return [
        Quad(s, p, Literal("hello")),
        Quad(s, p, Literal("hola", language="es")),
        Quad(s, p, NamedNode("https://example.org/o")),
        Quad(s, p, BlankNode("b1")),
    ]


def test_quads_to_record_batch_preserves_kinds():
    batch = quads_to_record_batch(_sample_quads())
    kinds = batch.column("object_kind").to_pylist()
    assert kinds == [LITERAL, LITERAL, IRI, BNODE]
    langs = batch.column("object_lang").to_pylist()
    assert langs == [None, "es", None, None]


def test_dataset_roundtrip_and_sparql(tmp_path):
    ds = Dataset.open(tmp_path / "lake")
    ds._append_quads(_sample_quads(), batch_size=10)  # exercises the write path
    assert ds.count() == 4

    rows = ds.query(
        "SELECT ?o WHERE { <https://example.org/s> <https://example.org/p> ?o }"
    )
    values = sorted(r["o"]["value"] for r in rows)
    assert values == ["b1", "hello", "hola", "https://example.org/o"]


# ── UC-1.3 typed literal ─────────────────────────────────────


def test_1_3_typed_literal_preserves_datatype(tmp_path):
    ttl = tmp_path / "typed.ttl"
    ttl.write_text(
        '<https://example.org/s> <https://example.org/age> '
        '"42"^^<http://www.w3.org/2001/XMLSchema#integer> .\n',
        encoding="utf-8",
    )
    ds = Dataset.open(tmp_path / "kg")
    ds.ingest_ttl(ttl)
    tbl = ds.scan()
    dtypes = tbl.column("object_datatype").to_pylist()
    assert "http://www.w3.org/2001/XMLSchema#integer" in dtypes


# ── UC-1.5 named graphs ──────────────────────────────────────


def test_1_5_named_graphs_preserved_from_nquads(tmp_path):
    nq = tmp_path / "data.nq"
    nq.write_text(
        '<https://example.org/s> <https://example.org/p> "a" '
        '<https://example.org/g1> .\n'
        '<https://example.org/s> <https://example.org/p> "b" .\n',  # default graph
        encoding="utf-8",
    )
    ds = Dataset.open(tmp_path / "kg")
    ds.ingest_ttl(nq)
    tbl = ds.scan()
    graphs = tbl.column("graph").to_pylist()
    assert "https://example.org/g1" in graphs
    assert None in graphs
