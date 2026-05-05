"""Regression tests for the discovery tools (guide + schema).

These aren't performance tests — they pin the *contract* that
agents discovering the server for the first time can self-onboard:

  1. `guide` returns a non-trivial how-to text covering the canonical
     workflow keywords (checkpoint, rollback, validate, etc.)
  2. `schema` on an empty dataset returns the expected "nothing here"
     shape, not a raise.
  3. `schema` on a populated dataset reports the classes and
     predicates that actually exist, with counts.
"""

import json

from turtlelake import Dataset


# ── guide() ──────────────────────────────────────────────────


def test_guide_mentions_every_canonical_workflow_step():
    txt = Dataset.open("/tmp/turtlelake-guide-unused-path").guide()
    assert isinstance(txt, str) and len(txt) > 200
    for keyword in (
        "schema", "entity", "sparql",
        "checkpoint", "insert", "validate",
        "rollback", "provenance", "diff",
    ):
        assert keyword in txt, f"guide() missing reference to {keyword!r}"


# ── schema() ─────────────────────────────────────────────────


def test_schema_on_empty_dataset_has_empty_shape(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    s = ds.schema()
    assert s["triples"] == 0
    assert s["classes"] == []
    assert s["predicates"] == []
    assert s["namespaces"] == []
    assert s["versions"] == 0
    assert s["tags"] == []


def test_schema_reports_classes_predicates_and_namespaces(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    ds.insert_turtle(
        "@prefix ex: <https://ex.org/> .\n"
        "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n"
        "ex:a a ex:Device ; ex:label \"A\" ; ex:supports ex:DDR5 .\n"
        "ex:b a ex:Device ; ex:label \"B\" ; ex:supports ex:DDR4 .\n"
        "ex:c a ex:Family ; ex:label \"Fam\" .\n"
    )
    s = ds.schema()
    class_iris = {c["iri"] for c in s["classes"]}
    assert "https://ex.org/Device" in class_iris
    assert "https://ex.org/Family" in class_iris
    # Class counts match reality.
    device = next(c for c in s["classes"] if c["iri"].endswith("Device"))
    assert device["count"] == 2

    pred_iris = {p["iri"] for p in s["predicates"]}
    assert "https://ex.org/label" in pred_iris
    assert "https://ex.org/supports" in pred_iris

    # Namespace histogram rolls predicates up by prefix.
    ex_ns = [n for n in s["namespaces"] if n["prefix"] == "https://ex.org/"]
    assert ex_ns and ex_ns[0]["count"] >= 3

    assert s["triples"] >= 7
    assert s["versions"] >= 1


def test_schema_is_serializable_as_mcp_json(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    ds.insert_turtle('<https://ex.org/a> <https://ex.org/p> "v" .')
    json.dumps(ds.schema())  # must not raise
