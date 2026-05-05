"""Checkpoint / rollback / entity-expand — the agent-facing primitives.
These three drive the MCP surface, so lock their contracts."""

import json

import pytest
from pyoxigraph import Literal, NamedNode, Quad

from turtlelake import Dataset


def _ds(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    s = NamedNode("https://example.org/device/A")
    p_label = NamedNode("http://www.w3.org/2000/01/rdf-schema#label")
    p_family = NamedNode("https://example.org/ont#family")
    fam = NamedNode("https://example.org/family/X")
    ds._append_quads(
        [
            Quad(s, p_label, Literal("A device")),
            Quad(s, p_family, fam),
            Quad(fam, p_label, Literal("Family X")),
        ],
        batch_size=10,
    )
    return ds, s, p_label, fam


def test_entity_one_hop_returns_outgoing_and_incoming(tmp_path):
    ds, s, _, fam = _ds(tmp_path)
    got = ds.entity(s.value, hops=1)
    assert got["iri"] == s.value
    predicates = {e["predicate"] for e in got["outgoing"]}
    assert "http://www.w3.org/2000/01/rdf-schema#label" in predicates
    assert "https://example.org/ont#family" in predicates
    # The family IRI is a subject elsewhere but not pointing *to* A.
    assert got["incoming"] == []


def test_entity_two_hops_follows_iri_objects(tmp_path):
    ds, s, _, fam = _ds(tmp_path)
    got = ds.entity(s.value, hops=2)
    assert "neighbors" in got
    assert fam.value in got["neighbors"]


def test_checkpoint_then_rollback_restores_count(tmp_path):
    ds, s, p, _ = _ds(tmp_path)
    baseline = ds.count()
    ds.checkpoint("before-write")
    ds._append_quads(
        [Quad(s, p, Literal("a hallucinated value"))], batch_size=10
    )
    assert ds.count() == baseline + 1
    restored = ds.rollback("before-write")
    assert restored.count() == baseline


# ── UC-2.2 incoming edges ────────────────────────────────────


def test_2_2_entity_reports_incoming_edges(tmp_path):
    ds, _s, _, fam = _ds(tmp_path)
    got = ds.entity(fam.value, hops=1)
    incoming_preds = [edge["predicate"] for edge in got["incoming"]]
    assert "https://example.org/ont#family" in incoming_preds


# ── UC-2.4 unknown IRI returns empty ─────────────────────────


def test_2_4_unknown_iri_returns_empty_shape(tmp_path):
    ds, _, _, _ = _ds(tmp_path)
    got = ds.entity("https://example.org/does-not-exist", hops=1)
    assert got["iri"] == "https://example.org/does-not-exist"
    assert got["outgoing"] == []
    assert got["incoming"] == []


# ── UC-2.6 JSON-serializable output ──────────────────────────


def test_2_6_entity_output_is_json_serializable(tmp_path):
    ds, s, _, _ = _ds(tmp_path)
    got = ds.entity(s.value, hops=2)
    # Must not raise — typed literals, nested neighbors etc. all coerce.
    json.dumps(got)


# ── UC-3.4 rollback to missing tag ───────────────────────────


def test_3_4_rollback_to_missing_tag_raises_clear_error(tmp_path):
    ds, _, _, _ = _ds(tmp_path)
    with pytest.raises(KeyError, match="No such tag 'nope'"):
        ds.rollback("nope")
