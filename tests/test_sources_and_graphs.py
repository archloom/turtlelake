"""UC-M scenarios: layered mode with external TTL sources + named graphs.

Pins the contract for the new design:
  - Attached sources are NOT copied into Lance.
  - Sources are mtime-watched; cache rebuilds on change.
  - Writes default to `turtlelake://agent-overlay` when sources exist.
  - `GRAPH <uri> { ... }` SPARQL reveals the split.
  - `dump` round-trips only the overlay.
"""

import time

from turtlelake import Dataset


VENDOR_TTL = """@prefix alt: <https://example.org/vendor#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

alt:A5E_220B a alt:Device ;
    rdfs:label "Acme Widget Series 220B" ;
    alt:supports alt:DDR5 .
"""

VENDOR_GRAPH = "https://example.org/graphs/vendor"
AGENT_GRAPH = "turtlelake://agent-overlay"


# ── UC-M1: sources attach; data visible without being copied ──────


def test_m1_source_attached_is_queryable_without_lance_write(tmp_path):
    vendor = tmp_path / "vendor.ttl"
    vendor.write_text(VENDOR_TTL, encoding="utf-8")
    kg = Dataset.open(tmp_path / "kg", sources={VENDOR_GRAPH: vendor})
    # No Lance overlay yet, but the source triples are queryable.
    rows = kg.query(
        "SELECT ?label WHERE { ?d <http://www.w3.org/2000/01/rdf-schema#label> ?label }"
    )
    labels = {r["label"]["value"] for r in rows}
    assert "Acme Widget Series 220B" in labels
    # The Lance overlay was NOT created on disk just from opening sources.
    assert not kg.triples_path.exists()


def test_m1_source_in_its_own_named_graph(tmp_path):
    vendor = tmp_path / "vendor.ttl"
    vendor.write_text(VENDOR_TTL, encoding="utf-8")
    kg = Dataset.open(tmp_path / "kg", sources={VENDOR_GRAPH: vendor})
    rows = kg.query(
        f"SELECT ?label WHERE {{ GRAPH <{VENDOR_GRAPH}> "
        "{ ?d <http://www.w3.org/2000/01/rdf-schema#label> ?label } }"
    )
    assert [r["label"]["value"] for r in rows] == ["Acme Widget Series 220B"]


# ── UC-M2: agent writes auto-route to agent-overlay ──────────────


def test_m2_insert_with_sources_goes_to_agent_overlay(tmp_path):
    vendor = tmp_path / "vendor.ttl"
    vendor.write_text(VENDOR_TTL, encoding="utf-8")
    kg = Dataset.open(tmp_path / "kg", sources={VENDOR_GRAPH: vendor})
    kg.insert_turtle(
        '<https://ex.org/customer/1> '
        '<https://example.org/vendor#matches> '
        '<https://example.org/vendor#ER_A5_001> .'
    )
    # The agent's triple is in the overlay graph, NOT in the vendor graph.
    vendor_rows = kg.query(
        f"SELECT ?s ?p ?o WHERE {{ GRAPH <{VENDOR_GRAPH}> {{ ?s ?p ?o }} }}"
    )
    vendor_subjects = {r["s"]["value"] for r in vendor_rows}
    assert "https://ex.org/customer/1" not in vendor_subjects

    overlay_rows = kg.query(
        f"SELECT ?s WHERE {{ GRAPH <{AGENT_GRAPH}> {{ ?s ?p ?o }} }}"
    )
    overlay_subjects = {r["s"]["value"] for r in overlay_rows}
    assert "https://ex.org/customer/1" in overlay_subjects


def test_m2_insert_without_sources_uses_default_graph(tmp_path):
    """Preserves the old behavior: no sources → default graph."""
    kg = Dataset.open(tmp_path / "kg")  # no sources
    kg.insert_turtle('<https://ex.org/a> <https://ex.org/p> "v" .')
    # GRAPH <turtlelake://agent-overlay> should be empty.
    in_overlay = kg.query(
        f"ASK {{ GRAPH <{AGENT_GRAPH}> {{ ?s ?p ?o }} }}"
    )
    assert in_overlay is False
    # Default graph has the triple.
    default = kg.query(
        "SELECT ?o WHERE { ?s ?p ?o }"
    )
    assert len(default) == 1


# ── UC-M3: rollback on overlay does not touch sources ───────────


def test_m3_rollback_only_affects_overlay(tmp_path):
    vendor = tmp_path / "vendor.ttl"
    vendor.write_text(VENDOR_TTL, encoding="utf-8")
    kg = Dataset.open(tmp_path / "kg", sources={VENDOR_GRAPH: vendor})
    kg.insert_turtle('<https://ex.org/x> <https://ex.org/p> "x" .')
    kg.checkpoint("pre")
    kg.insert_turtle('<https://ex.org/y> <https://ex.org/p> "bad" .')
    assert kg.count() == 2
    kg.rollback("pre")
    # After rollback: one overlay quad, vendor triples still visible.
    assert kg.count() == 1
    labels = {
        r["label"]["value"]
        for r in kg.query(
            "SELECT ?label WHERE { ?d <http://www.w3.org/2000/01/rdf-schema#label> ?label }"
        )
    }
    assert "Acme Widget Series 220B" in labels
    # Vendor file was never touched.
    assert vendor.read_text(encoding="utf-8") == VENDOR_TTL


# ── UC-M4: source mtime change triggers cache rebuild ───────────


def test_m4_source_mtime_change_picked_up(tmp_path):
    vendor = tmp_path / "vendor.ttl"
    vendor.write_text(VENDOR_TTL, encoding="utf-8")
    kg = Dataset.open(tmp_path / "kg", sources={VENDOR_GRAPH: vendor})
    before = kg.query("ASK { ?s <https://example.org/vendor#newpredicate> ?o }")
    assert before is False

    # Edit the vendor file to add a new fact; bump mtime explicitly in
    # case filesystem mtime resolution is coarse.
    time.sleep(0.02)
    vendor.write_text(
        VENDOR_TTL
        + '<https://example.org/vendor#A5E_220B> '
        '<https://example.org/vendor#newpredicate> "hello" .\n',
        encoding="utf-8",
    )
    import os
    os.utime(vendor, (time.time() + 1, time.time() + 1))

    # Next query picks up the change automatically.
    after = kg.query("ASK { ?s <https://example.org/vendor#newpredicate> ?o }")
    assert after is True


# ── UC-M5: sources() + schema() expose the attached sources ─────


def test_m5_sources_exposes_metadata(tmp_path):
    vendor = tmp_path / "vendor.ttl"
    vendor.write_text(VENDOR_TTL, encoding="utf-8")
    kg = Dataset.open(tmp_path / "kg", sources={VENDOR_GRAPH: vendor})
    s_list = kg.sources()
    assert len(s_list) == 1
    assert s_list[0]["graph"] == VENDOR_GRAPH
    assert s_list[0]["path"] == str(vendor)
    assert s_list[0]["sha256"]

    schema = kg.schema()
    assert schema["sources"] == s_list


# ── UC-M6: dump the overlay back to TTL ─────────────────────────


def test_m6_dump_agent_overlay_only(tmp_path):
    vendor = tmp_path / "vendor.ttl"
    vendor.write_text(VENDOR_TTL, encoding="utf-8")
    kg = Dataset.open(tmp_path / "kg", sources={VENDOR_GRAPH: vendor})
    kg.insert_turtle('<https://ex.org/inf/1> <https://ex.org/p> "agent-wrote" .')
    out = tmp_path / "out.ttl"
    n = kg.dump(out, format="turtle", graph=AGENT_GRAPH)
    assert n == 1
    text = out.read_text(encoding="utf-8")
    # The agent's fact is there; vendor content is not.
    assert "agent-wrote" in text
    assert "Widget" not in text


def test_m6_dump_with_no_lance_raises_clean_error(tmp_path):
    vendor = tmp_path / "vendor.ttl"
    vendor.write_text(VENDOR_TTL, encoding="utf-8")
    kg = Dataset.open(tmp_path / "kg", sources={VENDOR_GRAPH: vendor})
    import pytest
    with pytest.raises(RuntimeError, match="Nothing to dump"):
        kg.dump(tmp_path / "wont-exist.ttl")


# ── UC-M7: owl:imports ──────────────────────────────────────────


def test_m7_follow_imports_transitive(tmp_path):
    # Root ontology imports "extension.ttl" (same directory).
    ext = tmp_path / "extension.ttl"
    ext.write_text(
        '@prefix alt: <https://example.org/vendor#> .\n'
        'alt:Extension a alt:Concept .\n',
        encoding="utf-8",
    )
    root = tmp_path / "root.ttl"
    root.write_text(
        '@prefix owl: <http://www.w3.org/2002/07/owl#> .\n'
        '<https://example.org/vendor/root> a owl:Ontology ;\n'
        '    owl:imports <extension.ttl> .\n',
        encoding="utf-8",
    )
    kg = Dataset.open(
        tmp_path / "kg",
        sources={"https://example.org/vendor/root": root},
        follow_imports=True,
    )
    # The imported extension's triple is visible via query.
    rows = kg.query(
        "SELECT ?s WHERE { ?s a <https://example.org/vendor#Concept> }"
    )
    assert [r["s"]["value"] for r in rows] == [
        "https://example.org/vendor#Extension"
    ]
    # sources() reports both root and the resolved import.
    s_paths = {s["path"] for s in kg.sources()}
    assert str(root) in s_paths
    assert str(ext.resolve()) in s_paths
