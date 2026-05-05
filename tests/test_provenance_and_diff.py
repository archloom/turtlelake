"""Covers UC-10 (provenance) and UC-12 (diff) from TEST_SCENARIOS.md.
Scenarios implemented: 10.1, 10.2, 10.4, 12.1, 12.2, 12.3, 12.4."""

import pytest
from pyoxigraph import Literal, NamedNode, Quad

from turtlelake import Dataset


def _seed(ds: Dataset, n: int = 2) -> list[Quad]:
    s = NamedNode("https://example.org/s")
    p = NamedNode("https://example.org/p")
    quads = [Quad(s, p, Literal(f"v{i}")) for i in range(n)]
    ds._append_quads(quads, batch_size=10)
    return quads


# ---- UC-10: provenance ---------------------------------------------------


def test_10_1_ingest_records_source_and_author(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    ttl = tmp_path / "data.ttl"
    ttl.write_text(
        '<https://example.org/s> <https://example.org/p> "hi" .\n',
        encoding="utf-8",
    )
    ds.ingest_ttl(ttl, source="unit-test", author="alice")
    log = ds.provenance()
    assert len(log) == 1
    assert log[0]["source"] == "unit-test"
    assert log[0]["author"] == "alice"
    assert log[0]["kind"] == "ingest_ttl"
    assert log[0]["row_delta"] == 1


def test_10_2_mixed_sequence_ordered_provenance(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    ttl = tmp_path / "a.ttl"
    ttl.write_text(
        '<https://example.org/a> <https://example.org/p> "x" .\n',
        encoding="utf-8",
    )
    ds.ingest_ttl(ttl, source="a", author="u")
    ds.checkpoint("mid", author="u")
    ds.ingest_ttl(ttl, source="b", author="u")
    log = ds.provenance()
    assert [r["kind"] for r in log] == ["ingest_ttl", "checkpoint", "ingest_ttl"]
    assert [r["source"] for r in log] == ["a", "checkpoint:mid", "b"]


# ---- UC-12: diff ---------------------------------------------------------


def test_12_1_pure_append_diff_returns_only_added(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    _seed(ds, n=2)
    v_old = ds._lance.version
    _seed(ds, n=3)  # different literal values so we actually add 3 new rows
    v_new = ds._lance.version
    d = ds.diff(v_old, v_new)
    # v_old literals were v0,v1; v_new adds v0,v1,v2 but since v0/v1 already
    # exist they are added rows again (duplicate quads are valid in Lance —
    # we dedup by tuple). The truly *new* set equals {v2}.
    added_objs = {q["object"] for q in d["added"]}
    removed_objs = {q["object"] for q in d["removed"]}
    assert "v2" in added_objs
    assert removed_objs == set()


def test_12_2_identical_versions_diff_is_empty(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    _seed(ds, n=1)
    v = ds._lance.version
    d = ds.diff(v, v)
    assert d == {"added": [], "removed": []}


# ── UC-12.3 set-difference cardinality ────────────────────────


def test_12_3_diff_cardinality_matches_set_difference(tmp_path):
    """Distinct literal values go in; diff returns the right set cardinality."""
    ds = Dataset.open(tmp_path / "kg")
    s = NamedNode("https://example.org/s")
    p = NamedNode("https://example.org/p")
    ds._append_quads([Quad(s, p, Literal(f"v{i}")) for i in range(3)], batch_size=10)
    v_old = ds._lance.version
    ds._append_quads([Quad(s, p, Literal(f"w{i}")) for i in range(5)], batch_size=10)
    v_new = ds._lance.version
    d = ds.diff(v_old, v_new)
    added_objs = {q["object"] for q in d["added"]}
    assert added_objs == {f"w{i}" for i in range(5)}
    assert d["removed"] == []


# ── UC-12.4 out-of-range version → clear error ───────────────


def test_12_4_out_of_range_version_raises_clear_error(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    _seed(ds, n=1)
    # Lance raises for missing versions; we let it propagate (the error path
    # is rare and the message is descriptive). Tighten this later if Lance
    # exposes a typed exception we can catch precisely.
    with pytest.raises(Exception):  # noqa: B017
        ds.diff(9999, 10000)


# ── UC-10.4 default source when omitted ──────────────────────


def test_10_4_default_source_is_filename(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    ttl = tmp_path / "facts.ttl"
    ttl.write_text(
        '<https://example.org/s> <https://example.org/p> "v" .\n',
        encoding="utf-8",
    )
    ds.ingest_ttl(ttl)  # no source=, no author=
    log = ds.provenance()
    assert log[0]["source"] == "facts.ttl"
    assert log[0]["author"]  # whatever is in $USER or "unknown"
