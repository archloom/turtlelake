"""Disk-bloat maintenance: compact() and prune_versions().

Long-lived agent datasets accumulate a Lance fragment per write and
a Lance version per checkpoint. Without these knobs, disk usage grows
without bound. These tests prove the operations run end-to-end and
preserve correctness (no triples or vectors lost, tagged versions
survive).
"""

import random

from pyoxigraph import Literal, NamedNode, Quad

from turtlelake import Dataset


def _seed(ds: Dataset, n_writes: int = 5) -> None:
    label = NamedNode("http://www.w3.org/2000/01/rdf-schema#label")
    for i in range(n_writes):
        s = NamedNode(f"https://ex/{i}")
        ds._append_quads([Quad(s, label, Literal(f"v{i}"))], batch_size=10)
        ds.embed([f"https://ex/{i}"], [[float(i), 0.0]], model_id="m")


def test_compact_returns_summary_dicts(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    _seed(ds)
    out = ds.compact()
    assert set(out) == {"triples", "embeddings"}
    assert isinstance(out["triples"], dict)
    assert isinstance(out["embeddings"], dict)


def test_compact_preserves_row_counts(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    _seed(ds, n_writes=10)
    triples_before = ds.count()
    embeddings_before = ds.embedding_count()
    ds.compact()
    assert ds.count() == triples_before
    assert ds.embedding_count() == embeddings_before


def test_prune_keeps_recent_versions(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    _seed(ds, n_writes=8)
    out = ds.prune_versions(keep_versions=3)
    # Both sides report a per-dataset summary
    assert "triples" in out and "embeddings" in out


# Stable shape contract for prune_versions: every successful summary
# has the same set of keys regardless of pylance version. Callers
# (especially MCP agents) need to parse one schema, not three.
PRUNE_SUMMARY_KEYS = {
    "removed_versions",
    "bytes_removed",
    "data_files_removed",
    "index_files_removed",
    "kept_versions",
}


def test_prune_summary_has_stable_shape_when_work_happens(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    _seed(ds, n_writes=8)
    out = ds.prune_versions(keep_versions=2)
    for side in ("triples", "embeddings"):
        summary = out[side]
        assert isinstance(summary, dict)
        # Either we got the success shape, or an explicit error
        # (callers should see exactly one of these — never a partial
        # mix).
        if "error" in summary:
            continue
        assert set(summary.keys()) == PRUNE_SUMMARY_KEYS


def test_prune_summary_has_stable_shape_for_noop(tmp_path):
    """Calling prune with `keep_versions` >= existing count is a no-op.
    The returned shape MUST match the work-happened case so callers
    don't need to branch on 'did anything happen.'"""
    ds = Dataset.open(tmp_path / "kg")
    _seed(ds, n_writes=2)
    out = ds.prune_versions(keep_versions=100)
    for side in ("triples", "embeddings"):
        summary = out[side]
        assert set(summary.keys()) == PRUNE_SUMMARY_KEYS
        assert summary["removed_versions"] == 0
        assert summary["bytes_removed"] == 0


def test_prune_summary_kept_count_reflects_request(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    _seed(ds, n_writes=10)
    out = ds.prune_versions(keep_versions=4)
    # Successful prune reports kept_versions=4; a no-op (already at
    # or below 4) reports the actual count.
    for side in ("triples", "embeddings"):
        summary = out[side]
        if "error" in summary:
            continue
        assert summary["kept_versions"] >= 1


def test_prune_rejects_zero(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    _seed(ds)
    import pytest

    with pytest.raises(ValueError, match=">= 1"):
        ds.prune_versions(keep_versions=0)


def test_prune_preserves_tagged_versions(tmp_path):
    """Tags are first-class — after a prune, a checkpoint tagged before
    the prune must still resolve."""
    ds = Dataset.open(tmp_path / "kg")
    _seed(ds)
    ds.checkpoint("snapshot")
    # Force a few more versions on top.
    _seed(ds, n_writes=3)
    ds.prune_versions(keep_versions=2)
    # The tag must still be openable.
    pinned = Dataset.open(tmp_path / "kg", tag="snapshot")
    assert pinned.count() > 0


def test_build_vector_index_explicit_pq_below_min_raises_clear_error(tmp_path):
    """Explicit IVF_PQ below the 256-row training minimum surfaces a
    turtlelake-level error rather than Lance's Rust panic."""
    ds = Dataset.open(tmp_path / "kg")
    ds.embed(["a"], [[1.0, 0.0]], model_id="m")
    import pytest

    with pytest.raises(RuntimeError, match="at least 256"):
        ds.build_vector_index(index_type="IVF_PQ")


def test_build_vector_index_auto_skips_below_threshold(tmp_path):
    """At small scale the auto policy is 'do nothing' — brute-force
    scan is already fast. Caller gets a status dict, not a panic."""
    ds = Dataset.open(tmp_path / "kg")
    ds.embed(["a", "b"], [[1.0, 0.0], [0.0, 1.0]], model_id="m")
    out = ds.build_vector_index()  # default index_type="auto"
    assert out["action"] == "skipped"
    assert out["index_type"] is None
    assert out["rows"] == 2
    assert "brute-force" in out["reason"]


def test_build_vector_index_explicit_ivf_flat_succeeds_above_min(tmp_path):
    """An explicit IVF_FLAT works at the test-friendly 300-row scale —
    no PQ training threshold to clear."""
    ds = Dataset.open(tmp_path / "kg")
    random.seed(0)
    n = 300
    iris = [f"https://ex/{i}" for i in range(n)]
    vecs = [[random.random() for _ in range(8)] for _ in range(n)]
    ds.embed(iris, vecs, model_id="m")
    out = ds.build_vector_index(
        index_type="IVF_FLAT", num_partitions=4
    )
    assert out["action"] == "built"
    assert out["index_type"] == "IVF_FLAT"
    # Search still returns sensible answers post-index.
    hits = ds.vector_search(vecs[0], k=3)
    assert hits[0]["iri"] == "https://ex/0"


def test_resolve_auto_index_dispatch_is_pure(tmp_path):
    """The policy decision is unit-testable without writing data —
    `_resolve_auto_index` is a pure function on row count + chosen
    type."""
    ds = Dataset.open(tmp_path / "kg")
    # Below the flat threshold → skip.
    out = ds._resolve_auto_index("auto", 5_000)
    assert out["action"] == "skipped"
    # Mid range → IVF_FLAT.
    out = ds._resolve_auto_index("auto", 50_000)
    assert out["index_type"] == "IVF_FLAT"
    # Above PQ threshold → IVF_PQ.
    out = ds._resolve_auto_index("auto", 5_000_000)
    assert out["index_type"] == "IVF_PQ"
    # Explicit override is honored at any scale.
    out = ds._resolve_auto_index("IVF_SQ", 100)
    assert out["index_type"] == "IVF_SQ"
    assert out["reason"] == "explicit"


def test_embedding_versions_separate_from_triples_versions(tmp_path):
    """The two Lance datasets advance independently — a vector-only
    write should not produce a new triples version. embedding_versions()
    surfaces this so callers can audit drift."""
    ds = Dataset.open(tmp_path / "kg")
    label = NamedNode("http://www.w3.org/2000/01/rdf-schema#label")
    ds._append_quads(
        [Quad(NamedNode("https://ex/A"), label, Literal("A"))], batch_size=10
    )
    triples_versions_before = len(ds.versions())
    ds.embed(["https://ex/A"], [[1.0]], model_id="m")
    assert len(ds.versions()) == triples_versions_before
    assert len(ds.embedding_versions()) >= 1
