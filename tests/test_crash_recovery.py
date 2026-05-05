"""Crash-safe paired checkpoint via the manifest write-ahead log.

The pitch says "checkpoint() and rollback() are atomic across triples
and embeddings." Implementation: we record `pending_checkpoint` in
manifest.json BEFORE creating either Lance tag, and clear it after
both succeed. If the process dies between the two creates, the next
`Dataset.open(path)` reconciles the partial state. These tests
simulate that crash by writing the manifest record manually and then
asserting that opening recovers correctly.
"""

import json

from pyoxigraph import Literal, NamedNode, Quad

from turtlelake import Dataset


def _seed(ds: Dataset) -> None:
    label = NamedNode("http://www.w3.org/2000/01/rdf-schema#label")
    a = NamedNode("https://ex.org/A")
    ds._append_quads([Quad(a, label, Literal("A"))], batch_size=10)
    ds.embed(["https://ex.org/A"], [[1.0, 0.0]], model_id="m")


def test_checkpoint_clears_pending_marker_on_success(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    _seed(ds)
    ds.checkpoint("ok")
    manifest = json.loads((tmp_path / "kg" / "manifest.json").read_text())
    assert "pending_checkpoint" not in manifest
    assert "ok" in ds.tags()
    assert "ok" in list(ds._embeddings.tags.list())


def test_recovery_forward_rolls_when_only_triples_tagged(tmp_path):
    """Simulate a crash AFTER the triples tag was created but BEFORE
    the embeddings tag. Recovery on next open must create the missing
    embeddings tag at the recorded version."""
    ds = Dataset.open(tmp_path / "kg")
    _seed(ds)
    triples_v = ds.version
    emb_v = ds._embeddings.version

    # Manually re-create the on-disk state of a partial checkpoint:
    # - pending_checkpoint marker present
    # - triples tag created
    # - embeddings tag absent
    ds._lance.tags.create("torn", triples_v)
    manifest = ds._read_manifest()
    manifest["pending_checkpoint"] = {
        "name": "torn",
        "triples_version": triples_v,
        "embeddings_version": emb_v,
    }
    ds._write_manifest_atomic(manifest)
    assert "torn" not in list(ds._embeddings.tags.list())

    # Drop the handle and re-open from scratch — this is what the next
    # process / restart would do.
    del ds
    again = Dataset.open(tmp_path / "kg")

    assert "torn" in again.tags()
    assert "torn" in list(again._embeddings.tags.list())
    manifest = json.loads((tmp_path / "kg" / "manifest.json").read_text())
    assert "pending_checkpoint" not in manifest


def test_recovery_forward_rolls_when_only_embeddings_tagged(tmp_path):
    """Inverse partial state: embeddings tag exists but triples tag
    does not. Shouldn't happen given our normal call order, but the
    recovery path handles it defensively so a future refactor (or a
    third-party caller that tags directly) cannot leave an orphan
    embeddings tag with no triples-side counterpart."""
    ds = Dataset.open(tmp_path / "kg")
    _seed(ds)
    triples_v = ds.version
    emb_v = ds._embeddings.version

    # Set up the inverse partial state on disk.
    ds._embeddings.tags.create("torn-rev", emb_v)
    manifest = ds._read_manifest()
    manifest["pending_checkpoint"] = {
        "name": "torn-rev",
        "triples_version": triples_v,
        "embeddings_version": emb_v,
    }
    ds._write_manifest_atomic(manifest)
    assert "torn-rev" not in ds.tags()

    del ds
    again = Dataset.open(tmp_path / "kg")

    # Both tags now exist; pending marker cleared.
    assert "torn-rev" in again.tags()
    assert "torn-rev" in list(again._embeddings.tags.list())
    manifest = json.loads((tmp_path / "kg" / "manifest.json").read_text())
    assert "pending_checkpoint" not in manifest


def test_recovery_clears_marker_when_neither_tag_was_created(tmp_path):
    """Crash before either tag — pending marker exists alone. Open
    should clear the marker without inventing a tag."""
    ds = Dataset.open(tmp_path / "kg")
    _seed(ds)
    triples_v = ds.version
    emb_v = ds._embeddings.version
    manifest = ds._read_manifest()
    manifest["pending_checkpoint"] = {
        "name": "ghost",
        "triples_version": triples_v,
        "embeddings_version": emb_v,
    }
    ds._write_manifest_atomic(manifest)

    del ds
    again = Dataset.open(tmp_path / "kg")
    assert "ghost" not in again.tags()
    manifest = json.loads((tmp_path / "kg" / "manifest.json").read_text())
    assert "pending_checkpoint" not in manifest


def test_torn_manifest_is_tolerated_on_open(tmp_path):
    """A previous writer that crashed mid-write could leave the
    manifest as invalid JSON. Open must not crash; embedding_dim()
    falls back to None and the next embed re-records the dim."""
    (tmp_path / "kg").mkdir(parents=True)
    (tmp_path / "kg" / "manifest.json").write_text("{ this is not json")
    ds = Dataset.open(tmp_path / "kg")
    assert ds.embedding_dim() is None
    ds.embed(["a"], [[1.0, 2.0]], model_id="m")
    assert ds.embedding_dim() == 2


def test_manifest_write_is_atomic_via_tmp_rename(tmp_path):
    """We use tmp + os.replace, so a partial write never leaves a
    half-written manifest.json. Direct check: after one
    `_write_manifest_atomic`, no `.json.tmp` sibling remains."""
    ds = Dataset.open(tmp_path / "kg")
    ds._write_manifest_atomic({"hello": "world"})
    files = sorted(p.name for p in (tmp_path / "kg").iterdir())
    assert "manifest.json" in files
    assert all(not f.endswith(".json.tmp") for f in files)
