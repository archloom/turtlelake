"""UC-4 portability: tar + untar + reopen = same SPARQL results.

Scenarios implemented: 4.1 (tar round-trip), 4.3 (path with spaces).
"""

import json
import shutil
import subprocess

from pyoxigraph import Literal, NamedNode, Quad

from turtlelake import Dataset


def _materialize(tmp_path, dir_name="kg"):
    ds = Dataset.open(tmp_path / dir_name)
    s = NamedNode("https://example.org/s")
    p = NamedNode("https://example.org/p")
    ds._append_quads(
        [Quad(s, p, Literal("a")), Quad(s, p, Literal("b"))],
        batch_size=10,
    )
    return ds


def _all_quads_sorted(ds: Dataset) -> list[dict]:
    out = ds.query(
        "SELECT ?s ?p ?o WHERE { ?s ?p ?o } ORDER BY ?s ?p ?o"
    )
    return json.loads(json.dumps(out, default=str))  # normalize to JSON


def test_4_1_tar_untar_roundtrip_preserves_results(tmp_path):
    ds_a = _materialize(tmp_path, "source")
    expected = _all_quads_sorted(ds_a)

    archive = tmp_path / "kg.tar"
    subprocess.check_call(
        ["tar", "-cf", str(archive), "-C", str(tmp_path), "source"]
    )
    extract_root = tmp_path / "extracted"
    extract_root.mkdir()
    subprocess.check_call(
        ["tar", "-xf", str(archive), "-C", str(extract_root)]
    )

    ds_b = Dataset.open(extract_root / "source")
    assert _all_quads_sorted(ds_b) == expected


def test_4_3_path_with_spaces(tmp_path):
    weird = tmp_path / "some dir"
    weird.mkdir()
    ds = _materialize(weird, "my kg")
    assert ds.count() == 2
    out = ds.query(
        "SELECT ?s WHERE { ?s <https://example.org/p> ?o } LIMIT 1"
    )
    assert len(out) == 1


def test_4_2_copytree_roundtrip_preserves_provenance(tmp_path):
    """Bonus: cp -r preserves provenance.jsonl (NFR-11)."""
    ds_a = Dataset.open(tmp_path / "source")
    ttl = tmp_path / "data.ttl"
    ttl.write_text(
        '<https://example.org/s> <https://example.org/p> "x" .\n',
        encoding="utf-8",
    )
    ds_a.ingest_ttl(ttl, source="unit", author="alice")
    shutil.copytree(tmp_path / "source", tmp_path / "copy")
    ds_b = Dataset.open(tmp_path / "copy")
    assert ds_b.provenance() == ds_a.provenance()
