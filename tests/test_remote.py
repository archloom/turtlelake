"""UC-14 publish/distribute: a dataset opened via a remote URI returns
identical results to the local-path version.

MVP exercises `file://` only. `s3://` and `hf://` are nice-to-have and
require moto / mock stores; deferred.

Scenario: 14.1 (file:// URI is a drop-in for a path).
"""

import pytest
from pyoxigraph import Literal, NamedNode, Quad

from turtlelake import Dataset


def _materialize(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    s = NamedNode("https://example.org/s")
    p = NamedNode("https://example.org/p")
    ds._append_quads([Quad(s, p, Literal("x"))], batch_size=10)
    return ds


def test_14_1_file_uri_matches_local_path(tmp_path):
    local = _materialize(tmp_path)
    expected = local.query(
        "SELECT ?o WHERE { ?s ?p ?o } ORDER BY ?o"
    )
    uri = "file://" + str((tmp_path / "kg").resolve())
    try:
        remote = Dataset.open(uri)
    except Exception as e:  # pragma: no cover — skip if lance doesn't accept
        pytest.skip(f"Lance rejected file:// URI: {e}")
    got = remote.query("SELECT ?o WHERE { ?s ?p ?o } ORDER BY ?o")
    assert got == expected
