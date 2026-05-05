"""UC-6 external readers: any Arrow-compatible engine can consume
turtlelake's Lance dataset without importing turtlelake.

Scenarios implemented: 6.1 (Polars), 6.2 (DuckDB). DataFusion (6.3) skips
at MVP since `lance-datafusion` wiring is optional. All tests skip
gracefully when the external engine isn't installed.
"""

import pytest
from pyoxigraph import Literal, NamedNode, Quad

from turtlelake import Dataset


def _seed(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    s = NamedNode("https://example.org/s")
    p = NamedNode("https://example.org/p")
    ds._append_quads(
        [
            Quad(s, p, Literal("a")),
            Quad(s, p, Literal("b")),
            Quad(s, p, NamedNode("https://example.org/o")),
        ],
        batch_size=10,
    )
    return ds


def test_6_1_polars_from_arrow_sees_every_quad(tmp_path):
    pl = pytest.importorskip("polars")
    ds = _seed(tmp_path)
    df = pl.from_arrow(ds.scan())
    assert df.height == ds.count()
    # Schema contract: the public column names from §6.1 must be present.
    for col in ("subject", "predicate", "object", "object_kind"):
        assert col in df.columns


def test_6_2_duckdb_aggregates_by_object_kind(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    ds = _seed(tmp_path)
    tbl = ds.scan()  # pa.Table — DuckDB consumes PyArrow natively
    con = duckdb.connect()
    con.register("triples", tbl)
    result = dict(
        con.sql(
            "SELECT object_kind, COUNT(*) c FROM triples GROUP BY object_kind"
        ).fetchall()
    )
    assert result["literal"] == 2
    assert result["iri"] == 1
