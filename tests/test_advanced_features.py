"""Regression tests for the advanced-feature batch:

  - Dataset.explain(sparql) returns a usable plan sketch
  - Query timeout aborts long-running queries
  - save_query / run_saved / stored_queries catalog + substitutions
  - sources= accepts http(s):// URLs with local ETag cache

These pin the contracts the MCP tools wrap, so we don't need separate
MCP-stdio coverage for each.
"""

import http.server
import socketserver
import threading
from contextlib import contextmanager

import pytest

from turtlelake import Dataset
from turtlelake.engine import QueryTimeout


# ── explain() ────────────────────────────────────────────────


def test_explain_returns_plan_sketch(tmp_path):
    kg = Dataset.open(tmp_path / "kg")
    kg.insert_turtle('<https://ex.org/a> <https://ex.org/p> "v" .')
    plan = kg.explain(
        "SELECT ?o WHERE { ?s <https://ex.org/p> ?o }"
    )
    assert isinstance(plan, str)
    assert "store size" in plan
    assert "ex.org/p" in plan


def test_explain_handles_complex_query(tmp_path):
    kg = Dataset.open(tmp_path / "kg")
    kg.insert_turtle(
        '@prefix ex: <https://ex.org/> .\n'
        'ex:a a ex:T ; ex:label "hi" .'
    )
    plan = kg.explain(
        "PREFIX ex: <https://ex.org/> SELECT ?l WHERE { ?s a ex:T ; ex:label ?l } LIMIT 5"
    )
    # Multiple triple patterns recognized
    assert "a ex:T" in plan or "ex:T" in plan


# ── query timeout ────────────────────────────────────────────


def test_query_timeout_raises_cleanly(tmp_path):
    kg = Dataset.open(tmp_path / "kg")
    # Seed enough rows that an unbounded cross-product is measurable.
    n = 2000
    lines = ["@prefix ex: <https://ex.org/> ."]
    for i in range(n):
        lines.append(f'ex:s{i} ex:p ex:o{i} .')
    ttl = tmp_path / "seed.ttl"
    ttl.write_text("\n".join(lines), encoding="utf-8")
    kg.ingest_ttl(ttl)
    # Pure Cartesian: 2000^3 = 8e9 intermediate rows; seconds at least.
    bad = (
        "SELECT ?a ?b ?c WHERE { "
        "?a ?p1 ?o1 . ?b ?p2 ?o2 . ?c ?p3 ?o3 "
        "}"
    )
    with pytest.raises(QueryTimeout):
        kg.query(bad, timeout_ms=1)


def test_query_timeout_does_not_fire_when_fast_enough(tmp_path):
    kg = Dataset.open(tmp_path / "kg")
    kg.insert_turtle('<https://ex.org/a> <https://ex.org/p> "v" .')
    rows = kg.query("SELECT ?o WHERE { ?s ?p ?o }", timeout_ms=5000)
    assert rows


# ── stored queries ───────────────────────────────────────────


def test_save_and_run_saved_with_bindings(tmp_path):
    kg = Dataset.open(tmp_path / "kg")
    kg.insert_turtle(
        '@prefix ex: <https://ex.org/> .\n'
        'ex:a ex:family ex:F1 .\n'
        'ex:b ex:family ex:F2 .\n'
    )
    kg.save_query(
        "by-family",
        "SELECT ?s WHERE { ?s <https://ex.org/family> ?fam }",
        description="subjects in a family",
    )
    rows = kg.run_saved("by-family", bindings={"fam": "https://ex.org/F1"})
    assert [r["s"]["value"] for r in rows] == ["https://ex.org/a"]


def test_stored_queries_persists_across_reopen(tmp_path):
    kg = Dataset.open(tmp_path / "kg")
    kg.insert_turtle('<https://ex.org/a> <https://ex.org/p> "1" .')
    kg.save_query("all", "SELECT ?o WHERE { ?s ?p ?o }")
    # New handle on same directory sees the catalog.
    kg2 = Dataset.open(tmp_path / "kg")
    catalog = kg2.stored_queries()
    assert "all" in catalog
    assert catalog["all"]["sparql"].startswith("SELECT")


def test_run_saved_unknown_name_clear_error(tmp_path):
    kg = Dataset.open(tmp_path / "kg")
    kg.insert_turtle('<https://ex.org/a> <https://ex.org/p> "1" .')
    with pytest.raises(KeyError, match="No saved query"):
        kg.run_saved("does-not-exist")


def test_save_query_rejects_bad_names(tmp_path):
    kg = Dataset.open(tmp_path / "kg")
    with pytest.raises(ValueError, match="alphanumeric"):
        kg.save_query("hey there!", "SELECT * WHERE { ?s ?p ?o }")


# ── HTTP sources (local mock) ────────────────────────────────


@contextmanager
def _serve(ttl_bytes: bytes):
    """Spin up a tiny HTTP server on a random port serving the TTL once."""
    payload = ttl_bytes

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.headers.get("If-None-Match") == '"v1"':
                self.send_response(304)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/turtle")
            self.send_header("ETag", '"v1"')
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *a, **k):
            return  # stay quiet in tests

    with socketserver.TCPServer(("127.0.0.1", 0), Handler) as httpd:
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{httpd.server_address[1]}/vendor.ttl"
        finally:
            httpd.shutdown()
            thread.join(timeout=2)


def test_http_source_fetched_and_queryable(tmp_path):
    ttl = (
        b"@prefix ex: <https://ex.org/> .\n"
        b'ex:remote a ex:Thing ; ex:label "from-http" .\n'
    )
    with _serve(ttl) as url:
        kg = Dataset.open(
            tmp_path / "kg",
            sources={"https://ex.org/graphs/remote": url},
        )
        rows = kg.query(
            'SELECT ?l WHERE { ?s <https://ex.org/label> ?l }'
        )
        assert [r["l"]["value"] for r in rows] == ["from-http"]


def test_http_source_caches_under_source_cache_dir(tmp_path):
    ttl = b'<https://ex.org/a> <https://ex.org/p> "1" .\n'
    with _serve(ttl) as url:
        kg = Dataset.open(
            tmp_path / "kg",
            sources={"https://ex.org/remote": url},
        )
        cache = kg.path / "_source_cache"
        assert cache.exists()
        # At least the ttl + etag file should be present.
        files = {f.name for f in cache.iterdir()}
        assert any(f.endswith(".ttl") for f in files)
        assert any(f.endswith(".etag") for f in files)


def test_http_source_offline_falls_back_to_cache(tmp_path):
    ttl = b'<https://ex.org/a> <https://ex.org/p> "1" .\n'
    remote_graph = "https://ex.org/graphs/remote"
    with _serve(ttl) as url:
        # First open populates the cache.
        Dataset.open(tmp_path / "kg", sources={remote_graph: url})

    # Server is now down (context exited). A second open should still
    # succeed using the cached file — "offline-tolerant federation".
    kg2 = Dataset.open(tmp_path / "kg", sources={remote_graph: url})
    rows = kg2.query("SELECT ?o WHERE { ?s ?p ?o }")
    assert [r["o"]["value"] for r in rows] == ["1"]
