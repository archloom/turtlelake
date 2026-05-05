"""Regression tests for the "cleanup" pass after the adversarial review.
Each test pins one of the remaining fixes so they don't silently regress.
"""


import pytest
from pyoxigraph import BlankNode, Literal, NamedNode, Quad

from turtlelake import Dataset
from turtlelake.ingest import quads_to_record_batch
from turtlelake.security import (
    _VERY_LONG_INPUT_THRESHOLD,
    redact_error,
    reset_rate_limits,
    scan_input,
)


@pytest.fixture(autouse=True)
def _clean():
    reset_rate_limits()
    yield
    reset_rate_limits()


# ── Prometheus metric duplicate registration ─────────────────


def test_observability_reimport_does_not_crash():
    """Simulate the pytest-xdist / IPython-autoreload case."""
    import importlib

    from turtlelake import observability
    importlib.reload(observability)  # must not raise
    metrics = observability.metrics
    metrics.tool_calls.labels(tool="x", status="success").inc()


# ── refresh() on pinned handle ────────────────────────────────


def test_refresh_refuses_to_leak_tagged_pin(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    ds._append_quads(
        [Quad(NamedNode("https://a"), NamedNode("https://p"), Literal("v"))],
        batch_size=10,
    )
    ds.tag("snap")
    snap = Dataset.open(tmp_path / "kg", tag="snap")
    with pytest.raises(RuntimeError, match="pinned handle"):
        snap.refresh()


# ── redact_error always str(exc) ──────────────────────────────


def test_redact_error_handles_non_string_exception_args():
    """Previously redact_error picked `exc.args[0]` which could itself be
    a dict, leaving its contents unredacted. Now always `str(exc)` + scan.
    Test with a tuple-arg shape whose str() preserves the redactable form."""

    class WeirdError(Exception):
        pass

    # The tuple's stringified form embeds a pattern we catch (AWS key).
    e = WeirdError(("context", "AKIAIOSFODNN7EXAMPLE"))
    out = redact_error(e)
    assert "AKIAIOSFODNN7EXAMPLE" not in out
    assert "REDACTED" in out


def test_redact_error_no_crash_on_empty_exception():
    """The old implementation indexed `args[0]` which raised `IndexError`
    on an argless exception."""

    class Bare(Exception):
        pass

    out = redact_error(Bare())
    assert isinstance(out, str)


# ── very_long_input uses len() not anchored DOTALL ────────────


def test_very_long_input_warns_even_with_newlines():
    value = "a" * (_VERY_LONG_INPUT_THRESHOLD - 1) + "\n" + "b" * 1000
    r = scan_input(value)
    assert r.safe  # warning, not block
    assert "very_long_input" in r.warnings


# ── base64_blob doesn't fire on IRIs ──────────────────────────


def test_base64_blob_does_not_false_positive_on_long_iri():
    # Realistic IRI that has a long base-ish path — should NOT warn.
    iri = "https://example.org/api/v2/resource/abc123def456ghi789jkl000xyz111"
    r = scan_input(iri)
    assert "base64_blob" not in r.warnings


def test_base64_blob_fires_on_bare_base64_run():
    bare = "A" * 120 + "=="
    r = scan_input(bare)
    assert "base64_blob" in r.warnings


# ── engine drops xsd:string for bare literals ─────────────────


def test_xsd_string_normalized_to_no_datatype(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    # Ingest a TTL with an explicit xsd:string and a plain literal.
    ttl = tmp_path / "d.ttl"
    ttl.write_text(
        '<https://ex.org/s> <https://ex.org/p> '
        '"hello"^^<http://www.w3.org/2001/XMLSchema#string> .\n',
        encoding="utf-8",
    )
    ds.ingest_ttl(ttl)
    rows = ds.query("SELECT ?o WHERE { ?s ?p ?o }")
    assert rows[0]["o"]["datatype"] is None


# ── MCP ingest path allowlist ─────────────────────────────────


def test_ingest_policy_rejects_path_outside_root(tmp_path):
    from turtlelake.mcp_server import _enforce_ingest_policy

    root = tmp_path / "allowed"
    root.mkdir()
    inside = root / "ok.ttl"
    inside.write_text("x", encoding="utf-8")
    outside = tmp_path / "escape.ttl"
    outside.write_text("x", encoding="utf-8")

    # Inside passes.
    _enforce_ingest_policy(inside, cap_bytes=10 * 1024, root=root)
    # Outside fails.
    with pytest.raises(PermissionError):
        _enforce_ingest_policy(outside, cap_bytes=10 * 1024, root=root)


# ── Bnode-named graphs preserved ──────────────────────────────


def test_bnode_graph_label_preserved_in_storage(tmp_path):
    s = NamedNode("https://ex.org/s")
    p = NamedNode("https://ex.org/p")
    g = BlankNode("g1")
    batch = quads_to_record_batch([Quad(s, p, Literal("v"), g)])
    graph_col = batch.column("graph").to_pylist()
    assert graph_col == ["_:g1"]


# ── Unknown env vars are validated, not silently ignored ────


def test_ingest_root_config_is_resolved_absolute(monkeypatch, tmp_path):
    from turtlelake.config import RuntimeConfig

    monkeypatch.setenv("TURTLELAKE_INGEST_ROOT", str(tmp_path / "root"))
    cfg = RuntimeConfig.from_env()
    assert cfg.ingest_root is not None
    assert cfg.ingest_root.is_absolute()
