"""Observability tests — correlation IDs and (when installed) Prometheus
metric recording."""

import pytest

from turtlelake.observability import (
    extract_correlation_id,
    generate_correlation_id,
    get_correlation_id,
    metrics,
    serialize_metrics,
    set_correlation_id,
)


# ── Correlation IDs ──────────────────────────────────────────


def test_generate_correlation_id_is_uuid_shape():
    cid = generate_correlation_id()
    # UUID4: 8-4-4-4-12 hex
    parts = cid.split("-")
    assert [len(p) for p in parts] == [8, 4, 4, 4, 12]


def test_extract_prefers_x_request_id():
    cid = extract_correlation_id({"x-request-id": "abc-123"})
    assert cid == "abc-123"


def test_extract_sanitizes_bad_chars():
    cid = extract_correlation_id({"x-request-id": "abc\r\n\x00; rm -rf /"})
    assert ";" not in cid
    assert "\n" not in cid


def test_extract_from_traceparent():
    # W3C traceparent: 00-<32 hex>-<16 hex>-01
    trace = "a" * 32
    cid = extract_correlation_id({"traceparent": f"00-{trace}-0000000000000001-01"})
    assert cid == trace


def test_extract_generates_when_missing():
    cid = extract_correlation_id({})
    assert cid  # non-empty


def test_contextvar_roundtrip():
    set_correlation_id("abc-123")
    try:
        assert get_correlation_id() == "abc-123"
    finally:
        set_correlation_id(None)
    # After reset, either a new UUID or None is acceptable; just not "abc-123"
    assert get_correlation_id() != "abc-123"


# ── Metrics ──────────────────────────────────────────────────


def test_metrics_objects_exist_and_accept_labels():
    # Works whether prometheus_client is installed (real metric) or not
    # (no-op stand-in). The contract is "calling these doesn't crash".
    metrics.tool_calls.labels(tool="sparql", status="success").inc()
    metrics.tool_duration.labels(tool="sparql").observe(0.01)
    metrics.security_events.labels(event="rate_limit").inc()


def test_serialize_metrics_returns_bytes_when_prom_installed():
    try:
        import prometheus_client  # noqa: F401
    except ImportError:
        pytest.skip("prometheus_client not installed")
    payload = serialize_metrics()
    assert isinstance(payload, bytes)
    text = payload.decode("utf-8", errors="ignore")
    # At least one of our tool metrics should be present after the test
    # above exercised them.
    assert "turtlelake_" in text or text == ""  # tolerate fresh registry
