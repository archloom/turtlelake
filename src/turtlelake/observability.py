"""Observability — correlation IDs + Prometheus metrics.

Mirrors src/observability.ts in the parent repo. Uses `prometheus_client`
when available (official library); falls back to a no-op registry when
it isn't installed, so the package works without optional deps.
"""

from __future__ import annotations

import contextvars
import re
import uuid

# ── Correlation IDs ──────────────────────────────────────────

_correlation_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "turtlelake_correlation_id", default=None
)

_ALLOWED = re.compile(r"[^A-Za-z0-9_.\-]")


def generate_correlation_id() -> str:
    return str(uuid.uuid4())


def extract_correlation_id(headers: dict) -> str:
    """Mirrors the TS logic: prefer X-Request-Id / X-Correlation-Id, then
    W3C traceparent, else generate a new UUID."""
    for key in ("x-request-id", "X-Request-Id", "x-correlation-id", "X-Correlation-Id"):
        val = headers.get(key)
        if isinstance(val, str) and val:
            # Whitelist charset + cap length (defends against log injection).
            cleaned = _ALLOWED.sub("", val)[:128]
            return cleaned or generate_correlation_id()
    traceparent = headers.get("traceparent") or headers.get("Traceparent")
    if isinstance(traceparent, str):
        match = re.match(r"^00-([a-f0-9]{32})-", traceparent)
        if match:
            return match.group(1)
    return generate_correlation_id()


def set_correlation_id(corr_id: str | None) -> contextvars.Token:
    return _correlation_id.set(corr_id)


def get_correlation_id() -> str | None:
    return _correlation_id.get()


# ── Metrics ──────────────────────────────────────────────────

try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY  # type: ignore
    _PROM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PROM_AVAILABLE = False


class _NoopMetric:
    """No-op stand-in when prometheus_client isn't installed. Keeps call
    sites identical so we never need branches around metric calls."""

    def labels(self, **_kwargs):
        return self

    def inc(self, _value: float = 1.0) -> None:
        return None

    def observe(self, _value: float) -> None:
        return None

    def set(self, _value: float) -> None:
        return None


def _counter(name: str, help_: str, labels: tuple[str, ...]):
    if not _PROM_AVAILABLE:
        return _NoopMetric()
    try:
        return Counter(name, help_, labelnames=list(labels))
    except ValueError:
        # Metric already registered — happens under pytest-xdist workers
        # or when the module is re-imported. Fetch the existing one.
        return REGISTRY._names_to_collectors.get(name, _NoopMetric())


def _histogram(name: str, help_: str, labels: tuple[str, ...]):
    if not _PROM_AVAILABLE:
        return _NoopMetric()
    try:
        return Histogram(
            name, help_, labelnames=list(labels),
            buckets=(0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
        )
    except ValueError:
        return REGISTRY._names_to_collectors.get(name, _NoopMetric())


def _gauge(name: str, help_: str, labels: tuple[str, ...]):
    if not _PROM_AVAILABLE:
        return _NoopMetric()
    try:
        return Gauge(name, help_, labelnames=list(labels))
    except ValueError:
        return REGISTRY._names_to_collectors.get(name, _NoopMetric())


# The metric set intentionally parallels the TS server:
# tool call counts, latency, security events, rate-limit rejections.
class _Metrics:
    tool_calls = _counter(
        "turtlelake_tool_calls_total",
        "Total MCP tool invocations",
        ("tool", "status"),
    )
    tool_duration = _histogram(
        "turtlelake_tool_duration_seconds",
        "Tool execution duration in seconds",
        ("tool",),
    )
    security_events = _counter(
        "turtlelake_security_events_total",
        "Security events by type",
        ("event",),
    )
    rate_limit_rejections = _counter(
        "turtlelake_rate_limit_rejections_total",
        "Rate limit rejections by tool",
        ("tool",),
    )
    ingest_rows = _counter(
        "turtlelake_ingest_rows_total",
        "Total rows written by ingest_ttl / insert_turtle",
        ("kind",),
    )
    dataset_version = _gauge(
        "turtlelake_dataset_version",
        "Current Lance version of the in-process dataset",
        ("path",),
    )


metrics = _Metrics()


def serialize_metrics() -> bytes:
    """Return the current metric snapshot in Prometheus text format.
    Empty bytes if prometheus_client isn't installed."""
    if not _PROM_AVAILABLE:
        return b""
    return generate_latest(REGISTRY)
