"""`@secure` decorator — the Python equivalent of SecureFluidTopicsTool.

Wraps an MCP tool function with the production layer stack, in order:

  1. Rate limit
  2. Input scan (regex-based, Unicode-safe)
  3. Audit-log the call
  4. Execute the tool
  5. Frame output with data-boundary markers
  6. Update Prometheus metrics
  7. Audit-log success / redacted error

Usage:

    @mcp.tool()
    @secure("sparql")
    def sparql(query: str) -> str:
        ...

FastMCP's `@mcp.tool()` must be *outside* `@secure` so the schema it
derives reflects the original signature.
"""

from __future__ import annotations

import functools
import json
import time
from typing import Any, Callable

from turtlelake.observability import metrics
from turtlelake.security import (
    InputBlockedError,
    RateLimitExceeded,
    audit_log,
    check_rate_limit,
    frame_tool_output,
    redact_error,
    scan_tool_inputs,
)


def secure(tool_name: str, *, frame: bool = False) -> Callable:
    """Wrap a FastMCP tool function with the turtlelake security stack.

    `frame=True` causes the return value to be JSON-wrapped with a data
    boundary preamble; off by default because most turtlelake tools already
    return structured JSON, and double-wrapping is ugly for agents.
    """

    def decorator(fn: Callable[..., str]) -> Callable[..., str]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> str:
            start = time.perf_counter()

            # 1. Rate limit (per-tool, honors TURTLELAKE_RATE_LIMIT_<TOOL>)
            try:
                check_rate_limit(tool_name)  # per-tool limit resolved internally
            except RateLimitExceeded as e:
                audit_log("rate_limit", {"tool": tool_name})
                metrics.rate_limit_rejections.labels(tool=tool_name).inc()
                metrics.security_events.labels(event="rate_limit").inc()
                metrics.tool_calls.labels(tool=tool_name, status="rate_limited").inc()
                return json.dumps({"error": str(e)})

            # 2. Input scan
            try:
                scan_tool_inputs(tool_name, {"args": list(args), "kwargs": kwargs})
            except InputBlockedError as e:
                metrics.security_events.labels(event="input_blocked").inc()
                metrics.tool_calls.labels(tool=tool_name, status="blocked").inc()
                return json.dumps({"error": redact_error(e)})

            # 3. Call log (redacted; we don't dump arbitrary inputs)
            audit_log(
                "tool_call",
                {
                    "tool": tool_name,
                    "params": _redact_args(kwargs) if kwargs else _redact_args(
                        dict(enumerate(args))
                    ),
                },
            )

            # 4. Execute
            try:
                result = fn(*args, **kwargs)
            except Exception as e:  # noqa: BLE001 — we redact and rethrow as a safe string
                duration = time.perf_counter() - start
                metrics.tool_calls.labels(tool=tool_name, status="error").inc()
                metrics.tool_duration.labels(tool=tool_name).observe(duration)
                redacted = redact_error(e)
                audit_log(
                    "tool_error",
                    {"tool": tool_name, "error": redacted, "duration_ms": round(duration * 1000)},
                )
                return json.dumps({"error": redacted})

            # 5. Optional output framing
            duration = time.perf_counter() - start
            if frame:
                try:
                    parsed = json.loads(result) if isinstance(result, str) else result
                except (TypeError, ValueError):
                    parsed = result
                result = json.dumps(frame_tool_output(tool_name, parsed))

            # 6/7. Metrics + success log
            metrics.tool_calls.labels(tool=tool_name, status="success").inc()
            metrics.tool_duration.labels(tool=tool_name).observe(duration)
            audit_log(
                "tool_success",
                {"tool": tool_name, "duration_ms": round(duration * 1000)},
            )
            return result

        return wrapper

    return decorator


def _redact_args(params: dict) -> dict:
    """Truncate long string values before they end up in an audit line."""
    out: dict = {}
    for k, v in params.items():
        if isinstance(v, str) and len(v) > 200:
            out[k] = v[:50] + "...[truncated]"
        else:
            out[k] = v
    return out
