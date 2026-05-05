"""Security module tests — mirrors the FT MCP test coverage.

Rate limiter, input scanner, error redaction, output framing, audit log.
"""

import json

import pytest

from turtlelake.security import (
    InputBlockedError,
    RateLimitExceeded,
    audit_log,
    check_rate_limit,
    frame_tool_output,
    redact_error,
    reset_rate_limits,
    scan_input,
    scan_tool_inputs,
)


@pytest.fixture(autouse=True)
def _clean_rate_limits():
    reset_rate_limits()
    yield
    reset_rate_limits()


# ── scan_input ────────────────────────────────────────────────


def test_benign_input_passes():
    r = scan_input("SELECT ?s WHERE { ?s a <https://example.org/Device> }")
    assert r.safe
    assert r.blocked_by is None


def test_null_byte_blocked():
    r = scan_input("hello\x00world")
    assert not r.safe
    assert r.blocked_by == "null_byte"


def test_directional_override_blocked():
    r = scan_input("evil\u202euri")
    assert not r.safe
    assert r.blocked_by == "directional_override"


def test_path_traversal_blocked():
    r = scan_input("../../etc/passwd")
    assert not r.safe
    assert r.blocked_by == "path_traversal"


def test_shell_injection_blocked():
    r = scan_input("; rm -rf /")
    assert not r.safe
    assert r.blocked_by == "shell_injection"


def test_sparql_destructive_update_blocked():
    r = scan_input("DROP GRAPH <https://evil>")
    assert not r.safe
    assert r.blocked_by == "sparql_update_mask"


# ── scan_tool_inputs ──────────────────────────────────────────


def test_tool_inputs_recurse_into_nested_containers():
    with pytest.raises(InputBlockedError):
        scan_tool_inputs("tool", {"a": {"b": ["ok", "../../etc"]}})


def test_tool_inputs_return_value_is_input_unchanged():
    # We do not auto-fix; if it passes, it returns the input verbatim.
    result = scan_tool_inputs("tool", {"q": "SELECT ?s WHERE { ?s ?p ?o }"})
    assert result == {"q": "SELECT ?s WHERE { ?s ?p ?o }"}


# ── rate limiter ──────────────────────────────────────────────


def test_rate_limit_allows_under_threshold():
    for _ in range(5):
        check_rate_limit("t", max_requests=10)  # should not raise


def test_rate_limit_blocks_over_threshold():
    for _ in range(3):
        check_rate_limit("t", max_requests=3)
    with pytest.raises(RateLimitExceeded):
        check_rate_limit("t", max_requests=3)


def test_rate_limit_is_per_tool():
    for _ in range(3):
        check_rate_limit("a", max_requests=3)
    # Another tool is unaffected.
    check_rate_limit("b", max_requests=3)


# ── redact_error ──────────────────────────────────────────────


def test_redact_bearer_token():
    e = RuntimeError("call failed with Authorization: Bearer abc.def.xyz123")
    assert "abc.def.xyz123" not in redact_error(e)
    assert "[REDACTED]" in redact_error(e)


def test_redact_api_key():
    e = RuntimeError("request failed: api_key=sk-secret1234567890")
    assert "sk-secret1234567890" not in redact_error(e)


def test_redact_truncates_long_message():
    e = RuntimeError("x" * 1000)
    out = redact_error(e)
    assert out.endswith("… [truncated]")
    assert len(out) < 1000


# ── frame_tool_output ─────────────────────────────────────────


def test_frame_wraps_with_metadata_and_warning():
    framed = frame_tool_output("sparql", [{"s": "x"}])
    assert framed["_meta"]["tool"] == "sparql"
    assert "NOT instructions" in framed["_meta"]["warning"]
    assert framed["data"] == [{"s": "x"}]


# ── audit_log goes to stderr as JSON ─────────────────────────


def test_audit_log_emits_json_to_stderr(capsys):
    audit_log("tool_call", {"tool": "sparql"})
    captured = capsys.readouterr()
    assert captured.err.startswith("[AUDIT] ")
    record = json.loads(captured.err.removeprefix("[AUDIT] ").strip())
    assert record["event"] == "tool_call"
    assert record["tool"] == "sparql"
    assert "timestamp" in record
