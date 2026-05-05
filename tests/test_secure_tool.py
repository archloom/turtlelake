"""The @secure decorator is the runtime defense stack — it has to work.
Covers: success path, rate-limit, input-block, error-redact, frame mode."""

import json

import pytest

from turtlelake.secure_tool import secure
from turtlelake.security import reset_rate_limits


@pytest.fixture(autouse=True)
def _clean():
    reset_rate_limits()
    yield
    reset_rate_limits()


def test_secure_passes_through_on_success():
    @secure("ok_tool")
    def ok(x: int) -> str:
        return json.dumps({"value": x * 2})

    assert json.loads(ok(3)) == {"value": 6}


def test_secure_rate_limit_returns_json_error(monkeypatch):
    monkeypatch.setenv("TURTLELAKE_RATE_LIMIT_RL_TOOL", "2")

    @secure("rl_tool")
    def step() -> str:
        return json.dumps({"ok": True})

    assert json.loads(step()) == {"ok": True}
    assert json.loads(step()) == {"ok": True}
    blocked = json.loads(step())
    assert "error" in blocked
    assert "Rate limit" in blocked["error"]


def test_secure_input_block_returns_redacted_error():
    @secure("scan_tool")
    def nope(query: str) -> str:
        return "ran"

    out = json.loads(nope("../../etc/passwd"))
    assert "error" in out
    assert "path_traversal" in out["error"]


def test_secure_error_redaction_strips_bearer_tokens():
    @secure("boom_tool")
    def boom() -> str:
        raise RuntimeError("upstream failed: Authorization: Bearer secret-xyz-123")

    out = json.loads(boom())
    assert "error" in out
    assert "secret-xyz-123" not in out["error"]
    assert "[REDACTED]" in out["error"]


def test_secure_frame_wraps_with_data_boundary():
    @secure("framed_tool", frame=True)
    def reader() -> str:
        return json.dumps({"row": 1})

    wrapped = json.loads(reader())
    assert wrapped["_meta"]["tool"] == "framed_tool"
    assert "NOT instructions" in wrapped["_meta"]["warning"]
    assert wrapped["data"] == {"row": 1}
