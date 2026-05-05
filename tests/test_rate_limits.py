"""Per-tool rate limits — must match how the agent surface is actually used.

Flat limits would rate-limit a legitimate graph-browsing agent in seconds.
These tests pin the policy: read tools get high caps, destructive tools
get low caps, env vars can override either direction.
"""

import pytest

from turtlelake.security import (
    DEFAULT_RATE_LIMITS,
    RateLimitExceeded,
    check_rate_limit,
    reset_rate_limits,
    tool_rate_limit,
)


@pytest.fixture(autouse=True)
def _clean():
    reset_rate_limits()
    yield
    reset_rate_limits()


# ── Default policy contracts ─────────────────────────────────


def test_read_tools_have_high_caps():
    """Agents browse. 240/min for entity is the lower bound for real use."""
    assert DEFAULT_RATE_LIMITS["entity"] >= 200
    assert DEFAULT_RATE_LIMITS["sparql"] >= 100
    assert DEFAULT_RATE_LIMITS["refresh"] >= 200


def test_destructive_tools_have_low_caps():
    """Rollback is destructive — strict cap is a feature, not a bug."""
    assert DEFAULT_RATE_LIMITS["rollback"] <= 15


def test_memory_heavy_reads_have_moderate_caps():
    """diff and validate materialize — don't let agents spam them."""
    assert DEFAULT_RATE_LIMITS["diff"] <= 30
    assert DEFAULT_RATE_LIMITS["validate"] <= 30


def test_every_declared_tool_has_a_policy():
    """tool_names() in mcp_server must be a subset of DEFAULT_RATE_LIMITS."""
    from turtlelake.mcp_server import tool_names
    expected = set(tool_names())
    defined = set(DEFAULT_RATE_LIMITS)
    missing = expected - defined
    assert not missing, f"tools without a rate-limit policy: {missing}"


# ── Env overrides ────────────────────────────────────────────


def test_per_tool_env_override(monkeypatch):
    monkeypatch.setenv("TURTLELAKE_RATE_LIMIT_ENTITY", "5")
    assert tool_rate_limit("entity") == 5


def test_global_env_override(monkeypatch):
    monkeypatch.delenv("TURTLELAKE_RATE_LIMIT_ENTITY", raising=False)
    monkeypatch.setenv("TURTLELAKE_RATE_LIMIT", "17")
    # Global override applies to all tools.
    assert tool_rate_limit("entity") == 17
    assert tool_rate_limit("rollback") == 17


def test_per_tool_beats_global(monkeypatch):
    monkeypatch.setenv("TURTLELAKE_RATE_LIMIT", "17")
    monkeypatch.setenv("TURTLELAKE_RATE_LIMIT_ENTITY", "99")
    assert tool_rate_limit("entity") == 99
    assert tool_rate_limit("rollback") == 17  # falls through to global


def test_zero_disables_limit(monkeypatch):
    monkeypatch.setenv("TURTLELAKE_RATE_LIMIT_ENTITY", "0")
    # check_rate_limit must not raise regardless of how many calls.
    for _ in range(10_000):
        check_rate_limit("entity")


def test_invalid_env_raises(monkeypatch):
    monkeypatch.setenv("TURTLELAKE_RATE_LIMIT_ENTITY", "not-a-number")
    with pytest.raises(RuntimeError, match="must be an integer"):
        tool_rate_limit("entity")


def test_negative_env_raises(monkeypatch):
    monkeypatch.setenv("TURTLELAKE_RATE_LIMIT_ENTITY", "-1")
    with pytest.raises(RuntimeError, match="must be >= 0"):
        tool_rate_limit("entity")


# ── Behavior: rollback strict cap is enforced ────────────────


def test_rollback_rate_limit_trips_at_cap():
    cap = DEFAULT_RATE_LIMITS["rollback"]
    for _ in range(cap):
        check_rate_limit("rollback")
    with pytest.raises(RateLimitExceeded):
        check_rate_limit("rollback")


def test_entity_rate_limit_does_not_trip_on_normal_browsing():
    """100 entity calls — a real agent session — must not trip."""
    for _ in range(100):
        check_rate_limit("entity")  # would fail at 30 with the old flat cap
