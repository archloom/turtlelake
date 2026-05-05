"""RuntimeConfig env-var validation — fail-fast with actionable messages."""

import pytest

from turtlelake.config import RuntimeConfig


def test_defaults_when_no_env(monkeypatch):
    for k in (
        "TURTLELAKE_PATH",
        "TURTLELAKE_RATE_LIMIT",
        "TURTLELAKE_MAX_INGEST_BYTES",
        "TURTLELAKE_AUDIT",
    ):
        monkeypatch.delenv(k, raising=False)
    cfg = RuntimeConfig.from_env()
    assert str(cfg.store_path) == ".turtlelake"
    assert cfg.rate_limit_max_per_minute == 30
    assert cfg.max_ingest_bytes == 256 * 1024 * 1024
    assert cfg.audit_to_stderr is True


def test_custom_path_and_rate(monkeypatch):
    monkeypatch.setenv("TURTLELAKE_PATH", "/tmp/kg")
    monkeypatch.setenv("TURTLELAKE_RATE_LIMIT", "60")
    cfg = RuntimeConfig.from_env()
    assert str(cfg.store_path) == "/tmp/kg"
    assert cfg.rate_limit_max_per_minute == 60


def test_invalid_integer_raises(monkeypatch):
    monkeypatch.setenv("TURTLELAKE_RATE_LIMIT", "not-a-number")
    with pytest.raises(RuntimeError, match="not a valid integer"):
        RuntimeConfig.from_env()


def test_out_of_range_raises(monkeypatch):
    monkeypatch.setenv("TURTLELAKE_RATE_LIMIT", "0")
    with pytest.raises(RuntimeError, match="out of range"):
        RuntimeConfig.from_env()


def test_audit_off_accepts_common_negations(monkeypatch):
    for value in ("0", "false", "False", "NO"):
        monkeypatch.setenv("TURTLELAKE_AUDIT", value)
        cfg = RuntimeConfig.from_env()
        assert cfg.audit_to_stderr is False
