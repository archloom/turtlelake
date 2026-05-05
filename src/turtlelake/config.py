"""Runtime configuration — env vars validated once at boot.

Mirrors src/config.ts in the parent repo: fail fast with actionable
messages if required env is missing or malformed. This is the *runtime*
config for the MCP server; application-level index rules live in
`indexing.ttl` per the LanceDB-philosophy decision.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimeConfig:
    store_path: Path
    rate_limit_max_per_minute: int
    max_ingest_bytes: int
    audit_to_stderr: bool
    ingest_root: Path | None

    @classmethod
    def from_env(cls) -> RuntimeConfig:
        store = os.environ.get("TURTLELAKE_PATH", "./.turtlelake")
        rate = _int_env("TURTLELAKE_RATE_LIMIT", default=30, minimum=1, maximum=10_000)
        # Default 256 MiB — enough for most ontologies; explicit cap prevents
        # a runaway TTL from blowing through RAM via pyoxigraph's parser.
        max_bytes = _int_env(
            "TURTLELAKE_MAX_INGEST_BYTES",
            default=256 * 1024 * 1024,
            minimum=1024,
            maximum=100 * 1024 * 1024 * 1024,
        )
        audit = os.environ.get("TURTLELAKE_AUDIT", "1").lower() not in ("0", "false", "no")
        # Optional allowlist root for `ingest(path)`. When set, MCP
        # `ingest` calls must target a path under this directory —
        # prevents an agent from reading `/etc/shadow` via a prompt
        # injection. Unset (default) = no restriction; appropriate for
        # local dev, but shared deployments should set this.
        root_env = os.environ.get("TURTLELAKE_INGEST_ROOT")
        ingest_root = Path(root_env).resolve() if root_env else None
        return cls(
            store_path=Path(store),
            rate_limit_max_per_minute=rate,
            max_ingest_bytes=max_bytes,
            audit_to_stderr=audit,
            ingest_root=ingest_root,
        )


def _int_env(name: str, *, default: int, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as e:
        raise RuntimeError(
            f"{name} is not a valid integer: {raw!r}."
        ) from e
    if value < minimum or value > maximum:
        raise RuntimeError(
            f"{name}={value} out of range [{minimum}, {maximum}]."
        )
    return value
