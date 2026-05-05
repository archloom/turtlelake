"""Shared helpers for the domain demos in `examples/demo_*.py`.

Two responsibilities:

1. **Cached HTTPS download** of public ontology files. We hash the
   URL → file path under `~/.cache/turtlelake-demos/`, so re-running
   a demo doesn't re-download a 35 MB ontology each time.

2. **A consistent "naive vs grounded" printer** so every demo
   demonstrates the same value prop (the ungrounded LLM hallucinates
   X; the grounded agent produces a verifiable answer using
   turtlelake's primitives).

Each demo imports from this module to avoid duplicating the
download / formatting glue.
"""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from pathlib import Path

DEMO_CACHE = Path(
    os.environ.get(
        "TURTLELAKE_DEMO_CACHE", str(Path.home() / ".cache/turtlelake-demos")
    )
)


def cache_dir() -> Path:
    DEMO_CACHE.mkdir(parents=True, exist_ok=True)
    return DEMO_CACHE


def download(url: str, *, suffix: str | None = None) -> Path:
    """Fetch `url` into the demo cache. Idempotent. Returns a path
    suitable to pass straight to `Dataset.ingest_ttl(...)` or to a
    pronto/rdflib loader, depending on the file format."""
    name = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    ext = suffix or Path(url).suffix or ".bin"
    target = cache_dir() / f"{name}{ext}"
    if target.exists() and target.stat().st_size > 0:
        return target
    print(f"  downloading {url}", file=sys.stderr)
    req = urllib.request.Request(
        url, headers={"User-Agent": "turtlelake-demo/0.0.1"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp, open(target, "wb") as fh:
        while chunk := resp.read(64 * 1024):
            fh.write(chunk)
    return target


# ── presentation helpers ───────────────────────────────────────


def banner(title: str) -> None:
    line = "=" * max(len(title) + 2, 60)
    print(f"\n{line}\n {title}\n{line}\n")


def section(title: str) -> None:
    print(f"\n-- {title} --")


def naive(text: str) -> None:
    """What an ungrounded LLM might say. Always print this first so
    the contrast with the grounded answer is obvious."""
    print(f"  [naive]    ungrounded LLM: {text}")


def grounded(text: str) -> None:
    """The verifiable answer the agent produces using turtlelake's
    primitives."""
    print(f"  [grounded] {text}")


def shows(name: str, *, text: str) -> None:
    """One-liner reminder of which turtlelake primitive lit up here."""
    print(f"     -> uses {name}: {text}")
