"""End-to-end MCP stdio integration test.

Unlike `test_mcp_tools.py` (which introspects a FastMCP object in-process),
this test spawns `turtlelake-mcp` as a subprocess, speaks JSON-RPC over
stdio, and exercises the agent workflow the way Claude Code would.

This is the closest thing to "real agent testing" without firing up an
actual LLM.

Proves:
  1. The server boots and responds to `initialize`.
  2. `tools/list` returns all 12 declared tools with JSON schemas.
  3. `tools/call` round-trips: ingest → sparql → entity → checkpoint →
     insert (bad) → rollback → query (bad-gone) → diff → provenance.
  4. Error paths return structured errors, not crashes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest


class MCPClient:
    """Minimal MCP JSON-RPC-over-stdio client. Not for production -- just
    enough to exercise `turtlelake-mcp` end-to-end."""

    def __init__(self, env: dict | None = None):
        self.proc: subprocess.Popen | None = None
        self.env = env or {}
        self._req_id = 0

    def start(self) -> None:
        # Run via `python -m turtlelake.mcp_server` so the test works on
        # any platform regardless of how the console-script shim is wired
        # by setuptools/hatchling (`.venv/bin/turtlelake-mcp` on POSIX,
        # `.venv\Scripts\turtlelake-mcp.exe` on Windows). The Windows
        # .exe shim has historically had stdout-buffering quirks that
        # broke JSON-RPC framing; the `-m` invocation sidesteps them.
        full_env = {
            **os.environ,
            **self.env,
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        }
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "turtlelake.mcp_server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=full_env,
            text=True,
            encoding="utf-8",
            bufsize=0,  # unbuffered -- the server flushes each JSON line
        )
        # Drain stderr in a background thread so the server's audit/INFO
        # logs don't fill the pipe buffer and deadlock the JSON-RPC loop.
        self._stderr_lines: list[str] = []
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        # Give the server a moment to boot (audit-log "boot" event).
        time.sleep(1.0)
        if self.proc.poll() is not None:
            raise RuntimeError(
                f"server died at startup. stderr:\n{''.join(self._stderr_lines)}"
            )

    def _drain_stderr(self) -> None:
        if self.proc is None or self.proc.stderr is None:
            return
        for line in iter(self.proc.stderr.readline, ""):
            self._stderr_lines.append(line)

    def stop(self) -> None:
        if self.proc is None:
            return
        self.proc.stdin.close() if self.proc.stdin else None
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        self.proc = None

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    def _send(self, method: str, params: dict | None = None, *, timeout: float = 10.0) -> dict:
        assert self.proc is not None and self.proc.stdin and self.proc.stdout
        rid = self._next_id()
        msg = {"jsonrpc": "2.0", "id": rid, "method": method}
        if params is not None:
            msg["params"] = params
        line = json.dumps(msg) + "\n"
        self.proc.stdin.write(line)
        self.proc.stdin.flush()
        # Read responses until we see the matching id.
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            raw = self.proc.stdout.readline()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                # Server shouldn't emit non-JSON on stdout, but be lenient.
                continue
            if payload.get("id") == rid:
                return payload
        raise TimeoutError(f"no response to {method!r} (id={rid})")

    def _notify(self, method: str, params: dict | None = None) -> None:
        assert self.proc is not None and self.proc.stdin
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()

    # --- high-level helpers ------------------------------------

    def initialize(self) -> dict:
        resp = self._send(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "turtlelake-e2e", "version": "0.0.1"},
            },
        )
        self._notify("notifications/initialized")
        return resp

    def list_tools(self) -> list[dict]:
        resp = self._send("tools/list")
        return resp["result"]["tools"]

    def call_tool(
        self, name: str, arguments: dict | None = None, *, timeout: float = 10.0
    ) -> dict:
        resp = self._send(
            "tools/call",
            {"name": name, "arguments": arguments or {}},
            timeout=timeout,
        )
        if "error" in resp:
            return {"_error": resp["error"]}
        # FastMCP's tools/call response wraps text content:
        # {"content":[{"type":"text","text":"..."}], "isError": bool}
        return resp["result"]


def _text_from(result: dict) -> str:
    """Extract the first text-content block from a tools/call result."""
    if "_error" in result:
        return f"[protocol error] {result['_error']}"
    content = result.get("content", [])
    for block in content:
        if block.get("type") == "text":
            return block.get("text", "")
    return ""


@pytest.fixture
def mcp(tmp_path):
    """Spawn the server pointed at a fresh store; shut down after test."""
    client = MCPClient(env={"TURTLELAKE_PATH": str(tmp_path / "kg")})
    client.start()
    try:
        client.initialize()
        yield client
    finally:
        client.stop()


# ── Real agent workflow tests ───────────────────────────────


def test_01_tools_list_returns_full_surface(mcp, tmp_path):
    tools = mcp.list_tools()
    names = sorted(t["name"] for t in tools)
    expected = sorted([
        "guide", "schema", "sources",
        "sparql", "entity", "scan", "explain",
        "ingest", "insert",
        "checkpoint", "rollback", "versions", "refresh",
        "diff", "provenance", "validate", "dump",
        "save_query", "run_saved",
        "embed", "vector_search", "graph_rag", "build_vector_index",
        "compact", "prune_versions",
    ])
    assert names == expected
    # Each tool has a non-empty description + an inputSchema with type:object
    for t in tools:
        assert t.get("description"), f"{t['name']} missing description"
        schema = t.get("inputSchema", {})
        assert schema.get("type") == "object", f"{t['name']} bad schema: {schema}"


def test_01b_guide_and_schema_help_agents_self_onboard(mcp, tmp_path):
    """First-contact experience: agent calls `guide` to learn the
    workflow and `schema` to see what's in the graph."""
    ttl = tmp_path / "seed.ttl"
    ttl.write_text(
        '@prefix ex: <https://ex.org/> .\n'
        'ex:a a ex:Device ; ex:label "A" .\n'
        'ex:b a ex:Device ; ex:label "B" .\n',
        encoding="utf-8",
    )
    mcp.call_tool("ingest", {"path": str(ttl)})

    # guide → non-trivial plain text that mentions every canonical tool
    g = _text_from(mcp.call_tool("guide"))
    assert len(g) > 200
    for kw in ("schema", "checkpoint", "rollback", "validate", "provenance"):
        assert kw in g

    # schema → JSON with the classes/predicates the agent just loaded
    s = json.loads(_text_from(mcp.call_tool("schema")))
    class_iris = {c["iri"] for c in s["classes"]}
    assert "https://ex.org/Device" in class_iris
    assert s["triples"] >= 4
    assert s["versions"] >= 1


def test_02_agent_can_ingest_and_query(mcp, tmp_path):
    ttl = tmp_path / "seed.ttl"
    ttl.write_text(
        '@prefix ex: <https://ex.org/> .\n'
        'ex:a ex:p "hello" .\n'
        'ex:a ex:q ex:b .\n',
        encoding="utf-8",
    )
    ingested = _text_from(mcp.call_tool("ingest", {"path": str(ttl)}))
    assert "ingested 2 quads" in ingested

    sparql_out = _text_from(mcp.call_tool(
        "sparql", {"query": "SELECT ?o WHERE { ?s ?p ?o } ORDER BY ?o"}
    ))
    rows = json.loads(sparql_out)
    assert len(rows) == 2
    # One literal, one IRI.
    kinds = sorted(r["o"]["type"] for r in rows)
    assert kinds == ["iri", "literal"]


def test_03_agent_entity_browsing(mcp, tmp_path):
    ttl = tmp_path / "seed.ttl"
    ttl.write_text(
        '<https://ex.org/a> <https://ex.org/p> "v" .\n'
        '<https://ex.org/a> <https://ex.org/r> <https://ex.org/b> .\n',
        encoding="utf-8",
    )
    mcp.call_tool("ingest", {"path": str(ttl)})
    entity_out = _text_from(mcp.call_tool(
        "entity", {"iri": "https://ex.org/a", "hops": 2}
    ))
    got = json.loads(entity_out)
    assert got["iri"] == "https://ex.org/a"
    assert len(got["outgoing"]) == 2
    # 2-hop reached the IRI-valued object
    assert "https://ex.org/b" in got.get("neighbors", {})


def test_04_full_checkpoint_rollback_loop(mcp, tmp_path):
    """The flagship workflow -- agent makes a risky write and rolls back.
    Historically-broken (H-5 in adversarial review); locked here."""
    ttl = tmp_path / "seed.ttl"
    ttl.write_text(
        '<https://ex.org/a> <https://ex.org/p> "baseline" .\n',
        encoding="utf-8",
    )
    mcp.call_tool("ingest", {"path": str(ttl)})
    mcp.call_tool("checkpoint", {"name": "pre"})

    # Agent writes a "bad" inference.
    bad_out = _text_from(mcp.call_tool(
        "insert", {
            "turtle": '<https://ex.org/a> <https://ex.org/p> "hallucinated" .',
            "source": "agent-inference",
            "author": "test-bot",
        }
    ))
    assert "inserted 1" in bad_out

    # Rollback.
    rb = _text_from(mcp.call_tool("rollback", {"name": "pre"}))
    assert "rolled back" in rb

    # The hallucinated quad MUST NOT be visible.
    sparql_out = _text_from(mcp.call_tool(
        "sparql", {"query": "SELECT ?o WHERE { ?s ?p ?o }"}
    ))
    rows = json.loads(sparql_out)
    values = {r["o"]["value"] for r in rows}
    assert "hallucinated" not in values
    assert "baseline" in values


def test_05_diff_and_provenance_round_trip(mcp, tmp_path):
    ttl = tmp_path / "seed.ttl"
    ttl.write_text(
        '<https://ex.org/a> <https://ex.org/p> "first" .\n',
        encoding="utf-8",
    )
    mcp.call_tool("ingest", {"path": str(ttl), "source": "seed", "author": "t"})
    mcp.call_tool("insert", {
        "turtle": '<https://ex.org/a> <https://ex.org/p> "second" .',
        "source": "agent", "author": "t",
    })
    # Diff v1 -> v2 should contain exactly the new quad.
    diff_out = _text_from(mcp.call_tool(
        "diff", {"from_version": 1, "to_version": 2}
    ))
    diff = json.loads(diff_out)
    assert len(diff["added"]) == 1
    assert diff["added"][0]["object"] == "second"
    # Provenance shows both writes.
    prov_out = _text_from(mcp.call_tool("provenance"))
    prov = json.loads(prov_out)
    assert len(prov) == 2
    assert prov[0]["source"] == "seed"
    assert prov[1]["source"] == "agent"


# ── Negative paths -- structured errors, no crashes ─────────


def test_06_unknown_entity_returns_empty_not_error(mcp, tmp_path):
    # No ingest at all -- entity on anything returns empty, not 500.
    ttl = tmp_path / "one.ttl"
    ttl.write_text('<https://ex.org/x> <https://ex.org/p> "v" .\n', encoding="utf-8")
    mcp.call_tool("ingest", {"path": str(ttl)})
    out = _text_from(mcp.call_tool(
        "entity", {"iri": "https://ex.org/does-not-exist"}
    ))
    got = json.loads(out)
    assert got["outgoing"] == []
    assert got["incoming"] == []


def test_07_rollback_to_missing_tag_returns_error_json(mcp, tmp_path):
    ttl = tmp_path / "x.ttl"
    ttl.write_text('<https://ex.org/a> <https://ex.org/p> "v" .\n', encoding="utf-8")
    mcp.call_tool("ingest", {"path": str(ttl)})
    out = _text_from(mcp.call_tool("rollback", {"name": "never-tagged"}))
    # Either the secure decorator returned {"error": "..."} text, or the
    # server signaled isError. Both are acceptable -- what matters is that
    # the server didn't crash and the message mentions the missing tag.
    assert "never-tagged" in out or "error" in out.lower()


def test_08_malformed_sparql_returns_redacted_error(mcp, tmp_path):
    ttl = tmp_path / "x.ttl"
    ttl.write_text('<https://ex.org/a> <https://ex.org/p> "v" .\n', encoding="utf-8")
    mcp.call_tool("ingest", {"path": str(ttl)})
    out = _text_from(mcp.call_tool(
        "sparql", {"query": "SELECT ??? malformed {{"}
    ))
    # Secure decorator wraps errors as {"error": "..."}.
    assert "error" in out.lower()


def test_09_ingest_nonexistent_path_is_graceful(mcp, tmp_path):
    out = _text_from(mcp.call_tool(
        "ingest", {"path": str(tmp_path / "does-not-exist.ttl")}
    ))
    assert "does not exist" in out.lower() or "error" in out.lower()


def test_10_blocked_sparql_update_never_reaches_engine(mcp, tmp_path):
    ttl = tmp_path / "x.ttl"
    ttl.write_text('<https://ex.org/a> <https://ex.org/p> "v" .\n', encoding="utf-8")
    mcp.call_tool("ingest", {"path": str(ttl)})
    # DROP GRAPH is a destructive update -- the security scanner must block.
    out = _text_from(mcp.call_tool(
        "sparql", {"query": "DROP GRAPH <https://ex.org/g>"}
    ))
    assert "blocked" in out.lower() or "sparql_update_mask" in out.lower()
    # Sanity: benign data still queryable after the block.
    sparql_out = _text_from(mcp.call_tool(
        "sparql", {"query": "SELECT ?s WHERE { ?s ?p ?o } LIMIT 1"}
    ))
    assert json.loads(sparql_out)


def test_11_shacl_validate_flags_a_real_violation(mcp, tmp_path):
    """The 'agent writes are checked before they're trusted' pitch hinges
    on `validate` -- if SHACL silently passes a bad insert, the safety
    story is broken. Ingest a violator, run the shape check over MCP
    wire, assert non-conformance with the offending IRI in the report."""
    pytest.importorskip("pyshacl")

    shapes = tmp_path / "shapes.ttl"
    shapes.write_text(
        "@prefix sh: <http://www.w3.org/ns/shacl#> .\n"
        "@prefix ex: <https://ex.org/> .\n"
        "ex:DeviceShape a sh:NodeShape ;\n"
        "    sh:targetClass ex:Device ;\n"
        "    sh:property [ sh:path ex:label ; sh:minCount 1 ;\n"
        "                  sh:datatype <http://www.w3.org/2001/XMLSchema#string> ] .\n",
        encoding="utf-8",
    )

    # ex:c is a Device with no ex:label -- must violate.
    mcp.call_tool("insert", {
        "turtle": "@prefix ex: <https://ex.org/> . ex:c a ex:Device .",
    })
    # pyshacl's first-call init is heavy on Windows (rdflib OWL imports).
    # Allow 60s for the very first SHACL call; later calls in the same
    # process are fast.
    out = _text_from(mcp.call_tool(
        "validate", {"shapes_path": str(shapes)}, timeout=90.0
    ))
    report = json.loads(out)
    assert report["conforms"] is False, (
        f"shapes require ex:label on ex:Device; ex:c has none -- "
        f"validate should report non-conformance: {report}"
    )
    assert "https://ex.org/c" in report["report_text"], (
        f"validation report should name the violating IRI: {report['report_text']}"
    )
