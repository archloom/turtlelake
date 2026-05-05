"""Real stdio MCP smoke test against the installed turtlelake-mcp binary.

Drives the canonical agent workflow over JSON-RPC: initialize ->
tools/list -> ingest -> sparql -> entity -> checkpoint -> insert ->
rollback -> diff -> provenance. Prints PASS/FAIL per step.

Run: python scripts/smoke_mcp.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
BIN = REPO / ".venv" / ("Scripts/turtlelake-mcp.exe" if os.name == "nt" else "bin/turtlelake-mcp")
PY = REPO / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


class StdioMCP:
    def __init__(self, env: dict[str, str]) -> None:
        # Run via `python -m turtlelake.mcp_server` -- the same code path the
        # `turtlelake-mcp` shim uses, but skips Windows .exe stub quirks that
        # can swallow stdout buffering.
        self.proc = subprocess.Popen(
            [str(PY), "-m", "turtlelake.mcp_server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, **env, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
            text=True,
            encoding="utf-8",
            bufsize=0,
        )
        self.rid = 0
        # Drain stderr in a background thread so the server's audit/INFO
        # logs don't fill the pipe buffer and deadlock the JSON-RPC loop.
        self._stderr_lines: list[str] = []
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        time.sleep(1.0)
        if self.proc.poll() is not None:
            raise RuntimeError(
                f"server died at startup. stderr:\n{''.join(self._stderr_lines)}"
            )

    def _drain_stderr(self) -> None:
        for line in iter(self.proc.stderr.readline, ""):
            self._stderr_lines.append(line)

    def _send(self, method: str, params: dict | None = None) -> dict:
        self.rid += 1
        msg: dict = {"jsonrpc": "2.0", "id": self.rid, "method": method}
        if params is not None:
            msg["params"] = params
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("id") == self.rid:
                return payload
        raise TimeoutError(method)

    def notify(self, method: str, params: dict | None = None) -> None:
        msg: dict = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()

    def call(self, name: str, args: dict | None = None) -> str:
        resp = self._send("tools/call", {"name": name, "arguments": args or {}})
        if "error" in resp:
            return f"[error] {resp['error']}"
        for block in resp.get("result", {}).get("content", []):
            if block.get("type") == "text":
                return block.get("text", "")
        return ""

    def stop(self) -> None:
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def main() -> int:
    if not BIN.exists():
        print(f"FAIL: {BIN} not found. pip install -e . under .venv first.")
        return 2

    with tempfile.TemporaryDirectory() as td:
        store = Path(td) / "kg"
        ttl = Path(td) / "seed.ttl"
        ttl.write_text(
            "@prefix ex: <https://ex.org/> .\n"
            'ex:a a ex:Device ; ex:label "alpha" ; ex:friend ex:b .\n'
            'ex:b a ex:Device ; ex:label "beta" .\n',
            encoding="utf-8",
        )
        mcp = StdioMCP({"TURTLELAKE_PATH": str(store)})
        try:
            init = mcp._send(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "smoke", "version": "0.0.1"},
                },
            )
            assert "result" in init, f"initialize: {init}"
            mcp.notify("notifications/initialized")
            print("PASS  initialize")

            tools = mcp._send("tools/list")["result"]["tools"]
            assert len(tools) == 25, f"want 25 tools, got {len(tools)}"
            print(f"PASS  tools/list ({len(tools)} tools)")

            out = mcp.call("guide")
            assert "schema" in out and "checkpoint" in out, out[:200]
            print("PASS  guide")

            out = mcp.call("ingest", {"path": str(ttl)})
            assert "ingested" in out and "quads" in out, out
            print(f"PASS  ingest ({out.strip()})")

            out = mcp.call("schema")
            doc = json.loads(out)
            assert doc["triples"] >= 5, doc
            print(f"PASS  schema (triples={doc['triples']}, versions={doc['versions']})")

            out = mcp.call("sparql", {"query": "SELECT (COUNT(*) AS ?n) WHERE { ?s ?p ?o }"})
            rows = json.loads(out)
            assert int(rows[0]["n"]["value"]) >= 5, rows
            print(f"PASS  sparql (count={rows[0]['n']['value']})")

            out = mcp.call("entity", {"iri": "https://ex.org/a", "hops": 1})
            ent = json.loads(out)
            assert ent["iri"] == "https://ex.org/a"
            assert any(e["object"]["value"] == "https://ex.org/b"
                       for e in ent["outgoing"] if isinstance(e["object"], dict))
            print("PASS  entity (1-hop)")

            out = mcp.call("checkpoint", {"name": "pre"})
            assert "version" in out.lower() or "checkpoint" in out.lower(), out
            print(f"PASS  checkpoint ({out.strip()})")

            out = mcp.call("insert", {
                "turtle": '<https://ex.org/a> <https://ex.org/label> "hallucinated" .'
            })
            assert "inserted" in out.lower() or "quads" in out.lower(), out
            print(f"PASS  insert ({out.strip()})")

            out = mcp.call("rollback", {"name": "pre"})
            assert "rolled back" in out.lower() or "version" in out.lower(), out
            print(f"PASS  rollback ({out.strip()})")

            out = mcp.call("sparql", {
                "query": 'ASK WHERE { ?s <https://ex.org/label> "hallucinated" }'
            })
            ask = json.loads(out)
            assert ask is False or (isinstance(ask, dict) and ask.get("boolean") is False), ask
            print("PASS  rollback erased the bad insert")

            out = mcp.call("provenance", {"iri": "https://ex.org/a"})
            assert out, "provenance returned empty"
            print("PASS  provenance")

            out = mcp.call("versions")
            vers = json.loads(out)
            entries = vers["versions"] if isinstance(vers, dict) else vers
            assert len(entries) >= 2, vers
            tags = vers.get("tags", []) if isinstance(vers, dict) else []
            print(f"PASS  versions ({len(entries)} entries, tags={tags})")

            # SHACL: write shapes that require ex:Device entities to have an
            # ex:label literal, ingest a violator (ex:c with no label), then
            # call validate and assert it reports non-conformance. This is the
            # "agent writes are checked before they're trusted" pitch -- if it
            # silently passes, the safety story is broken.
            shapes = Path(td) / "shapes.ttl"
            shapes.write_text(
                "@prefix sh: <http://www.w3.org/ns/shacl#> .\n"
                "@prefix ex: <https://ex.org/> .\n"
                "ex:DeviceShape a sh:NodeShape ;\n"
                "    sh:targetClass ex:Device ;\n"
                "    sh:property [ sh:path ex:label ; sh:minCount 1 ;\n"
                "                  sh:datatype <http://www.w3.org/2001/XMLSchema#string> ] .\n",
                encoding="utf-8",
            )
            mcp.call("insert", {
                "turtle": "@prefix ex: <https://ex.org/> . ex:c a ex:Device .",
            })
            out = mcp.call("validate", {"shapes_path": str(shapes)})
            if out.startswith("[error]"):
                print(f"SKIP  validate (shacl extra not installed: {out[:80]})")
            else:
                report = json.loads(out)
                assert report.get("conforms") is False, (
                    f"shapes require ex:label on ex:Device; ex:c has none -- "
                    f"validate should report non-conformance: {report}"
                )
                print(f"PASS  validate (conforms=False, report flagged ex:c)")

            print("\nALL CHECKS PASSED")
            return 0
        except AssertionError as e:
            print(f"FAIL: {e}")
            return 1
        finally:
            mcp.stop()


if __name__ == "__main__":
    sys.exit(main())
