"""Verify that the installed `turtlelake-mcp` console script (the
.exe shim on Windows, the bash launcher on POSIX) works for stdio
JSON-RPC. If this passes, scripts that point Claude Desktop at the
shim path will work without falling back to `python -m`.
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


def main() -> int:
    if not BIN.exists():
        print(f"FAIL: {BIN} not installed")
        return 2

    with tempfile.TemporaryDirectory() as td:
        env = {
            **os.environ,
            "TURTLELAKE_PATH": str(Path(td) / "kg"),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        }
        proc = subprocess.Popen(
            [str(BIN)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            encoding="utf-8",
            bufsize=0,
        )
        stderr_lines: list[str] = []
        threading.Thread(
            target=lambda: stderr_lines.extend(iter(proc.stderr.readline, "")),
            daemon=True,
        ).start()
        time.sleep(1.0)
        if proc.poll() is not None:
            print(f"FAIL: shim exited rc={proc.returncode}; stderr:\n{''.join(stderr_lines)}")
            return 1

        msg = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "shim-probe", "version": "0.0.1"},
            },
        }
        proc.stdin.write(json.dumps(msg) + "\n")
        proc.stdin.flush()

        # Wait up to 10s for a single JSON line.
        deadline = time.monotonic() + 10.0
        got = ""
        while time.monotonic() < deadline:
            line = proc.stdout.readline()
            if line.strip():
                got = line
                break

        try:
            proc.stdin.close()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

        if not got:
            print("FAIL: shim accepted JSON-RPC but produced no stdout")
            print(f"stderr was:\n{''.join(stderr_lines)}")
            return 1
        try:
            payload = json.loads(got)
        except json.JSONDecodeError as e:
            print(f"FAIL: shim emitted non-JSON on stdout: {got!r} ({e})")
            return 1

        if payload.get("id") != 1 or "result" not in payload:
            print(f"FAIL: unexpected payload: {payload}")
            return 1

        print(f"PASS: shim responded to initialize ({BIN.name})")
        return 0


if __name__ == "__main__":
    sys.exit(main())
