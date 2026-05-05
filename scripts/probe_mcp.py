"""Minimal probe: send initialize + tools/list and dump everything."""

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


def reader(stream, label, sink):
    while True:
        line = stream.readline()
        if not line:
            return
        sink.append((label, line.rstrip("\n")))
        print(f"[{label}] {line.rstrip()}", flush=True)


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        env = {
            **os.environ,
            "TURTLELAKE_PATH": str(Path(td) / "kg"),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
            "FASTMCP_SHOW_SERVER_BANNER": "0",
        }
        # Run the module directly so we don't depend on the installed shim.
        py = REPO / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        proc = subprocess.Popen(
            [str(py), "-m", "turtlelake.mcp_server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        sink: list[tuple[str, str]] = []
        threading.Thread(target=reader, args=(proc.stdout, "out", sink), daemon=True).start()
        threading.Thread(target=reader, args=(proc.stderr, "err", sink), daemon=True).start()
        time.sleep(2.0)
        if proc.poll() is not None:
            print(f"server died: rc={proc.returncode}")
            return 2

        def send(method: str, rid: int | None = None, params: dict | None = None):
            msg: dict = {"jsonrpc": "2.0", "method": method}
            if rid is not None:
                msg["id"] = rid
            if params is not None:
                msg["params"] = params
            wire = json.dumps(msg) + "\n"
            print(f"[in ] {wire.rstrip()}", flush=True)
            proc.stdin.write(wire)
            proc.stdin.flush()

        send("initialize", 1, {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "probe", "version": "0.0.1"},
        })
        time.sleep(2.0)
        send("notifications/initialized")
        time.sleep(0.5)
        send("tools/list", 2)
        time.sleep(3.0)

        try:
            proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

        print(f"\n--- captured {len(sink)} lines ---")
        return 0


if __name__ == "__main__":
    sys.exit(main())
