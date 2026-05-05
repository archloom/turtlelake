"""UC-8 + UC-15: turtlelake's own source code contains no network clients.

We deliberately don't monkey-patch socket: Lance and pyoxigraph may open
local sockets for legitimate reasons (e.g. IPC). What UC-15 actually
requires is that *our code* never reaches out. A static import scan is
the right fidelity.

Scenarios implemented: 8.1 (static import scan), 15.1 (same coverage).
"""

from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "turtlelake"

NETWORK_LIBS = {
    "requests",
    "urllib",
    "urllib2",
    "urllib3",
    "httpx",
    "aiohttp",
    "websockets",
}


def test_15_1_no_network_client_libraries_imported_by_turtlelake():
    offenders: list[tuple[str, str]] = []
    for py in SRC.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        for lib in NETWORK_LIBS:
            # naive but effective: exact `import lib` or `from lib` at line start
            for prefix in (f"import {lib}", f"from {lib}"):
                if any(line.startswith(prefix) for line in text.splitlines()):
                    offenders.append((str(py.relative_to(SRC.parent)), lib))
    assert not offenders, f"network client imports found: {offenders}"
