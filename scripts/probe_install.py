"""Verify the README quickstart works against a fresh install.

Imports the package, runs the exact ingest -> SPARQL -> tag flow shown
in the README, and prints PASS / FAIL per step.
"""
from __future__ import annotations

import pathlib
import sys
import tempfile

import turtlelake
from turtlelake import Dataset


def main() -> int:
    print(f"PASS  import turtlelake (version {turtlelake.__version__ if hasattr(turtlelake, '__version__') else 'unknown'})")

    with tempfile.TemporaryDirectory() as td:
        kg = Dataset.open(pathlib.Path(td) / "verify.turtlelake")
        kg.insert_turtle(
            "@prefix ex: <https://ex.org/> .\n"
            'ex:foundation a ex:Book ; ex:title "Foundation" .\n'
        )
        assert kg.count() >= 2, kg.count()
        print(f"PASS  insert_turtle (count={kg.count()}, version={kg.version})")

        rows = kg.query(
            "SELECT ?title WHERE { ?b <https://ex.org/title> ?title }"
        )
        assert rows and rows[0]["title"]["value"] == "Foundation", rows
        print(f"PASS  sparql ({len(rows)} row, title={rows[0]['title']['value']!r})")

        kg.tag("baseline")
        assert "baseline" in kg.tags()
        print(f"PASS  tag (tags={kg.tags()})")

    # Confirm the MCP server can boot in-process (don't speak stdio here;
    # the e2e test already covers that path).
    from turtlelake.mcp_server import build_server, tool_names

    mcp = build_server()
    assert len(tool_names()) == 25
    print(f"PASS  mcp_server.build_server (25 tools declared)")

    # Confirm SHACL extra reachable.
    try:
        import pyshacl  # noqa
        print(f"PASS  pyshacl import (shacl extra installed)")
    except ImportError:
        print("FAIL  pyshacl not importable; the [shacl] extra is broken")
        return 1

    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
