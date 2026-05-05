"""Probe whether `Dataset.validate(shapes_path)` works in-process.

If this hangs we have a SHACL bug; if it returns we know the bug is
in the MCP wrapper, not the underlying validator.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from turtlelake import Dataset


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        ds = Dataset.open(Path(td) / "kg")
        ds.insert_turtle(
            "@prefix ex: <https://ex.org/> . ex:c a ex:Device ."
        )
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
        print("calling ds.validate ...", flush=True)
        report = ds.validate(shapes)
        print(json.dumps(report, default=str, indent=2)[:1200])
    return 0


if __name__ == "__main__":
    sys.exit(main())
