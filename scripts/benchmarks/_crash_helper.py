"""Subprocess helper for the crash-recovery operational benchmark.

Spawned by `operational.py::bench_crash_recovery`. Writes data to
the dataset, sets up the WAL pending-checkpoint marker, creates the
triples tag, and then deliberately exits non-zero before creating
the embeddings tag — simulating a crash mid-checkpoint.

The parent then reopens the dataset and verifies that
`_recover_pending_checkpoint` reconciles the partial state.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from pyoxigraph import Literal, NamedNode, Quad

from turtlelake import Dataset


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: _crash_helper.py <dataset_path>", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    ds = Dataset.open(path)
    label = NamedNode("http://www.w3.org/2000/01/rdf-schema#label")
    ds._append_quads(
        [Quad(NamedNode("https://ex/A"), label, Literal("A"))],
        batch_size=10,
    )
    # Embed something so there's an embeddings dataset to tag.
    ds.embed(["https://ex/A"], [[1.0, 0.0]], model_id="crash:m1")

    triples_v = ds.version
    emb_v = ds._embeddings.version

    # Manually set up a partial checkpoint: write the WAL marker,
    # create the triples tag, but skip the embeddings tag — then
    # crash. This mirrors what happens if a process is SIGKILL'd
    # between the two tag creates inside `checkpoint()`.
    manifest = ds._read_manifest()
    manifest["pending_checkpoint"] = {
        "name": "v1",
        "triples_version": triples_v,
        "embeddings_version": emb_v,
    }
    ds._write_manifest_atomic(manifest)
    ds._lance.tags.create("v1", triples_v)
    # Simulate a crash. SystemExit with non-zero return code is the
    # closest deterministic stand-in for SIGKILL we can use without
    # losing flush guarantees on the writes above.
    os._exit(137)  # noqa: SLF001 — deliberate; real SIGKILL would do the same


if __name__ == "__main__":
    raise SystemExit(main())
