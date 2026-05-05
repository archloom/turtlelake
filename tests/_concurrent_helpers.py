"""Worker functions invoked from concurrent-write tests via
multiprocessing's `spawn` start method. Lance is not fork-safe (its
own warning), so test workers MUST be defined in an importable module
rather than as inline lambdas / closures.
"""

from __future__ import annotations


def append_triples(path: str, label: str, n: int, queue) -> None:
    """Append `n` triples in a fresh process. Reports outcome to
    `queue` as `(label, status, error_message_or_None)`."""
    try:
        from pyoxigraph import Literal, NamedNode, Quad

        from turtlelake import Dataset

        ds = Dataset.open(path)
        pred = NamedNode("http://www.w3.org/2000/01/rdf-schema#label")
        quads = [
            Quad(NamedNode(f"https://ex/{label}/{i}"), pred, Literal(f"{label}{i}"))
            for i in range(n)
        ]
        ds._append_quads(quads, batch_size=64)
        queue.put((label, "ok", None))
    except Exception as e:  # pragma: no cover — surface failure to parent
        queue.put((label, type(e).__name__, str(e)[:200]))


def append_embeddings(path: str, label: str, n: int, dim: int, queue) -> None:
    """Append `n` per-IRI vectors in a fresh process."""
    try:
        from turtlelake import Dataset

        ds = Dataset.open(path)
        iris = [f"https://ex/{label}/{i}" for i in range(n)]
        vecs = [[float(i % 7) / 7.0] * dim for i in range(n)]
        ds.embed(iris, vecs, model_id=f"test:{label}")
        queue.put((label, "ok", None))
    except Exception as e:  # pragma: no cover
        queue.put((label, type(e).__name__, str(e)[:200]))


def compact_dataset(path: str, queue) -> None:
    """Compact in a fresh process, concurrent with reads / writes."""
    try:
        from turtlelake import Dataset

        ds = Dataset.open(path)
        result = ds.compact()
        queue.put(("compact", "ok", str(result)[:200]))
    except Exception as e:  # pragma: no cover
        queue.put(("compact", type(e).__name__, str(e)[:200]))


def read_count(path: str, queue) -> None:
    """Open and count, concurrent with another writer."""
    try:
        from turtlelake import Dataset

        ds = Dataset.open(path)
        queue.put(("reader", "ok", str(ds.count())))
    except Exception as e:  # pragma: no cover
        queue.put(("reader", type(e).__name__, str(e)[:200]))
