"""Zero-dependency GraphRAG demo: ingest an ontology, embed entity
labels with a deterministic hash projection, run `graph_rag`.

This file exists so a new user can `git clone && uv run python
examples/embed_with_hash.py` and see GraphRAG end-to-end without
installing torch / sentence-transformers / an API key.

The hash embedding has no semantic meaning — it just maps each label
to the same vector every time. Useful for plumbing demos and tests;
for real retrieval, copy `embed_with_sentence_transformers.py`
instead.

Run:
    uv run python examples/embed_with_hash.py
"""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

from turtlelake import Dataset

# ── 1. Tiny deterministic "embedding" function ───────────────────────

EMBEDDING_DIM = 32


def _hash_embed(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    """Produce a stable `dim`-float vector from `text`. Not semantic —
    just a hash unrolled into floats. Same input → same vector across
    runs and processes."""
    out: list[float] = []
    seed = text.encode("utf-8")
    counter = 0
    while len(out) < dim:
        h = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
        # Take 4 bytes at a time, interpret as float in [0,1).
        for offset in range(0, len(h), 4):
            if len(out) >= dim:
                break
            (n,) = struct.unpack(">I", h[offset:offset + 4])
            out.append(n / 2**32)
        counter += 1
    # Center around 0 so distances are not all biased positive.
    return [v - 0.5 for v in out]


# ── 2. Open / create the dataset (small inline ontology) ────────────

HERE = Path(__file__).parent
KG_PATH = HERE / "_hash_demo.turtlelake"

DEMO_TTL = """\
@prefix ex:   <https://example.org/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

ex:scifi    a ex:Genre  ; rdfs:label "Science Fiction" .
ex:mystery  a ex:Genre  ; rdfs:label "Mystery" .
ex:fantasy  a ex:Genre  ; rdfs:label "Fantasy" .

ex:asimov     a ex:Author ; rdfs:label "Isaac Asimov" .
ex:christie   a ex:Author ; rdfs:label "Agatha Christie" .
ex:tolkien    a ex:Author ; rdfs:label "J. R. R. Tolkien" .

ex:foundation     a ex:Book ; rdfs:label "Foundation" ;
                  ex:author ex:asimov ; ex:genre ex:scifi .
ex:robotsOfDawn   a ex:Book ; rdfs:label "The Robots of Dawn" ;
                  ex:author ex:asimov ; ex:genre ex:scifi .
ex:roger          a ex:Book ; rdfs:label "The Murder of Roger Ackroyd" ;
                  ex:author ex:christie ; ex:genre ex:mystery .
ex:fellowship     a ex:Book ; rdfs:label "The Fellowship of the Ring" ;
                  ex:author ex:tolkien ; ex:genre ex:fantasy .
"""

kg = Dataset.open(KG_PATH)
if kg.count() == 0:
    kg.insert_turtle(DEMO_TTL, source="hash-demo:books")

# ── 3. Embed every labelled entity ───────────────────────────────────

rows = kg.query(
    """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?iri ?label WHERE { ?iri rdfs:label ?label . }
    """
)
iris = [r["iri"]["value"] for r in rows]
texts = [r["label"]["value"] for r in rows]
vectors = [_hash_embed(t) for t in texts]
print(f"hash-embedding {len(iris)} entities (dim={EMBEDDING_DIM})")

# Idempotent: skip if we have any embeddings under this model_id already.
if kg.embedding_count() == 0:
    kg.embed(iris, vectors, model_id="demo:hash-32")

# ── 4. GraphRAG: query by meaning, get back facts ────────────────────

question = "Foundation"   # hash-embedding is text-similarity-ish only
q_vec = _hash_embed(question)
out = kg.graph_rag(q_vec, k=3, hops=1, model_id="demo:hash-32")

print("\n=== top hits ===")
for hit in out["hits"]:
    print(f"  {hit['iri']}  (distance={hit['distance']:.4f})")
    facts = out["entities"][hit["iri"]]["outgoing"][:4]
    for f in facts:
        obj = f["object"]
        val = obj.get("value", obj) if isinstance(obj, dict) else obj
        print(f"    {f['predicate'].rsplit('/', 1)[-1]} = {val}")

print(
    "\nThis demo uses a hash 'embedding' with no semantic meaning.\n"
    "For real retrieval, see embed_with_sentence_transformers.py."
)
