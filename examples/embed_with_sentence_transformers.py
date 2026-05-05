"""Cold-start GraphRAG: ingest an ontology, embed entity labels with
sentence-transformers, run a query, and let `graph_rag` do retrieval.

This is the production-shaped pipeline. If you're new to turtlelake,
this is the file to copy and adapt.

Prereqs:
    pip install "turtlelake[mcp]" sentence-transformers

Run:
    uv run python examples/embed_with_sentence_transformers.py
"""

from __future__ import annotations

from pathlib import Path

from turtlelake import Dataset

# ── 1. Open / create the dataset (small inline ontology) ────────────
# The TTL is inlined so the demo runs on a fresh clone with no
# external download. For domain-specific demos with real public
# ontologies see `demo_legal_lkif.py`, `demo_medical_doid.py`,
# `demo_science_go.py`, `demo_gov_dcat.py`.
HERE = Path(__file__).parent
KG_PATH = HERE / "_st_demo.turtlelake"

DEMO_TTL = """\
@prefix ex:   <https://example.org/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

ex:scifi    a ex:Genre  ; rdfs:label "Science Fiction" .
ex:mystery  a ex:Genre  ; rdfs:label "Mystery" .
ex:fantasy  a ex:Genre  ; rdfs:label "Fantasy" .

ex:asimov   a ex:Author ; rdfs:label "Isaac Asimov" .
ex:christie a ex:Author ; rdfs:label "Agatha Christie" .
ex:tolkien  a ex:Author ; rdfs:label "J. R. R. Tolkien" .

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
    kg.insert_turtle(DEMO_TTL, source="st-demo:books")

# ── 2. Pick the entities to embed ────────────────────────────────────
# We embed every IRI that has an `rdfs:label`. In your project you'd
# embed entity descriptions or summaries — the literal label is fine
# for a quickstart but only carries surface meaning.
rows = kg.query(
    """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?iri ?label WHERE {
        ?iri rdfs:label ?label .
    }
    """
)
iris = [r["iri"]["value"] for r in rows]
texts = [r["label"]["value"] for r in rows]
print(f"found {len(iris)} entities to embed")

# ── 3. Embed with sentence-transformers ──────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "This example needs sentence-transformers.\n"
        "  pip install sentence-transformers\n"
        "Or copy examples/embed_with_hash.py for a zero-dep variant."
    ) from e

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)
vectors = model.encode(texts, normalize_embeddings=True).tolist()

# Skip rows we've already embedded with this model — embed() will
# accept duplicates, but on a real corpus you want to avoid the cost.
already = {h["iri"] for h in kg.vector_search(vectors[0], k=10**6, model_id=MODEL_NAME)} \
    if kg.embedding_count() else set()
new_iris, new_vecs = zip(
    *((i, v) for i, v in zip(iris, vectors) if i not in already)
) if any(i not in already for i in iris) else ([], [])
if new_iris:
    n = kg.embed(list(new_iris), list(new_vecs), model_id=MODEL_NAME)
    print(f"embedded {n} new vectors with {MODEL_NAME}")

# ── 4. Build an ANN index when the corpus is big enough ──────────────
# `index_type='auto'` is a no-op below ~10k vectors (brute-force scan
# is already sub-millisecond at that scale). Calling it anyway is
# harmless and keeps the script working as your KG grows.
status = kg.build_vector_index()
print(f"index: {status['action']} ({status.get('reason')})")

# ── 5. Ask a question via GraphRAG ───────────────────────────────────
question = "robots in space"   # semantic match → Foundation, Robots of Dawn
q_vec = model.encode([question], normalize_embeddings=True)[0].tolist()
out = kg.graph_rag(q_vec, k=3, hops=1, model_id=MODEL_NAME)

print("\n=== top hits ===")
for hit in out["hits"]:
    print(f"  {hit['iri']}  (distance={hit['distance']:.4f})")
    facts = out["entities"][hit["iri"]]["outgoing"]
    for f in facts[:3]:
        obj = f["object"]
        val = obj.get("value", obj)
        print(f"    {f['predicate'].rsplit('/', 1)[-1]} = {val}")
