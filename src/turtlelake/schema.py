"""Arrow schemas that define a turtlelake dataset.

Two sibling Lance datasets live under one turtlelake directory:

  triples.lance/      one row per RDF quad      (TRIPLE_SCHEMA)
  embeddings.lance/   one row per IRI vector    (EMBEDDING_SCHEMA, optional)

The vector layer is a separate dataset rather than a column on triples
so it can have its own version cadence (re-embed without rewriting the
graph), its own compression, and an optional dependency footprint.
Joins between the two happen by IRI at query time, not at storage time.

Strings are intentionally not dictionary-encoded at the API layer —
Lance's own encodings handle compression, and a flat string schema stays
readable to every Arrow consumer (Polars, DuckDB, DataFusion) without
extra metadata.
"""

from __future__ import annotations

import pyarrow as pa

IRI = "iri"
BNODE = "bnode"
LITERAL = "literal"

TRIPLE_SCHEMA = pa.schema(
    [
        pa.field("subject", pa.string(), nullable=False),
        pa.field("predicate", pa.string(), nullable=False),
        pa.field("object", pa.string(), nullable=False),
        pa.field("object_kind", pa.string(), nullable=False),   # "iri" | "bnode" | "literal"
        pa.field("object_datatype", pa.string(), nullable=True),
        pa.field("object_lang", pa.string(), nullable=True),
        pa.field("graph", pa.string(), nullable=True),          # null = default graph
    ]
)

TRIPLES_TABLE = "triples"


def embedding_schema(dim: int) -> pa.Schema:
    """Build the embedding schema for a given vector dimension.

    The vector dimension is fixed per dataset (Lance's `fixed_size_list`
    encoding requires it). `dim` is recorded in `manifest.json` so
    re-opening a directory uses the same dim without re-introspecting.

      iri        : the subject IRI the vector embeds (foreign key into triples)
      vector     : fixed_size_list<float32>[dim]
      model_id   : free-form identifier for the embedding model that
                   produced this vector (e.g. "openai:text-embedding-3-small",
                   "sentence-transformers/all-MiniLM-L6-v2"). Versioning
                   embeddings by model is the caller's responsibility — we
                   don't enforce uniqueness on (iri, model_id), but
                   `vector_search` can filter on it.
      created_at : write timestamp (UTC, microsecond precision)
    """
    return pa.schema(
        [
            pa.field("iri", pa.string(), nullable=False),
            pa.field("vector", pa.list_(pa.float32(), dim), nullable=False),
            pa.field("model_id", pa.string(), nullable=False),
            pa.field("created_at", pa.timestamp("us", tz="UTC"), nullable=False),
        ]
    )


EMBEDDINGS_TABLE = "embeddings"
