"""TTL (and the other Oxigraph-supported formats) → Lance.

Parsing is delegated entirely to pyoxigraph — it wraps the maintained `oxttl`
Rust crate and covers Turtle, TriG, N-Triples, N-Quads, RDF/XML and JSON-LD.
We only handle the Arrow mapping.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pyarrow as pa
from pyoxigraph import BlankNode, Literal, NamedNode, Quad, RdfFormat, parse

from turtlelake.schema import BNODE, IRI, LITERAL, TRIPLE_SCHEMA

# Map file suffixes to pyoxigraph's RdfFormat enum members.
# `.owl` is treated as RDF/XML — that's how OWL ontologies are
# typically serialized in the wild (the W3C-recommended Turtle
# alternative `.ttl` already maps separately above).
_FORMAT_BY_SUFFIX = {
    ".ttl": RdfFormat.TURTLE,
    ".trig": RdfFormat.TRIG,
    ".nt": RdfFormat.N_TRIPLES,
    ".nq": RdfFormat.N_QUADS,
    ".rdf": RdfFormat.RDF_XML,
    ".xml": RdfFormat.RDF_XML,
    ".owl": RdfFormat.RDF_XML,
    ".jsonld": RdfFormat.JSON_LD,
}


def _term_subject(term: NamedNode | BlankNode) -> tuple[str, str]:
    if isinstance(term, NamedNode):
        return term.value, IRI
    return term.value, BNODE  # BlankNode


def _term_object(term):
    if isinstance(term, NamedNode):
        return term.value, IRI, None, None
    if isinstance(term, BlankNode):
        return term.value, BNODE, None, None
    # Literal
    lit: Literal = term
    return lit.value, LITERAL, (lit.datatype.value if lit.datatype else None), lit.language


def quads_to_record_batch(quads: Iterable[Quad]) -> pa.RecordBatch:
    """Convert an iterable of pyoxigraph Quads into one Arrow RecordBatch.

    Streaming note: callers ingesting large files should chunk quads (e.g. 50k
    at a time) and write one batch per chunk so Lance can checkpoint.
    """
    s, p, o = [], [], []
    okind, odt, olang, g = [], [], [], []
    for q in quads:
        subj, kind_s = _term_subject(q.subject)
        obj, kind, dt, lang = _term_object(q.object)
        s.append(f"_:{subj}" if kind_s == BNODE else subj)
        p.append(q.predicate.value)
        o.append(f"_:{obj}" if kind == BNODE else obj)
        okind.append(kind)
        odt.append(dt)
        olang.append(lang)
        # Named graphs: IRI-named → URI, bnode-named → "_:<label>",
        # default graph → null. This mirrors the blank-node storage
        # convention used for subjects/objects.
        if isinstance(q.graph_name, NamedNode):
            g.append(q.graph_name.value)
        elif isinstance(q.graph_name, BlankNode):
            g.append(f"_:{q.graph_name.value}")
        else:
            g.append(None)
    return pa.RecordBatch.from_arrays(
        [
            pa.array(s, type=pa.string()),
            pa.array(p, type=pa.string()),
            pa.array(o, type=pa.string()),
            pa.array(okind, type=pa.string()),
            pa.array(odt, type=pa.string()),
            pa.array(olang, type=pa.string()),
            pa.array(g, type=pa.string()),
        ],
        schema=TRIPLE_SCHEMA,
    )


def parse_rdf_file(path: Path, format: RdfFormat | None = None) -> Iterable[Quad]:
    """Yield Quads from any RDF file format Oxigraph understands."""
    fmt = format or _FORMAT_BY_SUFFIX.get(path.suffix.lower())
    if fmt is None:
        raise ValueError(
            f"Cannot infer RDF format from suffix {path.suffix!r}; "
            "pass format=RdfFormat.TURTLE (or similar) explicitly."
        )
    with path.open("rb") as fh:
        yield from parse(fh, format=fmt)
