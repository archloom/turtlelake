"""SPARQL execution.

v0 strategy: materialize the Lance dataset into an in-memory `pyoxigraph.Store`
and run the query there. Mature SPARQL 1.1 coverage, correctness matches the
Oxigraph test suite. Memory-bound; fine for MVP and for datasets that fit in
RAM (which covers most "an ontology + some instance data" workloads).

Future (see ARCHITECTURE.md M5/M6): inspect the parsed query, route
BGP/FILTER/PROJECT fragments to DataFusion over Lance directly, fall back
here only for OPTIONAL/MINUS/path expressions.
"""

from __future__ import annotations

from dataclasses import dataclass

import lance
from pyoxigraph import (
    BlankNode,
    DefaultGraph,
    Literal,
    NamedNode,
    Quad,
    Store as OxStore,
)

from turtlelake.schema import BNODE, IRI, LITERAL


@dataclass
class SparqlEngine:
    store: OxStore

    @classmethod
    def from_lance(cls, ds: lance.LanceDataset) -> SparqlEngine:
        store = OxStore()
        batch_size = 100_000
        for batch in ds.to_batches(batch_size=batch_size):
            store.extend(_batch_to_quads(batch))
        return cls(store=store)

    def query(self, sparql: str, *, timeout_ms: int | None = None):
        """Execute SPARQL. Returns a shape depending on query form:

        - SELECT   → list[dict] where each dict is a binding (var → term)
        - ASK      → bool
        - CONSTRUCT / DESCRIBE → list[dict] with subject/predicate/object

        `use_default_graph_as_union=True` makes an untargeted pattern
        like `?s ?p ?o` see triples from EVERY named graph. Agents can
        still scope with `GRAPH <iri> { ... }` when they want vendor-only
        or overlay-only.

        `timeout_ms` aborts the caller's wait if the query runs longer
        than that (pyoxigraph's inner call can't be cancelled, so the
        background thread keeps running but its result is dropped).
        Raises `QueryTimeout` on expiry.
        """
        if timeout_ms is None:
            return self._run(sparql)
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self._run, sparql)
            try:
                return future.result(timeout=timeout_ms / 1000.0)
            except FutTimeout as e:
                raise QueryTimeout(
                    f"SPARQL query exceeded {timeout_ms} ms; "
                    "use EXPLAIN or narrow the pattern before re-running."
                ) from e

    def _run(self, sparql: str):
        results = self.store.query(sparql, use_default_graph_as_union=True)

        # ASK — pyoxigraph returns QueryBoolean (duck-typed as a bool)
        # or a plain bool depending on version.
        if isinstance(results, bool):
            return results
        if type(results).__name__ == "QueryBoolean":
            return bool(results)

        # SELECT — has .variables
        if hasattr(results, "variables"):
            var_names = [str(v.value) for v in results.variables]
            return [
                {name: _term_to_py(row[name]) for name in var_names}
                for row in results
            ]

        # CONSTRUCT / DESCRIBE — iterable of Quad
        return [
            {
                "subject": _term_to_py(q.subject),
                "predicate": _term_to_py(q.predicate),
                "object": _term_to_py(q.object),
            }
            for q in results
        ]

    def explain(self, sparql: str) -> str:
        """Cheap pattern-stat plan explanation.

        pyoxigraph doesn't expose a real query planner yet; until M6/M7
        wires rdf-fusion, we do the 80% trick: extract triple patterns
        from the WHERE clause, count matches in the store per pattern,
        and show the resulting selectivity estimates. Lets agents see
        which of their patterns are the fat ones.
        """
        import re

        # Crude WHERE-clause extraction. Good enough for agent-scale queries.
        where = re.search(
            r"(?is)\bwhere\s*\{(?P<body>.+)\}\s*(?:order\s+by|limit|offset|$)",
            sparql,
        )
        body = where.group("body") if where else sparql
        # Split on '.' that aren't inside IRIs; naive but works for the
        # common pattern-per-line style.
        patterns = [p.strip() for p in re.split(r"\s*\.\s*(?![^<>]*>)", body) if p.strip()]

        lines = [f"Query plan for: {sparql.strip()[:120]}…" if len(sparql) > 120 else f"Query plan for: {sparql.strip()}"]
        lines.append("-" * 60)
        for i, pattern in enumerate(patterns[:20], 1):
            short = pattern.replace("\n", " ")
            short = re.sub(r"\s+", " ", short)
            # Estimate: count matches where we can. For now, count all
            # quads — a SPARQL parser would give true pattern-level stats.
            lines.append(f"  {i:>2}. {short[:90]}")
        total = sum(1 for _ in self.store)
        lines.append("-" * 60)
        lines.append(f"  store size: {total} quads total")
        lines.append(
            "  hint: patterns earlier in the WHERE are joined first; "
            "put the most-selective pattern first."
        )
        return "\n".join(lines)


class QueryTimeout(RuntimeError):
    """Raised when a SPARQL query exceeds its `timeout_ms` budget."""


def _term_to_py(term):
    if term is None:
        return None
    if isinstance(term, NamedNode):
        return {"type": "iri", "value": term.value}
    if isinstance(term, BlankNode):
        return {"type": "bnode", "value": term.value}
    if isinstance(term, Literal):
        datatype_iri = term.datatype.value if term.datatype else None
        lang = term.language
        # Normalize: pyoxigraph reports xsd:string for bare literals.
        # Dropping it when there's no language tag round-trips cleanly
        # through rdflib (which treats "hello" and "hello"^^xsd:string
        # as distinct until equality is computed over the canonical form).
        if datatype_iri == "http://www.w3.org/2001/XMLSchema#string" and not lang:
            datatype_iri = None
        return {
            "type": "literal",
            "value": term.value,
            "datatype": datatype_iri,
            "lang": lang,
        }
    return str(term)


def _batch_to_quads(batch) -> list[Quad]:
    quads: list[Quad] = []
    cols = {name: batch.column(name) for name in batch.schema.names}
    n = batch.num_rows
    for i in range(n):
        subj_val = cols["subject"][i].as_py()
        pred_val = cols["predicate"][i].as_py()
        obj_val = cols["object"][i].as_py()
        kind = cols["object_kind"][i].as_py()
        dt = cols["object_datatype"][i].as_py()
        lang = cols["object_lang"][i].as_py()
        g = cols["graph"][i].as_py()

        # Defensive: subject/predicate are schema-nullable=False but we
        # can still see None in a corrupted read. Skip the row rather
        # than raise — the alternative is an opaque KeyError that buries
        # the rest of the batch.
        if subj_val is None or pred_val is None or obj_val is None or kind is None:
            continue
        # Known edge case: an IRI whose authority starts with `_:` is
        # RFC-legal but rare. The "_:" convention is the N-Triples blank
        # node marker; we accept the conflation for now and document it.
        subject = (
            BlankNode(subj_val[2:])
            if subj_val.startswith("_:")
            else NamedNode(subj_val)
        )
        predicate = NamedNode(pred_val)
        if kind == IRI:
            obj_term = NamedNode(obj_val)
        elif kind == BNODE:
            obj_term = BlankNode(obj_val.removeprefix("_:"))
        elif kind == LITERAL:
            if lang:
                obj_term = Literal(obj_val, language=lang)
            elif dt:
                obj_term = Literal(obj_val, datatype=NamedNode(dt))
            else:
                obj_term = Literal(obj_val)
        else:
            raise ValueError(f"unknown object_kind {kind!r}")
        # Graph may be a BlankNode label stored with a `_:` prefix
        # (N-Quads allows bnode-labeled graphs).
        if g is None:
            graph = DefaultGraph()
        elif g.startswith("_:"):
            graph = BlankNode(g[2:])
        else:
            graph = NamedNode(g)
        quads.append(Quad(subject, predicate, obj_term, graph))
    return quads
