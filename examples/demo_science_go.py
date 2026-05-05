"""Scientific-AI grounding demo: the Gene Ontology (GO).

GO is the canonical ontology of biological function (~50k terms across
biological_process, molecular_function, cellular_component). It's
massive — 35 MB OBO — and ships in OBO format, which pyoxigraph does
not parse natively. The demo:

  1. Downloads the full GO (cached, one-time ~30 s).
  2. Parses it via the `pronto` library.
  3. Subsets to terms reachable from a seed (default: 'apoptotic
     process') within N hops along subClassOf + RO:0002211 (regulates).
  4. Emits a small TTL slice and ingests THAT into turtlelake.

This pattern — "load the big public ontology once, subset for the
question at hand" — is how production scientific agents handle GO,
ChEBI, MONDO, etc. without paying full-corpus ingest cost on every
query.

The story we walk through:

  USER: "What biological processes regulate programmed cell death?"

  An ungrounded LLM lists processes from training data with no
  guarantee they're real GO terms or actually regulate apoptosis.
  The grounded agent walks the actual GO regulates+/subClassOf+
  graph and returns a verifiable list.

Run:
    uv run python examples/demo_science_go.py
    uv run python examples/demo_science_go.py --seed "DNA repair" --hops 2
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from turtlelake import Dataset

sys.path.insert(0, str(Path(__file__).parent))
from _demo_runner import (  # noqa: E402
    banner, download, grounded, naive, section, shows,
)

GO_URL = (
    "https://raw.githubusercontent.com/geneontology/go-ontology/master/"
    "src/ontology/go-edit.obo"
)

OBO = "http://purl.obolibrary.org/obo/"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
RO_REGULATES = "http://purl.obolibrary.org/obo/RO_0002211"


def _go_iri(go_id: str) -> str:
    """Map a 'GO:0006915' style id to the canonical OBO IRI."""
    return f"{OBO}{go_id.replace(':', '_')}"


def _subset_go_to_ttl(
    obo_path: Path, *, seeds: list[str], hops: int
) -> tuple[Path, int]:
    """Parse the full GO via pronto, BFS from `seeds` for `hops`
    levels along subClassOf + regulates edges, write the resulting
    subset to TTL. Returns (path, term_count)."""
    import pronto

    print(f"  parsing {obo_path.name} via pronto...", file=sys.stderr)
    # `import_depth=0` skips owl:imports — those would try to fetch
    # external ontologies (RO, BFO) that may be unreachable. We get
    # what we need from the GO OBO file alone.
    o = pronto.Ontology(str(obo_path), import_depth=0)
    print(f"  parsed: {len(list(o.terms()))} terms", file=sys.stderr)

    # Collect IRIs of seed terms by name (case-insensitive).
    seed_terms: list = []
    for s in seeds:
        for t in o.terms():
            if t.name and t.name.lower() == s.lower():
                seed_terms.append(t)
                break
        else:
            print(f"  warning: seed '{s}' not found by exact name; "
                  "ignoring", file=sys.stderr)

    if not seed_terms:
        raise SystemExit(f"no seeds found: {seeds}")

    # BFS along subClassOf (both directions) + every typed relation.
    visited: set = set()
    frontier = list(seed_terms)
    for _ in range(hops):
        next_frontier = []
        for t in frontier:
            if t.id in visited:
                continue
            visited.add(t.id)
            try:
                for p in t.superclasses(distance=1):
                    if p.id != t.id and p.id not in visited:
                        next_frontier.append(p)
            except KeyError:
                pass
            try:
                for c in t.subclasses(distance=1):
                    if c.id != t.id and c.id not in visited:
                        next_frontier.append(c)
            except KeyError:
                pass
            for _kind, related in t.relationships.items():
                try:
                    related_list = list(related)
                except KeyError:
                    continue
                for r in related_list:
                    if r.id not in visited:
                        next_frontier.append(r)
        frontier = next_frontier

    # Inverse-regulates scan: GO models "X regulates Y" as a property
    # on X, so to find regulators of Y we scan everything once. Cheap
    # at 50k terms; we only walk the relationships dict per term.
    seed_set = {t.id for t in seed_terms}
    expanded_set = set(visited) | seed_set
    print("  scanning all terms for regulators of seeds...",
          file=sys.stderr)
    regulator_pairs: list[tuple[str, str]] = []
    for t in o.terms():
        for kind, targets in t.relationships.items():
            if "regulates" not in (kind.id or ""):
                continue
            try:
                target_list = list(targets)
            except KeyError:
                # GO references some external ontology terms (OBA, BFO,
                # …) we didn't import. Skip those targets.
                continue
            for r in target_list:
                if r.id in expanded_set:
                    regulator_pairs.append((t.id, r.id))
                    visited.add(t.id)
    print(f"  subset: {len(visited)} terms ({len(regulator_pairs)} "
          f"regulator edges) across {hops} hops",
          file=sys.stderr)

    out_path = obo_path.with_suffix(
        f".subset-{len(visited)}-hops{hops}.ttl"
    )
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(f"@prefix rdfs: <{RDFS}> .\n")
        fh.write(f"@prefix obo: <{OBO}> .\n")
        fh.write("@prefix owl: <http://www.w3.org/2002/07/owl#> .\n\n")
        for term_id in visited:
            t = o[term_id]
            iri = _go_iri(term_id)
            fh.write(f'<{iri}> a owl:Class .\n')
            if t.name:
                fh.write(f'<{iri}> rdfs:label "{_escape(t.name)}" .\n')
            if t.definition:
                fh.write(
                    f'<{iri}> obo:IAO_0000115 "{_escape(str(t.definition))}" .\n'
                )
            if t.namespace:
                fh.write(
                    f'<{iri}> obo:hasOBONamespace "{_escape(t.namespace)}" .\n'
                )
            for p in t.superclasses(distance=1):
                if p.id != t.id and p.id in visited:
                    fh.write(f'<{iri}> rdfs:subClassOf <{_go_iri(p.id)}> .\n')
        # Emit regulator edges (collected via the inverse scan).
        for src_id, tgt_id in regulator_pairs:
            if src_id in visited and tgt_id in visited:
                fh.write(
                    f'<{_go_iri(src_id)}> <{RO_REGULATES}> '
                    f'<{_go_iri(tgt_id)}> .\n'
                )
    return out_path, len(visited)


def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def _label_of(ds: Dataset, iri: str) -> str:
    rows = ds.query(
        f"PREFIX rdfs: <{RDFS}> "
        f"SELECT ?l WHERE {{ <{iri}> rdfs:label ?l }} LIMIT 1"
    )
    return rows[0]["l"]["value"] if rows else iri.rsplit("/", 1)[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="apoptotic process",
                    help="seed term name; default 'apoptotic process'")
    ap.add_argument("--hops", type=int, default=2,
                    help="BFS depth from the seed (default 2)")
    args = ap.parse_args()

    banner("SCIENCE — agent grounded against the Gene Ontology")

    with tempfile.TemporaryDirectory() as td:
        ds = Dataset.open(Path(td) / "science.turtlelake")

        section("setup: download GO (35 MB; cached after first run)")
        obo_path = download(GO_URL, suffix=".obo")
        print(f"  GO at {obo_path} ({obo_path.stat().st_size // 1024 // 1024} MB)")

        section(f"setup: subset to terms ≤{args.hops} hops from "
                f"'{args.seed}' (real-world pattern)")
        ttl_path, n = _subset_go_to_ttl(
            obo_path, seeds=[args.seed], hops=args.hops
        )
        ds.ingest_ttl(ttl_path, source="go:subset", author="Gene Ontology")
        print(f"  ingested {ds.count()} quads from {n} terms")

        section("setup: build text index over labels + definitions")
        text_info = ds.preload_text_index(
            predicates=[f"{RDFS}label", f"{OBO}IAO_0000115"]
        )
        print(f"  text index: {text_info['rows']} entities, "
              f"{text_info['vocab_size']} vocab terms")

        section("user question")
        user = "What biological processes regulate programmed cell death?"
        print(f'  USER: "{user}"')

        # ── 1. Resolve the seed concept to a real GO term ──
        section(f"step 1 — resolve '{args.seed}' → real GO term")
        naive(
            "answers without checking; risks confusing closely related "
            "concepts (e.g. 'apoptotic process' vs 'programmed cell death')."
        )
        # Prefer an exact-label match before falling back to lexical
        # similarity. This is the production pattern: try the cheap
        # deterministic resolution first, defer to BM25 / vector only
        # when it fails.
        exact = ds.query(
            f"PREFIX rdfs: <{RDFS}> "
            "SELECT ?s WHERE { ?s rdfs:label ?l . FILTER(LCASE(STR(?l)) = "
            f'"{args.seed.lower()}") }} LIMIT 1'
        )
        if exact:
            primary = exact[0]["s"]["value"]
            primary_label = _label_of(ds, primary)
            go_id = "GO:" + primary.split("GO_")[-1]
            grounded(f"exact-label match: {go_id:14}  →  {primary_label}")
        else:
            hits = ds.bm25_search(args.seed, k=3)
            primary = hits[0]["iri"] if hits else _go_iri("GO:0012501")
            primary_label = _label_of(ds, primary)
            go_id = "GO:" + primary.split("GO_")[-1]
            for h in hits[:3]:
                iri = h["iri"]
                lbl = _label_of(ds, iri)
                gid = "GO:" + iri.split("GO_")[-1]
                grounded(f"BM25 match: {gid:14}  →  {lbl}")
        shows("bm25_search", text="lexical match against rdfs:label + IAO definition")

        # ── 2. Find regulators via the typed regulates relation ──
        section(f"step 2 — what regulates {primary_label}? (SPARQL on RO:0002211)")
        naive(
            "freely lists 'p53', 'BCL-2 family', 'caspases' — true facts "
            "but mixed with hallucinations; no guarantee of GO IDs."
        )
        regulators = ds.query(f"""
            PREFIX rdfs: <{RDFS}>
            SELECT ?r ?label WHERE {{
                ?r <{RO_REGULATES}> <{primary}> .
                ?r rdfs:label ?label .
            }} LIMIT 12
        """)
        if regulators:
            for r in regulators:
                go_id = "GO:" + r["r"]["value"].split("GO_")[-1]
                grounded(f"{go_id:14}  →  {r['label']['value']}")
        else:
            grounded("(no direct regulators in this subset; widen --hops)")
        shows(
            "query (typed SPARQL pattern)",
            text="every result is a verifiable GO ID; nothing hallucinated",
        )

        # ── 3. Class hierarchy traversal — broader/narrower terms ──
        section(f"step 3 — narrower subtypes of {primary_label}")
        naive(
            "may list paraphrases instead of GO subtypes; loses the "
            "is_a hierarchy needed for downstream enrichment analysis."
        )
        narrower = ds.query(f"""
            PREFIX rdfs: <{RDFS}>
            SELECT ?s ?label WHERE {{
                ?s rdfs:subClassOf+ <{primary}> .
                ?s rdfs:label ?label .
            }} ORDER BY ?label LIMIT 8
        """)
        for s in narrower:
            go_id = "GO:" + s["s"]["value"].split("GO_")[-1]
            grounded(f"⤵ {go_id:14}  →  {s['label']['value']}")
        shows(
            "query (subClassOf+)",
            text="property-path traversal — exhaustive subtype enumeration",
        )

        # ── 4. Authoritative definition for the LLM context window ──
        section("step 4 — authoritative definition")
        defn_rows = ds.query(
            f"SELECT ?d WHERE {{ <{primary}> <{OBO}IAO_0000115> ?d }} LIMIT 1"
        )
        defn = defn_rows[0]["d"]["value"] if defn_rows else "(no definition)"
        grounded(f'"{defn[:200]}..."')
        shows(
            "query",
            text="IAO:0000115 is the OBO Foundry's standard definition predicate",
        )

        section("summary")
        print(
            "  This demo grounded a scientific question against the\n"
            "  Gene Ontology — 50k terms — by subsetting to a relevant\n"
            "  neighborhood at ingest time. The agent's queries returned\n"
            "  exact GO IDs, real regulator relations, and authoritative\n"
            "  definitions, with no risk of hallucinated terms.\n"
            "\n"
            "  In production: pre-build a turtlelake directory per\n"
            "  research domain (apoptosis, immune response, …), ship\n"
            "  alongside the agent. Pair with ChEBI for compounds and\n"
            "  UniProt for proteins via named graphs."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
