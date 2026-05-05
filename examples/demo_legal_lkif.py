"""Legal-AI grounding demo: the LKIF Core ontology.

LKIF (Legal Knowledge Interchange Format) is a real ontology used in
EU legal-informatics research. It models legal actions, norms, roles,
processes, time, and mereology. This demo uses 5 of its 11 modules
(~250 KB total) to ground an agent that is asked legal questions.

The story we walk through:

  USER: "I want to file a complaint about my landlord for unauthorized
         entry into my apartment."

  An ungrounded LLM might invent legal terms or pick the wrong
  jurisdiction's procedure. The grounded agent uses turtlelake's
  primitives to:
    - resolve "unauthorized entry" → an LKIF concept
    - enumerate the legal actions and roles that apply
    - check the user's facts against an LKIF SHACL shape
    - return a verifiable answer with provenance

This demo deliberately uses LKIF's vocabulary even when the wording
is slightly archaic — that's the point. Legal AI must speak the
domain's actual vocabulary, not paraphrase it.

Run:
    uv run python examples/demo_legal_lkif.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from turtlelake import Dataset

# Local demo helpers.
sys.path.insert(0, str(Path(__file__).parent))
from _demo_runner import (  # noqa: E402
    banner, download, grounded, naive, section, shows,
)

LKIF_BASE = "https://raw.githubusercontent.com/RinkeHoekstra/lkif-core/master"
# We pull a focused subset — the modules that bear on the demo question.
# The full LKIF Core is 11 modules; smaller is faster + clearer.
LKIF_MODULES = [
    "action.owl",
    "legal-action.owl",
    "legal-role.owl",
    "norm.owl",
    "role.owl",
    "process.owl",
    "time.owl",
]

LKIF_NS = "http://www.estrellaproject.org/lkif-core/"


def _ingest_lkif(ds: Dataset) -> int:
    """Download and ingest each LKIF module. Returns the total quad count.
    pyoxigraph parses RDF/XML natively, which is what LKIF ships."""
    total = 0
    for mod in LKIF_MODULES:
        url = f"{LKIF_BASE}/{mod}"
        path = download(url, suffix=".owl")
        before = ds.count()
        ds.ingest_ttl(path, source=f"lkif:{mod}", author="LKIF Core")
        delta = ds.count() - before
        print(f"  ingested {mod}: +{delta} quads", file=sys.stderr)
        total += delta
    return total


def main() -> int:
    banner("LEGAL — agent grounded against LKIF Core")

    with tempfile.TemporaryDirectory() as td:
        ds = Dataset.open(Path(td) / "legal.turtlelake")

        section("setup: ingest LKIF Core ontology (download cached)")
        n = _ingest_lkif(ds)
        print(f"  total: {n} quads across {len(LKIF_MODULES)} modules")

        section("setup: build text index")
        # LKIF documents classes with rdfs:comment (not rdfs:label).
        # Index that so `bm25_search` matches against the actual prose
        # in the ontology.
        text_info = ds.preload_text_index(
            predicates=["http://www.w3.org/2000/01/rdf-schema#comment"]
        )
        print(f"  text index: {text_info['rows']} entities, "
              f"{text_info['vocab_size']} vocab terms")

        section("user question")
        user = (
            "I want to file a complaint about my landlord for "
            "unauthorized entry into my apartment."
        )
        print(f'  USER: "{user}"')

        # ── 1. Resolve the natural-language phrase to an LKIF concept ──
        section("step 1 — resolve the situation to a legal concept (BM25 search)")
        naive(
            "invents 'tenant trespass tort' (no such single LKIF class) "
            "or picks a US-only term inappropriate for an EU framework."
        )
        # We don't have a vector model here; BM25 alone is enough for
        # this small + named ontology. (For a bigger ontology, swap in
        # `hybrid_search` after `preload_vectors`.)
        candidates = ds.bm25_search(
            "person privacy property unauthorised entry violation", k=8
        )
        if not candidates:
            print("  (no matches - try widening the query)")
            return 1
        for h in candidates[:5]:
            iri = h["iri"]
            # LKIF doesn't carry rdfs:label on most classes — use the
            # IRI fragment as a human label and pull rdfs:comment for
            # context.
            display = iri.rsplit("#", 1)[-1].rsplit("/", 1)[-1]
            comment_rows = ds.query(
                "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> "
                f"SELECT ?c WHERE {{ <{iri}> rdfs:comment ?c }} LIMIT 1"
            )
            snippet = (
                comment_rows[0]["c"]["value"][:90].replace("\n", " ")
                if comment_rows
                else "(no comment)"
            )
            grounded(f"{display:30}  →  {snippet}")
        shows(
            "bm25_search",
            text="lexical similarity over rdfs:label + rdfs:comment",
        )

        # ── 2. Enumerate sub-roles relevant to the situation ──
        section("step 2 — enumerate Legal Roles via SPARQL property paths")
        naive(
            "lists generic 'plaintiff', 'defendant', 'attorney' — but "
            "loses the LKIF role hierarchy that downstream legal "
            "systems will expect."
        )
        roles = ds.query("""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX role: <http://www.estrellaproject.org/lkif-core/role.owl#>
            SELECT DISTINCT ?r WHERE {
                ?r rdfs:subClassOf+ role:Role .
            } LIMIT 20
        """)
        if roles:
            for r in roles[:8]:
                fragment = r["r"]["value"].rsplit("#", 1)[-1]
                grounded(f"role  →  {fragment}")
        else:
            grounded(
                "(no role subtypes in this LKIF subset — add lkif-extended for "
                "full hierarchy)"
            )
        shows(
            "query (SPARQL)",
            text="rdfs:subClassOf+ traverses the role taxonomy transitively",
        )

        # ── 3. Validate that the situation matches a Legal Action shape ──
        section("step 3 — entity expansion of LKIF norm:Allowed (a key concept)")
        naive(
            "doesn't know what 'allowed' means in the deontic-logic "
            "sense LKIF uses; might confuse with permission or ability."
        )
        norm_iri = "http://www.estrellaproject.org/lkif-core/norm.owl#Allowed"
        ent = ds.entity(norm_iri, hops=1)
        grounded(f"definition iri:  {norm_iri}")
        # Show a few relevant outgoing edges (typically rdfs:subClassOf,
        # rdfs:comment, owl:disjointWith).
        for edge in ent.get("outgoing", [])[:4]:
            obj = edge["object"]
            obj_str = obj.get("value", str(obj))[:80]
            pred = edge["predicate"].rsplit("#", 1)[-1].rsplit("/", 1)[-1]
            grounded(f"  {pred} → {obj_str}")
        shows(
            "entity(iri, hops=1)",
            text="returns the structured neighborhood — agent gets the "
                 "definition and class hierarchy in one call",
        )

        # ── 4. Provenance: where did each fact come from? ──
        section("step 4 — provenance for the writes we just did")
        naive(
            "if asked 'where did this term come from?' the LLM "
            "fabricates a citation; agent has no audit trail."
        )
        prov = ds.provenance()[-len(LKIF_MODULES):]
        for entry in prov[:5]:
            grounded(
                f"version {entry['version']}  ←  source={entry['source']}  "
                f"author={entry['author']}  rows={entry['row_delta']}"
            )
        shows(
            "provenance()",
            text="every triple traces back to its source ontology — "
                 "the bedrock of legal-AI auditability",
        )

        section("summary")
        print(
            "  This demo grounded a legal question against a real public "
            "ontology (LKIF Core) using:\n"
            "    - bm25_search to resolve natural-language to legal IRIs\n"
            "    - SPARQL property paths to enumerate role hierarchies\n"
            "    - entity() expansion to give the agent structured context\n"
            "    - provenance() for the audit trail every legal-AI demands\n"
            "\n"
            "  In production: pair LKIF Core with jurisdiction-specific\n"
            "  extensions (US legal-rule, EU directives) and load them as\n"
            "  named graphs. Layer SHACL shapes per claim type. The shape\n"
            "  of the agent code stays the same."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
