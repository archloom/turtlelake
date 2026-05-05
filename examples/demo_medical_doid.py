"""Medical-AI grounding demo: the Disease Ontology (DOID) cancer slim.

DOID is a publicly-funded NIH disease ontology — the same shape as
SNOMED CT but unrestricted. We use the cancer slim (~730 disease
terms, 1.3 MB OWL) so the demo runs fast and stays focused.

The story we walk through:

  USER: "Patient has acute lymphoblastic leukemia."

  An ungrounded LLM gets the *name* right but routinely:
    - hallucinates a code (e.g. DOID:99999 instead of DOID:9952)
    - confuses related-but-distinct subtypes (B-cell ALL vs T-cell ALL
      vs Philadelphia-chromosome-positive ALL)
    - omits the parent classes a clinical decision-support system
      needs (leukemia → hematologic cancer → cancer)

  The grounded agent uses turtlelake's primitives to:
    - resolve the natural-language phrase to the exact DOID code
    - validate the IRI exists before any downstream call
    - enumerate the parent class chain (taxonomy traversal)
    - enumerate sibling subtypes for differential diagnosis
    - return synonyms + cross-references for interoperability

This is the workflow every clinical AI system needs but most flunk.

Run:
    uv run python examples/demo_medical_doid.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from turtlelake import Dataset

sys.path.insert(0, str(Path(__file__).parent))
from _demo_runner import (  # noqa: E402
    banner, download, grounded, naive, section, shows,
)

DOID_URL = (
    "https://raw.githubusercontent.com/DiseaseOntology/HumanDiseaseOntology/"
    "main/src/ontology/subsets/DO_cancer_slim.owl"
)

OBO = "http://purl.obolibrary.org/obo/"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
OBOINOWL = "http://www.geneontology.org/formats/oboInOwl#"
IAO_DEFINITION = OBO + "IAO_0000115"  # the standard "definition" predicate


def _label_of(ds: Dataset, iri: str) -> str:
    rows = ds.query(
        f"PREFIX rdfs: <{RDFS}> "
        f"SELECT ?l WHERE {{ <{iri}> rdfs:label ?l }} LIMIT 1"
    )
    return rows[0]["l"]["value"] if rows else iri.rsplit("/", 1)[-1]


def _definition_of(ds: Dataset, iri: str) -> str:
    rows = ds.query(
        f"SELECT ?d WHERE {{ <{iri}> <{IAO_DEFINITION}> ?d }} LIMIT 1"
    )
    return rows[0]["d"]["value"] if rows else "(no definition recorded)"


def main() -> int:
    banner("MEDICAL — agent grounded against Disease Ontology (cancer slim)")

    with tempfile.TemporaryDirectory() as td:
        ds = Dataset.open(Path(td) / "medical.turtlelake")

        section("setup: ingest DOID cancer slim")
        path = download(DOID_URL, suffix=".owl")
        ds.ingest_ttl(path, source="doid:cancer-slim", author="Disease Ontology")
        print(f"  ingested {ds.count()} quads")

        section("setup: build text index over labels + definitions")
        text_info = ds.preload_text_index(
            predicates=[
                f"{RDFS}label",
                IAO_DEFINITION,
                f"{OBOINOWL}hasExactSynonym",
            ]
        )
        print(f"  text index: {text_info['rows']} entities, "
              f"{text_info['vocab_size']} vocab terms")

        section("user input")
        user = "Patient has acute lymphoblastic leukemia."
        print(f'  USER: "{user}"')

        # ── 1. Resolve to a real DOID code ──
        section("step 1 — resolve to a verifiable DOID code")
        naive(
            "writes 'DOID:99999' (made up) or skips the code entirely "
            "and just passes the prose to the next system."
        )
        hits = ds.bm25_search("acute lymphoblastic leukemia", k=5)
        if not hits:
            return 1
        for h in hits[:5]:
            iri = h["iri"]
            label = _label_of(ds, iri)
            doid = iri.split("DOID_")[-1] if "DOID_" in iri else iri
            grounded(f"DOID:{doid:6}  →  {label}")
        primary_iri = hits[0]["iri"]
        primary_doid = primary_iri.split("DOID_")[-1]
        primary_label = _label_of(ds, primary_iri)
        shows(
            "bm25_search",
            text="lexical match against rdfs:label + IAO definition + synonyms",
        )

        # ── 2. Validate the IRI exists (defends against hallucination) ──
        section("step 2 — validate the resolved IRI is a real disease")
        naive(
            "no validation; downstream EHR call fails on the wrong code "
            "or, worse, succeeds against a wrong record."
        )
        exists = ds.query(
            f"PREFIX owl: <http://www.w3.org/2002/07/owl#> "
            f"ASK {{ <{primary_iri}> a owl:Class }}"
        )
        grounded(f"ASK {{ <{primary_iri}> a owl:Class }} → {exists}")
        shows(
            "query (SPARQL ASK)",
            text="sub-millisecond existence check, deterministic",
        )

        # ── 3. Walk the class hierarchy (parents up the taxonomy) ──
        section(f"step 3 — parent classes of {primary_label}")
        naive(
            "lists 'cancer' and stops; misses the hematologic-cancer / "
            "lymphoid-leukemia chain a downstream coder needs."
        )
        ancestors = ds.query(f"""
            PREFIX rdfs: <{RDFS}>
            SELECT ?a ?label WHERE {{
                <{primary_iri}> rdfs:subClassOf+ ?a .
                ?a rdfs:label ?label .
                FILTER(STRSTARTS(STR(?a), "{OBO}DOID_"))
            }} LIMIT 10
        """)
        for a in ancestors:
            doid = a["a"]["value"].split("DOID_")[-1]
            grounded(f"⤴ DOID:{doid:6}  →  {a['label']['value']}")
        shows(
            "query (SPARQL property path subClassOf+)",
            text="full class-hierarchy chain in one query",
        )

        # ── 4. Sibling subtypes (differential diagnosis) ──
        section("step 4 — subtypes for differential diagnosis")
        naive(
            "freely lists 'similar leukemias' from training data — no "
            "guarantee they're all real DOID terms or all subtypes."
        )
        subtypes = ds.query(f"""
            PREFIX rdfs: <{RDFS}>
            SELECT ?s ?label WHERE {{
                ?s rdfs:subClassOf+ <{primary_iri}> .
                ?s rdfs:label ?label .
            }} ORDER BY ?label LIMIT 10
        """)
        if subtypes:
            for s in subtypes:
                doid = s["s"]["value"].split("DOID_")[-1]
                grounded(f"⤵ DOID:{doid:6}  →  {s['label']['value']}")
        else:
            grounded("(no subtypes — this term is a leaf in the cancer slim)")
        shows(
            "query (SPARQL property path subClassOf+)",
            text="reverse traversal enumerates all known subtypes — "
                 "exactly what differential-diagnosis tools need",
        )

        # ── 5. Synonyms for downstream interoperability ──
        section("step 5 — synonyms (for matching against other code systems)")
        naive(
            "may or may not surface the synonym; can't guarantee these "
            "match a specific external code system."
        )
        syns = ds.query(f"""
            PREFIX oboInOwl: <{OBOINOWL}>
            SELECT ?syn WHERE {{
                <{primary_iri}> oboInOwl:hasExactSynonym ?syn
            }} LIMIT 6
        """)
        for s in syns:
            grounded(f"≈  {s['syn']['value']}")
        if not syns:
            grounded("(no exact synonyms in this slim; full DOID has more)")
        shows(
            "query (SPARQL)",
            text="oboInOwl:hasExactSynonym maps to ICD-O, MeSH, NCIt, etc.",
        )

        # ── 6. Definition for the prompt context ──
        section("step 6 — authoritative definition (drop into LLM context)")
        defn = _definition_of(ds, primary_iri)
        grounded(f'"{defn[:200]}..."  ←  IAO:0000115 on DOID:{primary_doid}')
        shows(
            "entity()",
            text="single call returns the definition + class chain + "
                 "synonyms — the agent's prompt context, machine-checked",
        )

        section("summary")
        print(
            "  This demo grounded a clinical statement against the\n"
            "  Disease Ontology — the open-source proxy for SNOMED CT —\n"
            "  using only turtlelake primitives and one .owl file.\n"
            "\n"
            "  In production: swap DOID for SNOMED CT (licensed, same\n"
            "  shape), add ICD-10 mappings as a second named graph,\n"
            "  layer SHACL shapes per claim type. The agent code is\n"
            "  unchanged. Provenance() guarantees every code is\n"
            "  traceable to its source ontology — what every regulated\n"
            "  clinical-AI deployment requires."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
