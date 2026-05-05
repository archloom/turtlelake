"""End-to-end walkthrough of the turtlelake MVP surface.

  1. Open a dataset (creates `.turtlelake-demo/triples.lance`).
  2. Insert a small inline ontology (one Lance version).
  3. Scan the Arrow table (zero-copy to Pandas / Polars / DuckDB possible here).
  4. Run a SPARQL query (dispatched to pyoxigraph internally).
  5. Tag the current version for reproducibility.

The TTL is inlined so this script runs on a fresh clone with no
external download. For domain-specific demos with real public
ontologies, see `demo_legal_lkif.py`, `demo_medical_doid.py`,
`demo_science_go.py`, and `demo_gov_dcat.py`.
"""

from pathlib import Path

from turtlelake import Dataset

HERE = Path(__file__).parent

# Tiny domain-neutral ontology: a few books with their genre and author.
DEMO_TTL = """\
@prefix ex:   <https://example.org/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

ex:Book           a rdfs:Class ; rdfs:label "Book" .
ex:Author         a rdfs:Class ; rdfs:label "Author" .
ex:Genre          a rdfs:Class ; rdfs:label "Genre" .

ex:scifi          a ex:Genre ; rdfs:label "Science Fiction" .
ex:mystery        a ex:Genre ; rdfs:label "Mystery" .

ex:asimov         a ex:Author ; rdfs:label "Isaac Asimov" .
ex:christie       a ex:Author ; rdfs:label "Agatha Christie" .

ex:foundation     a ex:Book ;
                  rdfs:label "Foundation" ;
                  ex:author ex:asimov ;
                  ex:genre ex:scifi .

ex:robotsOfDawn   a ex:Book ;
                  rdfs:label "The Robots of Dawn" ;
                  ex:author ex:asimov ;
                  ex:genre ex:scifi .

ex:roger          a ex:Book ;
                  rdfs:label "The Murder of Roger Ackroyd" ;
                  ex:author ex:christie ;
                  ex:genre ex:mystery .
"""


def main() -> None:
    ds = Dataset.open(HERE.parent / ".turtlelake-demo")
    n = ds.insert_turtle(DEMO_TTL, source="quickstart")
    print(f"inserted {n} quads; total rows: {ds.count()}")

    print("\n-- raw Arrow scan (first 3 rows) --")
    for row in ds.scan().slice(0, 3).to_pylist():
        print(f"  {row}")

    print("\n-- SPARQL: books and their author --")
    rows = ds.query("""
        PREFIX ex:   <https://example.org/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?bookLabel ?authorLabel WHERE {
            ?book a ex:Book ;
                  rdfs:label ?bookLabel ;
                  ex:author ?author .
            ?author rdfs:label ?authorLabel .
        }
    """)
    for r in rows:
        print(f"  {r['bookLabel']['value']}  by  {r['authorLabel']['value']}")

    ds.tag("quickstart-v1")
    print(f"\ntagged 'quickstart-v1' -> versions: {[v['version'] for v in ds.versions()]}")


if __name__ == "__main__":
    main()
