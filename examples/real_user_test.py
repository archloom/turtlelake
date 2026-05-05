"""
real_user_test.py — a mid-level engineer's first contact with turtlelake.

Goal: use turtlelake end-to-end after reading only the README. No cheating by
reading source. Exercise the happy path (ingest -> query -> entity -> insert ->
checkpoint -> rollback -> diff -> provenance) and several negative paths.

Run (POSIX):   .venv/bin/python examples/real_user_test.py
Run (Windows): .venv\Scripts\python.exe examples\real_user_test.py
"""

from __future__ import annotations

import shutil
import traceback
from pathlib import Path

from turtlelake import Dataset

HERE = Path(__file__).resolve().parent
WORK = HERE / "_real_user_test_work"
KG_DIR = WORK / "coffeeshop.turtlelake"
TTL_FILE = WORK / "coffeeshop.ttl"
BAD_TTL = WORK / "bad.ttl"
GOOD_TTL = WORK / "good_addition.ttl"


def banner(msg: str) -> None:
    print(f"\n=== {msg} ===")


def try_(label: str, fn):
    """Run a callable, report nicely, never abort the script."""
    print(f"\n--- {label} ---")
    try:
        result = fn()
        print(f"OK: {label}")
        if result is not None:
            # keep output terse
            s = repr(result)
            if len(s) > 400:
                s = s[:400] + "...<truncated>"
            print(f"  result: {s}")
        return result
    except Exception as e:
        print(f"RAISED {type(e).__name__}: {e}")
        # minimal trace — last frame only, so we can see origin file:line
        tb = traceback.extract_tb(e.__traceback__)
        if tb:
            last = tb[-1]
            print(f"  at {last.filename}:{last.lineno} in {last.name}")
        return e


def write_fixtures() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    # Fresh dataset dir each run
    if KG_DIR.exists():
        shutil.rmtree(KG_DIR)

    TTL_FILE.write_text(
        """
@prefix : <https://example.org/cafe#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

:Drink a rdfs:Class .
:Espresso a :Drink ; rdfs:label "Espresso" ; :priceUSD "3.00"^^xsd:decimal ; :sizeOz "1"^^xsd:integer .
:Latte a :Drink ; rdfs:label "Latte" ; :priceUSD "4.50"^^xsd:decimal ; :sizeOz "12"^^xsd:integer ; :pairsWith :Croissant .
:Cappuccino a :Drink ; rdfs:label "Cappuccino" ; :priceUSD "4.25"^^xsd:decimal ; :sizeOz "6"^^xsd:integer .
:ColdBrew a :Drink ; rdfs:label "Cold Brew" ; :priceUSD "4.00"^^xsd:decimal ; :sizeOz "16"^^xsd:integer .

:Pastry a rdfs:Class .
:Croissant a :Pastry ; rdfs:label "Butter Croissant" ; :priceUSD "3.75"^^xsd:decimal .
:Muffin a :Pastry ; rdfs:label "Blueberry Muffin" ; :priceUSD "3.25"^^xsd:decimal .
""".strip()
    )

    GOOD_TTL.write_text(
        """
@prefix : <https://example.org/cafe#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

:Mocha a :Drink ; rdfs:label "Mocha" ; :priceUSD "5.00"^^xsd:decimal ; :sizeOz "12"^^xsd:integer .
""".strip()
    )

    BAD_TTL.write_text(
        """
@prefix : <https://example.org/cafe#> .
:Unicorn a :MythicalDrink ; :priceUSD "infinity" .
""".strip()
    )


def main() -> None:
    write_fixtures()

    # -------------- happy path --------------
    banner("open empty dataset")
    kg = try_("Dataset.open(fresh dir)", lambda: Dataset.open(str(KG_DIR)))
    try_("count() on empty", lambda: kg.count())

    banner("ingest TTL")
    try_("ingest_ttl(coffeeshop.ttl)", lambda: kg.ingest_ttl(str(TTL_FILE)))
    try_("count() after ingest", lambda: kg.count())
    try_("tag 'baseline'", lambda: kg.tag("baseline"))

    banner("two SPARQL queries")
    q1 = """
    PREFIX : <https://example.org/cafe#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?label ?price WHERE {
      ?d a :Drink ; rdfs:label ?label ; :priceUSD ?price .
    } ORDER BY ?price
    """
    try_("SELECT drinks by price", lambda: kg.query(q1))

    q2 = """
    PREFIX : <https://example.org/cafe#>
    ASK { :Latte :pairsWith :Croissant }
    """
    try_("ASK latte pairs with croissant", lambda: kg.query(q2))

    banner("entity() browse")
    try_(
        "entity('https://example.org/cafe#Latte')",
        lambda: kg.entity("https://example.org/cafe#Latte"),
    )

    banner("insert a fact, checkpoint, then write a 'bad' fact, rollback")
    try_("checkpoint 'pre-mocha'", lambda: kg.checkpoint("pre-mocha"))
    try_("insert_turtle good_addition.ttl", lambda: kg.insert_turtle(GOOD_TTL.read_text()))
    try_("count after insert", lambda: kg.count())
    try_("checkpoint 'post-mocha'", lambda: kg.checkpoint("post-mocha"))

    # Now write a "bad" fact (syntactically valid TTL, semantically weird)
    try_(
        "insert_turtle bad.ttl (we'll roll this back)",
        lambda: kg.insert_turtle(BAD_TTL.read_text()),
    )
    try_("count after bad insert", lambda: kg.count())

    # Roll back to post-mocha (the last known good state).
    # Important: we reassign because README's agent_workflow pattern is
    # `kg = Dataset.open(..., tag=...)`. rollback() returns something —
    # let's find out what.
    rolled = try_("rollback('post-mocha')", lambda: kg.rollback("post-mocha"))
    if isinstance(rolled, Dataset):
        kg = rolled  # use whatever rollback handed back
    try_("count after rollback (same handle)", lambda: kg.count())
    # Belt-and-suspenders: re-open at the tag explicitly
    kg_at_tag = try_(
        "Dataset.open(tag='post-mocha')",
        lambda: Dataset.open(str(KG_DIR), tag="post-mocha"),
    )
    if isinstance(kg_at_tag, Dataset):
        try_("count() of tag-pinned handle", lambda: kg_at_tag.count())

    banner("verify rolled-back fact is gone")
    q_unicorn = """
    PREFIX : <https://example.org/cafe#>
    ASK { :Unicorn ?p ?o }
    """
    try_("ASK unicorn exists?", lambda: kg.query(q_unicorn))

    banner("diff() baseline vs current")
    # README's provenance/diff isn't documented — signature demands two args.
    try_("diff('baseline', 'post-mocha')", lambda: kg.diff("baseline", "post-mocha"))

    banner("provenance()")
    try_("provenance()", lambda: kg.provenance())

    banner("versions() / tags()")
    try_("versions()", lambda: kg.versions())
    try_("tags()", lambda: kg.tags())

    # -------------- negative paths --------------
    banner("NEGATIVE: entity() for an IRI that doesn't exist")
    try_(
        "entity('https://example.org/cafe#Nonexistent')",
        lambda: kg.entity("https://example.org/cafe#Nonexistent"),
    )

    banner("NEGATIVE: rollback with a nonexistent tag")
    try_("rollback('never-made-this-tag')", lambda: kg.rollback("never-made-this-tag"))

    banner("NEGATIVE: validate against a SHACL file that doesn't exist")
    try_(
        "validate('/tmp/does_not_exist.shacl.ttl')",
        lambda: kg.validate("/tmp/does_not_exist_shapes.ttl"),
    )

    banner("NEGATIVE: malformed SPARQL")
    try_(
        "query('SELECT ?x WHERE { ?x a')  # truncated",
        lambda: kg.query("SELECT ?x WHERE { ?x a"),
    )

    banner("NEGATIVE: open at a tag, then refresh()")
    kg_tagged = try_(
        "Dataset.open(KG_DIR, tag='baseline')",
        lambda: Dataset.open(str(KG_DIR), tag="baseline"),
    )
    if isinstance(kg_tagged, Dataset):
        try_("refresh() on a tag-pinned dataset", lambda: kg_tagged.refresh())

    banner("DONE")


if __name__ == "__main__":
    main()
