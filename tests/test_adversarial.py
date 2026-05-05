"""Regression tests for issues surfaced by the adversarial review.
Each test corresponds to a specific CVE-style finding that was real.
"""


import pytest
from pyoxigraph import Literal, NamedNode, Quad

from turtlelake import Dataset
from turtlelake.security import redact_error, scan_input


# ── H-5: rollback silently resurrects rolled-back history ────
#
# Before the fix: checkpoint → bad write → rollback → new write would
# leave the bad write visible (append goes to chain head, not pinned
# view). After the fix, restore() commits the rollback as the new HEAD.


def test_rollback_then_write_does_not_resurrect_bad_history(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    ds.insert_turtle('<https://ex.org/a> <https://ex.org/p> "baseline" .')
    ds.checkpoint("pre")
    ds.insert_turtle('<https://ex.org/a> <https://ex.org/p> "bad" .')
    ds = ds.rollback("pre")
    ds.insert_turtle('<https://ex.org/a> <https://ex.org/p> "recovery" .')
    objs = {r["o"]["value"] for r in ds.query(
        "SELECT ?o WHERE { ?s ?p ?o }"
    )}
    assert "bad" not in objs, "rolled-back write reappeared after new write"
    assert objs == {"baseline", "recovery"}


# ── Entity SPARQL injection via IRI ───────────────────────────
#
# Before: `_expand_entity` f-stringed the IRI into SPARQL, so an IRI
# containing ">" or other syntax could inject. After: uses
# `quads_for_pattern(NamedNode(iri), ...)` — typed terms, no string
# concatenation into the query.


def test_entity_does_not_SPARQL_inject_on_hostile_iri(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    s = NamedNode("https://example.org/s")
    p = NamedNode("https://example.org/p")
    ds._append_quads([Quad(s, p, Literal("v"))], batch_size=10)
    # A "hostile" IRI that would break the old f-string. We expect
    # `_expand_entity` to treat it as an invalid IRI and return an empty
    # neighborhood — NOT to raise or inject.
    hostile = '> ?z ?y . ?z ?p2 ?o2 . BIND("owned"'
    got = ds.entity(hostile, hops=1)
    assert got["outgoing"] == []
    assert got["incoming"] == []


# ── SPARQL destructive-update regex coverage ─────────────────
#
# Before: only DELETE WHERE / DROP GRAPH / CLEAR GRAPH were blocked.
# After: INSERT DATA, DELETE DATA, LOAD, ADD, MOVE, COPY, CREATE GRAPH,
# DROP SILENT, DROP DEFAULT / NAMED / ALL are all blocked.


@pytest.mark.parametrize(
    "hostile",
    [
        "INSERT DATA { <https://ex> <https://p> 'x' }",
        "DELETE DATA { <https://ex> <https://p> 'x' }",
        "LOAD <https://evil/ontology.ttl>",
        "LOAD SILENT INTO GRAPH <http://g> <http://src>",
        "DROP SILENT GRAPH <https://g>",
        "DROP DEFAULT",
        "DROP ALL",
        "CLEAR SILENT ALL",
        "CREATE GRAPH <https://new>",
        "ADD <https://s> TO <https://t>",
        "MOVE <https://s> TO <https://t>",
        "COPY <https://s> TO <https://t>",
    ],
)
def test_sparql_destructive_update_is_blocked(hostile):
    result = scan_input(hostile)
    assert not result.safe, f"should block: {hostile!r}"
    assert result.blocked_by == "sparql_update_mask"


# ── redact_error — broader secret patterns ───────────────────


@pytest.mark.parametrize(
    "secret",
    [
        "AKIAIOSFODNN7EXAMPLE",                                 # AWS access key (real shape: AKIA + 16)
        "ghp_" + "a" * 36,                                      # GitHub classic
        "github_pat_" + "a" * 23,                               # GitHub fine-grained
        "xoxb-1234567890-abcdefghij",                            # Slack
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NSJ9.ABCDEFGHIJ",  # JWT-shaped
        "https://alice:hunter2@example.org/api",                 # URL creds
    ],
)
def test_redact_error_strips_common_secret_shapes(secret):
    e = RuntimeError(f"call failed: {secret}")
    redacted = redact_error(e)
    # The verbatim secret must not appear in the redacted output.
    assert secret not in redacted, f"leaked {secret!r} in {redacted!r}"


# ── Dataset.open version+tag collision ───────────────────────


def test_open_with_both_version_and_tag_raises(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    ds._append_quads(
        [Quad(NamedNode("https://a"), NamedNode("https://p"), Literal("v"))],
        batch_size=10,
    )
    ds.tag("v1")
    with pytest.raises(ValueError, match="version.*tag|tag.*version"):
        Dataset.open(tmp_path / "kg", version=1, tag="v1")


# ── ASK / CONSTRUCT result shapes ─────────────────────────────


def test_ask_returns_bool(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    ds._append_quads(
        [Quad(NamedNode("https://a"), NamedNode("https://p"), Literal("v"))],
        batch_size=10,
    )
    yes = ds.query("ASK { ?s ?p ?o }")
    assert yes is True
    no = ds.query("ASK { <https://nope> ?p ?o }")
    assert no is False


def test_construct_returns_triple_dicts(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    ds._append_quads(
        [Quad(NamedNode("https://a"), NamedNode("https://p"), Literal("v"))],
        batch_size=10,
    )
    triples = ds.query(
        "CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }"
    )
    assert isinstance(triples, list)
    assert len(triples) == 1
    t = triples[0]
    assert set(t.keys()) == {"subject", "predicate", "object"}


# ── Public Dataset.version ────────────────────────────────────


def test_version_property_is_stable(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    assert ds.version == -1  # empty
    ds._append_quads(
        [Quad(NamedNode("https://a"), NamedNode("https://p"), Literal("v"))],
        batch_size=10,
    )
    assert ds.version == 1
    assert ds.version == ds.current_version()


# ── Provenance JSONL tolerates torn lines ────────────────────


def test_provenance_skips_torn_lines(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    ds._append_quads(
        [Quad(NamedNode("https://a"), NamedNode("https://p"), Literal("v"))],
        batch_size=10,
    )
    ds._log_provenance(source="ok", author="a", kind="insert", row_delta=1)
    # Inject a torn line (simulates concurrent-write corruption).
    with ds.provenance_path.open("a", encoding="utf-8") as fh:
        fh.write('{"this is torn\n')
    ds._log_provenance(source="ok2", author="a", kind="insert", row_delta=1)

    log = ds.provenance()
    # Two valid records; torn line skipped, no exception.
    kinds = [r["kind"] for r in log]
    assert kinds.count("insert") == 2
