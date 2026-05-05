"""UC-11 refresh() + UC-13 durable agent memory.

Scenarios implemented: 11.1 (same-process refresh), 13.1 (facts persist
across re-open), 13.2 (ad-hoc insert without checkpoint).
"""


from turtlelake import Dataset


USER = "https://example.org/user/david"
TTL_FACT = (
    f'<{USER}> <https://example.org/prefers> "SPARQL" .\n'
)


def _literals_for(ds: Dataset, subject: str) -> list[str]:
    rows = ds.query(
        f'SELECT ?o WHERE {{ <{subject}> ?p ?o }} ORDER BY ?o'
    )
    return [r["o"]["value"] for r in rows]


def test_11_1_refresh_sees_sibling_handle_writes(tmp_path):
    """Two Dataset handles at the same path. Handle A writes; handle B
    refreshes and observes the new version."""
    path = tmp_path / "kg"
    a = Dataset.open(path)
    b = Dataset.open(path)
    # Both exist; need to seed at least once for open() to have cached a
    # Lance dataset. Use a().ingest_ttl via insert_turtle to avoid files.
    a.insert_turtle(TTL_FACT, source="seed", author="a")
    assert a.count() == 1
    # b was opened before the write and didn't pick up the dataset.
    b.refresh()
    assert b.count() == 1


def test_13_1_facts_persist_across_open_close(tmp_path):
    """Write via insert_turtle, drop the handle, re-open, read."""
    path = tmp_path / "kg"
    writer = Dataset.open(path)
    writer.insert_turtle(TTL_FACT, source="session-1", author="david")
    del writer
    reader = Dataset.open(path)
    assert _literals_for(reader, USER) == ["SPARQL"]


def test_13_2_adhoc_insert_without_checkpoint(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    ds.insert_turtle(TTL_FACT)  # no source/author — provenance defaults fire
    assert ds.count() == 1
    log = ds.provenance()
    assert log[0]["kind"] == "insert_turtle"
    assert log[0]["source"] == "inline-turtle"  # default
