"""Regression tests for issues the user-perspective sub-agent found.
Each test pins one fix so the UX footgun doesn't come back.
"""

import pytest

from turtlelake import Dataset


# ── UX-1: rollback() must mutate self ─────────────────────────
#
# Before: `kg.rollback("pre")` returned a new handle but left the
# original `kg` pointing at the pre-rollback state; every query on
# the old handle saw stale (bad) data. The idiomatic "if bad:
# kg.rollback(...)" pattern silently returned wrong answers.


def test_rollback_mutates_original_handle(tmp_path):
    kg = Dataset.open(tmp_path / "kg")
    kg.insert_turtle('<https://ex.org/a> <https://ex.org/p> "baseline" .')
    kg.checkpoint("pre")
    kg.insert_turtle('<https://ex.org/a> <https://ex.org/p> "bad" .')
    assert kg.count() == 2

    # Call rollback WITHOUT capturing the return value — the user agent
    # did this and hit the bug.
    kg.rollback("pre")
    assert kg.count() == 1, (
        "rollback() must mutate self; old handle still saw stale data"
    )
    objs = {r["o"]["value"] for r in kg.query("SELECT ?o WHERE { ?s ?p ?o }")}
    assert "bad" not in objs
    assert "baseline" in objs


def test_rollback_still_returns_self_for_chaining(tmp_path):
    kg = Dataset.open(tmp_path / "kg")
    kg.insert_turtle('<https://ex.org/a> <https://ex.org/p> "v" .')
    kg.checkpoint("t")
    # Backward compatibility with the documented `kg = kg.rollback(...)`
    # idiom.
    returned = kg.rollback("t")
    assert returned is kg


# ── UX-2: validate() wraps FileNotFoundError ─────────────────
#
# Before: rdflib raised its own raw FileNotFoundError leaking internal
# paths. Now a turtlelake-level error names the `shapes_ttl` param.


def test_validate_missing_shapes_file_raises_clean_error(tmp_path):
    pytest.importorskip("pyshacl")
    pytest.importorskip("rdflib")
    kg = Dataset.open(tmp_path / "kg")
    kg.insert_turtle('<https://ex.org/a> <https://ex.org/p> "v" .')
    missing = tmp_path / "nope.shapes.ttl"
    with pytest.raises(FileNotFoundError) as exc_info:
        kg.validate(missing)
    msg = str(exc_info.value)
    assert "shapes_ttl=" in msg, "error must name the user-facing parameter"
    assert str(missing) in msg


# ── UX-3: diff(to_version=None) means current ─────────────────


def test_diff_to_version_default_is_current(tmp_path):
    kg = Dataset.open(tmp_path / "kg")
    kg.insert_turtle('<https://ex.org/a> <https://ex.org/p> "v1" .')
    kg.insert_turtle('<https://ex.org/a> <https://ex.org/p> "v2" .')
    # Explicit call.
    explicit = kg.diff(1, kg.current_version())
    # Default (to_version omitted) must equal the explicit form.
    implicit = kg.diff(1)
    assert explicit == implicit


# ── Bonus: bare `dataset._append_quads` is an internal ───────
# Guard that the public API as documented in the README works.


def test_public_api_methods_match_readme(tmp_path):
    """If someone adds a new public method to Dataset, they should
    document it in the README Python-API block. This lint-style test
    flags drift by checking a handful of methods exist and are callable."""
    kg = Dataset.open(tmp_path / "kg")
    for name in (
        "open", "count", "scan", "ingest_ttl", "insert_turtle", "insert",
        "query", "entity", "tag", "tags", "versions",
        "checkpoint", "rollback", "refresh", "diff", "provenance", "validate",
    ):
        assert hasattr(Dataset, name) or hasattr(kg, name), (
            f"documented API method {name!r} missing"
        )
    # Properties
    assert isinstance(type(kg).version, property)
