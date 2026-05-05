"""Boundary checks for the embed() / vector_search() input contract.

These exist to make sure our hardening (NaN/Inf rejection, dim/row
caps, model_id whitelisting, SQL filter escaping) actually fires
before bad data hits Lance.
"""

import pytest

from turtlelake import Dataset


def test_embed_rejects_nan(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    with pytest.raises(ValueError, match="not finite"):
        ds.embed(["a"], [[float("nan"), 1.0]], model_id="m")


def test_embed_rejects_inf(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    with pytest.raises(ValueError, match="not finite"):
        ds.embed(["a"], [[1.0, float("inf")]], model_id="m")


def test_embed_rejects_invalid_model_id(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    for bad in (
        "",
        "model with spaces",
        "with;semicolon",
        "with'quote",
        "with\"doublequote",
        "with\\backslash",
        "with\nnewline",
    ):
        with pytest.raises(ValueError, match="model_id"):
            ds.embed(["a"], [[1.0]], model_id=bad)


def test_embed_accepts_realistic_model_ids(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    for good in (
        "openai:text-embedding-3-small",
        "sentence-transformers/all-MiniLM-L6-v2",
        "voyage:v3.large",
        "model_v2.1",
    ):
        ds.embed(["x"], [[1.0]], model_id=good)


def test_embed_rejects_dim_above_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("TURTLELAKE_MAX_EMBEDDING_DIM", "4")
    ds = Dataset.open(tmp_path / "kg")
    with pytest.raises(ValueError, match="exceeds cap"):
        ds.embed(["a"], [[1.0] * 5], model_id="m")


def test_embed_rejects_too_many_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("TURTLELAKE_MAX_VECTORS_PER_EMBED", "2")
    ds = Dataset.open(tmp_path / "kg")
    with pytest.raises(ValueError, match="too large"):
        ds.embed(["a", "b", "c"], [[1.0], [1.0], [1.0]], model_id="m")


def test_embed_dim_consistency_error_names_existing_dim(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    ds.embed(["a"], [[1.0, 0.0]], model_id="m")
    with pytest.raises(ValueError, match="existing dataset dim 2"):
        ds.embed(["b"], [[1.0, 0.0, 0.0]], model_id="m")


def test_vector_search_rejects_dim_mismatch(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    ds.embed(["a"], [[1.0, 0.0]], model_id="m")
    with pytest.raises(ValueError, match="dim"):
        ds.vector_search([1.0, 2.0, 3.0], k=1)


def test_vector_search_rejects_nan_query(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    ds.embed(["a"], [[1.0, 0.0]], model_id="m")
    with pytest.raises(ValueError, match="not finite"):
        ds.vector_search([float("nan"), 0.0], k=1)


def test_vector_search_rejects_invalid_model_id_filter(tmp_path):
    ds = Dataset.open(tmp_path / "kg")
    ds.embed(["a"], [[1.0]], model_id="m")
    with pytest.raises(ValueError, match="model_id"):
        ds.vector_search([1.0], k=1, model_id="evil; DROP TABLE")


def test_sql_escape_rejects_null_byte(tmp_path):
    """Internal: ensure the helper itself rejects null bytes — not
    just that callers don't pass them."""
    from turtlelake.dataset import _sql_escape

    with pytest.raises(ValueError, match="null byte"):
        _sql_escape("ab\x00cd")


def test_sql_escape_rejects_control_chars(tmp_path):
    from turtlelake.dataset import _sql_escape

    with pytest.raises(ValueError, match="control character"):
        _sql_escape("ab\x01cd")


def test_sql_escape_escapes_backslash_before_quote(tmp_path):
    """Backslash must be doubled BEFORE the quote-doubling pass, so an
    attacker-controlled \\' sequence cannot become an unescaped '."""
    from turtlelake.dataset import _sql_escape

    out = _sql_escape(r"a\'b")
    # Expect the backslash doubled and the quote doubled.
    assert out == r"a\\''b"


def test_similar_to_iri_is_deterministic_across_models(tmp_path):
    """Multiple embeddings for the same IRI must resolve to the most
    recent by created_at — no Lance-scan-order dependence."""
    import time

    from pyoxigraph import Literal, NamedNode, Quad

    ds = Dataset.open(tmp_path / "kg")
    # entity() needs a triples backing — seed one minimal quad.
    label = NamedNode("http://www.w3.org/2000/01/rdf-schema#label")
    a = NamedNode("https://ex/A")
    ds._append_quads([Quad(a, label, Literal("A"))], batch_size=10)

    # Two writes: m1 first, m2 second. Most recent (m2) should win.
    ds.embed(["https://ex/A"], [[1.0, 0.0]], model_id="m1")
    time.sleep(0.01)  # ensure created_at differs
    ds.embed(["https://ex/A"], [[0.0, 1.0]], model_id="m2")
    ds.embed(["https://ex/B"], [[0.0, 1.0]], model_id="m2")
    out = ds.entity("https://ex/A", similar=1)
    # The seed vector picked by _similar_to_iri should be the m2 one
    # (most recent), so the closest neighbor is B (which is m2's
    # [0,1]) — not whatever m1 happened to be near.
    assert out["similar"][0]["iri"] == "https://ex/B"
