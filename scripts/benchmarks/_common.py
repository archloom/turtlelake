"""Shared helpers for the public-benchmark runners.

Two design choices worth flagging up front:

1. **Embeddings.** The standard reference embedding is
   `sentence-transformers/all-MiniLM-L6-v2`, but that requires a
   download from huggingface.co at first run. When that host is
   unreachable (e.g. air-gapped CI, sandbox), we fall back to a
   TF-IDF + Truncated-SVD (LSA) embedding via scikit-learn. The
   absolute numbers are lower with LSA, but it's a real
   text-similarity signal and the *shape* of the comparison
   (flat-search vs graph_rag) is preserved.

2. **No HF datasets dependency.** We download benchmark data via
   plain HTTPS from canonical mirrors (HippoRAG ships a sampled
   MuSiQue subset on github). Easier to reproduce, easier to audit.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path
from typing import Sequence


CACHE_DIR = Path(os.environ.get("TURTLELAKE_BENCHMARK_CACHE", str(Path.home() / ".cache/turtlelake-bench")))


def cache_dir() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def http_download(url: str, *, sha256: str | None = None) -> Path:
    """Fetch `url` into the benchmark cache, content-addressed by URL.
    Optional `sha256` verifies the download -- strongly recommended for
    benchmarks where data drift would invalidate published numbers."""
    fname = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    suffix = Path(url).suffix or ".bin"
    target = cache_dir() / f"{fname}{suffix}"
    if target.exists() and target.stat().st_size > 0:
        if sha256 is not None and _sha256_of(target) != sha256:
            raise RuntimeError(
                f"cached {target} sha256 mismatch -- delete and re-download"
            )
        return target
    print(f"downloading {url} -> {target}")
    req = urllib.request.Request(
        url, headers={"User-Agent": "turtlelake-benchmark/0.0.1"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp, open(target, "wb") as fh:
        while chunk := resp.read(64 * 1024):
            fh.write(chunk)
    if sha256 is not None and _sha256_of(target) != sha256:
        raise RuntimeError(
            f"downloaded {url} sha256 mismatch -- corrupt mirror?"
        )
    return target


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(64 * 1024):
            h.update(chunk)
    return h.hexdigest()


# ── embedding factory ───────────────────────────────────────────────


def build_embedder(
    texts: Sequence[str],
    *,
    dim: int = 128,
    prefer: str = "auto",
    st_model: str | None = None,
    st_query_prefix: str = "",
    st_passage_prefix: str = "",
):
    """Return `(embed_fn, model_id, dim)` for the given corpus.

    `prefer="st"`     → require a sentence-transformers model
                        (default: all-MiniLM-L6-v2; override via `st_model`)
    `prefer="lsa"`    → use TF-IDF + TruncatedSVD (no network)
    `prefer="auto"`   → try sentence-transformers, fall back to LSA on
                        any error (network blocked, model missing, etc.)

    `st_query_prefix` / `st_passage_prefix` are E5/BGE-style instructions
    prepended to inputs (e.g. "query: " / "passage: " for E5; BGE only
    needs a query prefix). The returned `_embed` distinguishes by string
    vs list-of-strings: single string → query, list → passages.

    The model fits on `texts` (LSA needs the corpus to compute the SVD
    basis; ST is corpus-independent). For the LSA path we project to
    `dim` dimensions -- 128 is a reasonable default that matches SIFT.
    """
    if prefer in ("auto", "st"):
        try:
            return _build_st(
                model_name=st_model,
                query_prefix=st_query_prefix,
                passage_prefix=st_passage_prefix,
            )
        except Exception as e:  # pragma: no cover -- exercised when HF is reachable
            if prefer == "st":
                raise
            print(f"sentence-transformers unavailable ({e}); falling back to LSA")
    return _build_lsa(texts, dim=dim)


def _build_st(
    *,
    model_name: str | None = None,
    query_prefix: str = "",
    passage_prefix: str = "",
):
    """Sentence-transformers model (default: all-MiniLM-L6-v2)."""
    from sentence_transformers import SentenceTransformer  # type: ignore

    name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(name)
    # SentenceTransformer renamed the accessor in v3; support both.
    get_dim = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    dim = get_dim()

    def _embed(text_or_list):
        single = isinstance(text_or_list, str)
        items = [text_or_list] if single else list(text_or_list)
        prefix = query_prefix if single else passage_prefix
        if prefix:
            items = [prefix + t for t in items]
        out = model.encode(items, normalize_embeddings=True, show_progress_bar=False)
        return out[0].tolist() if single else out.tolist()

    return _embed, name, dim


def _build_lsa(texts: Sequence[str], *, dim: int):
    """TF-IDF + Truncated SVD (LSA). Sparse → dense in `dim` dims.

    Using L2-normalized output so the dataset's L2 distances behave
    monotonically -- same metric the SentenceTransformer path uses.
    """
    from sklearn.decomposition import TruncatedSVD  # type: ignore
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.preprocessing import normalize  # type: ignore

    vectorizer = TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )
    sparse = vectorizer.fit_transform(texts)
    n_components = min(dim, sparse.shape[1] - 1, sparse.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=0)
    dense = svd.fit_transform(sparse)
    dense = normalize(dense, norm="l2", axis=1)

    # Keep the fitted state as closures so subsequent embeds run
    # the same transform (no re-fit on test queries).
    def _embed(text_or_list):
        single = isinstance(text_or_list, str)
        items = [text_or_list] if single else list(text_or_list)
        s = vectorizer.transform(items)
        d = svd.transform(s)
        d = normalize(d, norm="l2", axis=1)
        return d[0].tolist() if single else d.tolist()

    name = f"lsa:tfidf-svd-{n_components}"
    return _embed, name, n_components


# ── retrieval / scoring helpers ─────────────────────────────────────


def expand_graph_rag(out: dict) -> list[str]:
    """Flatten `graph_rag(...)` output into an ordered list of unique
    IRIs: seeds first (by distance), then BFS-discovered neighbors."""
    seen: list[str] = []
    seen_set: set[str] = set()
    for hit in out["hits"]:
        if hit["iri"] not in seen_set:
            seen.append(hit["iri"])
            seen_set.add(hit["iri"])
    for iri, entity in out["entities"].items():
        for edge in entity.get("outgoing", []):
            o = edge["object"]
            if isinstance(o, dict) and o.get("type") == "iri" and o["value"] not in seen_set:
                seen.append(o["value"])
                seen_set.add(o["value"])
        for edge in entity.get("incoming", []):
            s = edge["subject"]
            if s not in seen_set:
                seen.append(s)
                seen_set.add(s)
        for nb_iri, nb in entity.get("neighbors", {}).items():
            if nb_iri not in seen_set:
                seen.append(nb_iri)
                seen_set.add(nb_iri)
            for edge in nb.get("outgoing", []):
                o = edge["object"]
                if isinstance(o, dict) and o.get("type") == "iri" and o["value"] not in seen_set:
                    seen.append(o["value"])
                    seen_set.add(o["value"])
    return seen


def recall_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    top = set(retrieved[:k])
    return len(top & gold) / len(gold)
