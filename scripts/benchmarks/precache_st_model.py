"""One-time helper: download a sentence-transformers model into a
turtlelake-bench cache directory so subsequent benchmark runs work
fully offline.

Why this exists. The reference benchmarks (MuSiQue, SIFT-shape) report
honest numbers with our offline-friendly fallback (LSA / Gaussian
vectors). For a fair comparison to published GraphRAG / RAG
literature, you want a real transformer embedding. This script
downloads `sentence-transformers/all-MiniLM-L6-v2` (~90 MB) once,
points the HF transformers cache at a directory you control, and
verifies the download by encoding a test sentence.

Run once on a network-connected machine, then ship the cache
directory alongside your dataset for offline reproduction.

Usage:
    pip install sentence-transformers
    uv run python scripts/benchmarks/precache_st_model.py
    # → creates ~/.cache/turtlelake-bench/st-cache/

    # Then on the offline machine:
    HF_HOME=~/.cache/turtlelake-bench/st-cache \\
        uv run python scripts/benchmarks/musique.py --embedding-prefer st
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

DEFAULT_CACHE = Path.home() / ".cache" / "turtlelake-bench" / "st-cache"


def main() -> int:
    cache_dir = Path(os.environ.get("ST_CACHE_DIR", str(DEFAULT_CACHE)))
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        print(
            "sentence-transformers is not installed. Run:\n"
            "    pip install sentence-transformers",
            file=sys.stderr,
        )
        raise SystemExit(2) from e

    model_name = os.environ.get(
        "ST_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    print(f"downloading {model_name} into {cache_dir}", file=sys.stderr)
    model = SentenceTransformer(model_name)
    print(
        f"  embedding dim: {model.get_sentence_embedding_dimension()}",
        file=sys.stderr,
    )

    test = model.encode(["hello world"], normalize_embeddings=True)
    print(f"  test encode shape: {test.shape}", file=sys.stderr)
    print(
        f"\nCache populated at {cache_dir}.\n"
        "Set HF_HOME to that path on the offline machine to use it.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
