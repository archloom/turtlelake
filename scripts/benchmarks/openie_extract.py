"""Optional preprocessor: extract entity-relation triples from a text
corpus using an LLM, write them as RDF for turtlelake to ingest.

Replaces the title-mention edge heuristic in `musique.py` with the
denser graph that HippoRAG and other production GraphRAG systems
build via OpenIE. With an LLM-extracted graph, our `graph_rag` and
`graph_rag_ppr` results should close the rest of the gap to
HippoRAG's published numbers.

Why this is its own script. Calling an LLM costs money and requires
network access — running it as part of the benchmark harness would
break the "reproducible end-to-end on a fresh clone" promise. We
ship the extraction step as a separable preprocessing tool: pay
once, persist the extracted triples, then run the benchmark offline
from the persisted RDF.

The LLM itself is whatever you point this at. Default scaffolding
uses the OpenAI Chat Completions API (cheapest reasonable choice
with an OPENAI_API_KEY env var); swap the `_extract` function for
any other backend without touching the surrounding pipeline.

Cost estimate: at ~50-100 tokens of input per paragraph + ~50 tokens
of structured output, MuSiQue's 11.6k paragraphs cost roughly $0.50
on gpt-4o-mini, $5 on gpt-4o. Reuse the cached output across
benchmark runs.

Usage:
    export OPENAI_API_KEY=sk-...
    uv run python scripts/benchmarks/openie_extract.py \\
        --input scripts/benchmarks/_data/musique_corpus.json \\
        --output scripts/benchmarks/_data/musique_openie.ttl

    # Then point musique.py at the richer graph:
    uv run python scripts/benchmarks/musique.py \\
        --extra-edges-ttl scripts/benchmarks/_data/musique_openie.ttl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable

NS = "https://benchmark.turtlelake/musique/"

PROMPT = """You are an information extraction system. Given a passage,
extract the most important (subject, predicate, object) triples that
describe relationships between named entities. Return at most 6 triples
as a JSON array of objects with keys "subject", "predicate", "object".
All values must be strings. The predicate should be a short
lowercase_with_underscores phrase (e.g. "born_in", "founded_by",
"capital_of").

Passage:
{text}

Return ONLY the JSON array, no commentary."""


def _extract_openai(text: str, *, model: str = "gpt-4o-mini") -> list[dict]:
    """OpenAI Chat Completions backend. Replace with another backend
    by writing a function with the same signature."""
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as e:  # pragma: no cover
        print(
            "openai package not installed. Run:\n"
            "    pip install openai",
            file=sys.stderr,
        )
        raise SystemExit(2) from e

    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT.format(text=text)}],
        response_format={"type": "json_object"},
        temperature=0.0,
        max_tokens=400,
    )
    content = resp.choices[0].message.content or "[]"
    try:
        parsed = json.loads(content)
        # The model sometimes returns {"triples": [...]}; tolerate both.
        if isinstance(parsed, dict) and "triples" in parsed:
            parsed = parsed["triples"]
        if not isinstance(parsed, list):
            return []
        return [
            t for t in parsed
            if isinstance(t, dict)
            and isinstance(t.get("subject"), str)
            and isinstance(t.get("predicate"), str)
            and isinstance(t.get("object"), str)
        ]
    except json.JSONDecodeError:
        return []


def _slugify(text: str) -> str:
    """Map an entity string to a stable IRI slug."""
    import re

    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", text.strip()).strip("_")[:80]
    return slug or "blank"


def _emit_ttl(extracted: Iterable[tuple[int, list[dict]]], out_path: Path) -> int:
    """Write extracted triples to a TTL file. Returns triple count."""
    n = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(f"@prefix b: <{NS}> .\n")
        fh.write("@prefix p: <" + NS + "predicate/> .\n")
        fh.write("@prefix e: <" + NS + "entity/> .\n\n")
        for para_idx, triples in extracted:
            for t in triples:
                s = _slugify(t["subject"])
                p = _slugify(t["predicate"])
                o = _slugify(t["object"])
                fh.write(
                    f"e:{s} p:{p} e:{o} . "
                    f"# from para {para_idx}\n"
                )
                n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="MuSiQue corpus JSON")
    ap.add_argument("--output", required=True, help="output TTL path")
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = process every paragraph")
    ap.add_argument("--model", default="gpt-4o-mini")
    args = ap.parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        print("OPENAI_API_KEY not set in env.", file=sys.stderr)
        return 2

    corpus = json.loads(Path(args.input).read_text())
    if args.limit > 0:
        corpus = corpus[: args.limit]

    print(f"extracting from {len(corpus)} paragraphs via {args.model}...",
          file=sys.stderr)
    extracted: list[tuple[int, list[dict]]] = []
    t0 = time.perf_counter()
    for i, item in enumerate(corpus):
        text = (item.get("title", "") + ". " + item.get("text", ""))[:2000]
        try:
            triples = _extract_openai(text, model=args.model)
        except Exception as e:  # pragma: no cover — surface and continue
            print(f"  paragraph {i} extraction failed: {e}", file=sys.stderr)
            triples = []
        extracted.append((i, triples))
        if (i + 1) % 50 == 0:
            elapsed = time.perf_counter() - t0
            print(
                f"  {i+1}/{len(corpus)} ({elapsed:.0f}s, "
                f"{(i+1)/elapsed:.1f} paragraphs/s)",
                file=sys.stderr,
            )

    n_triples = _emit_ttl(extracted, Path(args.output))
    print(f"\nwrote {n_triples} triples to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
