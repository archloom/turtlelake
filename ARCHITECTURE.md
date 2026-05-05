# Architecture

## Primary user: a local agent

Every design call is resolved in favor of **an agent on the same machine opening a single directory off disk**. That's the whole operating model. Secondary users: a human at a notebook, a CI job. Server deployments are out of scope -- if you need a triple-store server, use Oxigraph or GraphDB; `turtlelake` is a file format + library, not a service.

Consequences of this choice:

- **MCP is the primary interface**, not an afterthought. The Python API exists so the MCP server can be thin.
- **Writes are versioned by default.** Agents experiment and need rollback; Lance makes every write a new version cheaply.
- **No background processes, no daemon, no network.** Opening a dataset is one syscall away.
- **Correct-and-complete beats fast.** We reuse pyoxigraph for SPARQL today because it's correct against the W3C test suite. Speed comes later without an API change.

## Scope

One embedded Python package. One on-disk directory (a Lance dataset). TTL in, Arrow out, SPARQL answered by whichever open-source engine best fits the workload. **We write no parser, no optimizer, no executor.**

Non-goals: building a SPARQL engine, outperforming Oxigraph on OLTP, distributed execution, hybrid graph+vector queries in the same repo (sibling project).

## The artifact: two sibling Lance datasets in one directory

```
<store>/
├── triples.lance/        # the RDF graph (one table, schema below)
├── embeddings.lance/     # per-IRI vectors (created lazily on first .embed())
├── manifest.json         # turtlelake version, engine fingerprints,
│                         # default prefixes, embedding_dim
└── prefixes.ttl          # optional, user-edited, used for query convenience
```

Each Lance dataset is independent on disk (its own version chain, its own
tags) but presented as one artifact: `Dataset.open(path)` opens both, and
`checkpoint(name)` / `rollback(name)` apply to the pair. The vector layer
is a sibling rather than a column on triples for three reasons:

1. **Independent version cadence.** Re-embedding the corpus with a new
   model should not rewrite the graph or vice versa.
2. **Optional dependency.** A user who wants only the RDF lakehouse never
   pays the disk or runtime cost of vectors. The embeddings dataset is
   created on the first call to `.embed(...)`.
3. **Different access patterns.** Triples are scanned by SPARQL
   pattern-matching; vectors by ANN. Lance encodes the two
   workloads' columns differently when they are separate datasets.

The two are joined at query time by IRI -- the agent retrieves IRIs via
ANN and immediately walks them in the graph via `entity()` /
`graph_rag()`. No physical foreign key, no integrity constraint: vectors
for IRIs that no longer exist in the graph are tolerated and surface as
hits whose `entity()` expansion returns an empty subgraph.

Schema (one row per RDF quad):

```
subject         : string
predicate       : string
object          : string
object_kind     : string              # "iri" | "bnode" | "literal"
object_datatype : string (nullable)   # literals only
object_lang     : string (nullable)   # literals only
graph           : string (nullable)   # named graph; null = default graph
```

Strings are intentionally not dictionary-encoded at the API layer -- Lance's own encodings handle compression, and a flat string schema stays readable to any Arrow consumer (Polars, DuckDB, DataFusion) without extra metadata or N3 string parsing (the trade-off pycottas makes).

Embeddings schema (one row per (IRI, model) pair):

```
iri        : string
vector     : fixed_size_list<float32>[dim]    # dim fixed at first .embed()
model_id   : string                           # e.g. "openai:text-embedding-3-small"
created_at : timestamp[us, UTC]
```

The dim is recorded in `manifest.json` so re-opening a directory does
not need to scan Lance to learn it. Multiple `model_id` values can
coexist; `vector_search` and `graph_rag` accept a `model_id=` filter.

## SPARQL dispatch

Users call `dataset.query(sparql_text)`. Internally, `turtlelake` picks an engine by rule:

1. (M5+) If the query reduces to a BGP + FILTER + PROJECT that maps cleanly to a DataFusion SQL plan → execute against Lance directly. Columnar speed, zero materialization.
2. Otherwise → materialize the Lance rows into an in-memory `pyoxigraph.Store` and evaluate there.

Both paths return the same result shape. Callers never pick the engine.

v1 upgrade path: replace #1 and #2 with `rdf-fusion` (Schwarzinger 2025) once it supports Lance as a table provider -- writing that provider is ~100 lines since `lance-datafusion` already bridges Lance and DataFusion.

## Why this staging is honest

v0 SPARQL materializes into in-memory pyoxigraph -- correct, full SPARQL 1.1, RAM-bound. The trade-off is explicit: we ship full correctness day one at the cost of speed for now. M4 adds the DataFusion SQL path for analytic queries (the columnar win that *already matters* for agents). M5 adds the SPARQL route planner. Callers change nothing as each lights up.

## Versioning -- the agent rollback primitive

Lance gives us:
- Per-write version numbers (automatic).
- Named tags (`dataset.tag("pre-reasoning")`).
- Time-travel reads (`Dataset.open(path, tag="pre-reasoning")`).
- Cheap column addition.

`turtlelake` exposes these as three agent primitives:

```python
kg.checkpoint("pre-reasoning")   # create a tag on the current version
kg.insert(...)                   # make risky writes
kg = Dataset.open(path, tag="pre-reasoning")   # rollback = re-open
```

Versioning is something we *expose*, not something we *invent*.

## Entity expansion -- the agent read primitive

`dataset.entity(iri, hops=1)` returns the subject's 1-hop subgraph as structured JSON -- outbound predicates with objects, plus inbound edges. Hop count configurable. This is what 90% of agent queries actually ask for ("what do you know about X?") and deserves a first-class primitive rather than requiring the agent to compose a SPARQL `DESCRIBE` query every turn.

`entity(iri, similar=N)` additionally appends the top-N IRIs by vector
distance to this entity's stored embedding, when one exists. This is the
"more like this" retrieval shape -- useful when the agent has a known
anchor and wants related entities.

## GraphRAG -- vector retrieval + structural expansion

`dataset.graph_rag(query_vector, k=5, hops=1)` is the canonical agent
retrieval shape:

1. ANN search over `embeddings.lance` for the `k` IRIs nearest to
   `query_vector`.
2. For each hit, run `entity()` to expand the surrounding subgraph.
3. Return both the ranked hit list and the structured neighborhoods.

Same effect as composing `vector_search` + `entity` manually; the
single call exists because every GraphRAG flow does exactly this and we
want the MCP surface to make it one tool, not two.

The vector index itself is whatever Lance offers. The MVP uses the flat
brute-force search Lance provides out of the box (correct on any size,
fast under ~1M vectors). For production-scale corpora the caller can
build an `IVF_PQ` index via `pylance` directly against
`embeddings.lance/` -- turtlelake does not wrap that API yet.

## Concurrency contract

What happens when two processes open the same directory and both write?

- **Concurrent appends are safe.** Lance uses optimistic concurrency on
  the manifest: each writer commits a new version, and a contending
  writer either wins on the first try or retries. From the user's
  perspective, all writes succeed and the version chain is linear. We
  test this with up to N=3 spawn-mode workers (`tests/test_concurrent_writes.py`).
- **Reads during writes are snapshot-consistent.** A reader sees either
  the pre- or post-write state, never a torn manifest or partial
  fragment. Lance's manifest replace is atomic on POSIX file systems.
- **`compact()` is concurrent-safe with readers.** The compaction
  produces a new version; old fragments stay live until
  `prune_versions()` cleans them up, so any in-flight reader continues
  reading its own version without error.
- **Python's `fork` start method is unsafe.** Lance's library prints
  this warning explicitly. Multi-process workers must use
  `multiprocessing.get_context("spawn")` (or `forkserver`). This is the
  same constraint as PyTorch / numpy with OpenMP -- common in the
  ecosystem, but worth calling out.
- **Threads in one process share one Lance handle.** Lance is
  thread-safe for reads; for writes, prefer one writer thread per
  dataset to avoid retries you'd otherwise pay for at the manifest
  level. The MCP server runs single-threaded by default.

What we do NOT promise:

- A locking discipline across hosts on shared storage (NFS, object
  store). Lance's concurrency control is per-manifest, which works on
  POSIX-coherent storage. On NFS without close-to-open consistency or
  on object stores with eventual list-consistency, you may see retries
  but no data loss; turtlelake exposes Lance's behavior as-is.
- Cross-process coordination of `checkpoint(name)` / `rollback(name)`.
  Two processes calling `checkpoint("foo")` simultaneously will both
  succeed, but only one tag wins (last-writer wins). For coordinated
  multi-agent workflows, route checkpoints through one writer.

## MCP server

FastMCP, stdio by default (for local agents), HTTP available (for shared dev). Tools:

| Tool | Semantics |
|---|---|
| `sparql(query)` | SPARQL 1.1 against current version |
| `entity(iri, hops=1)` | structured 1–N-hop subgraph as JSON |
| `ingest(path)` | parse + append + new Lance version |
| `checkpoint(name)` | Lance tag |
| `rollback(name)` | re-open at tag (returns to caller) |
| `versions()` | list tags + versions |
| `scan(limit)` | first N quads, debugging only |

Security posture matches the rest of this repo's MCP servers: rate limit, input scan, audit log. Ingest and rollback are write operations and must be behind access keys in shared deployments.

## Reasoning story (why we don't ship a reasoner)

turtlelake does **not** contain a reasoner and will not ship one. Reasoning is an external materialization step; the inferred closure enters turtlelake as a new Lance version with full provenance. Three reasons:

1. **Correctness-critical, mature OSS exists.** Open Ontologies (OWL2-DL tableaux), `owlrl` (OWL-RL Python), ELK (EL profile), HermiT / Pellet (OWL-DL legacy) -- all maintained, all battle-tested. Shipping our own would duplicate years of work and lose on day one.
2. **SPARQL 1.1 property paths cover the 80%.** Transitive and inverse closures over `rdfs:subClassOf`, `skos:broader`, or any domain predicate are native at query time via pyoxigraph: `?x rdfs:subClassOf+ ?y`. No closure step needed for the common "what are all the subtypes of X" query.
3. **LanceDB doesn't train models.** It stores vectors. Training happens elsewhere, vectors get ingested. We mirror that split: inference happens elsewhere, closures get ingested.

The canonical pattern:

```
ontology.ttl ──reasoner──► closure.ttl ──ingest_ttl──► new Lance version
                                                     (provenance: source=reasoner-id)
```

`Dataset.provenance()` then lets you trace every triple back to the reasoner run that produced it, including the reasoner's ID and timestamp.

## Python API

```python
from turtlelake import Dataset

ds = Dataset.open("./my_kg")
ds.ingest_ttl("ontology.ttl")
ds.tag("baseline")

ds.query("SELECT ?s WHERE { ?s a <Device> }")   # SPARQL
ds.entity("https://example.org/foundation", hops=2)         # subgraph
ds.scan()                                         # Arrow table → Polars/DuckDB

# Versioning
ds.checkpoint("pre-write")
ds.insert_ttl_string("<a> <b> <c> .")
if bad: ds = Dataset.open("./my_kg", tag="pre-write")
```

## Key design decisions

| Decision | Choice | Why |
|---|---|---|
| Primary user | local agent | Defines every other choice |
| Language | Python | Rust engines already have Python bindings; glue stays Python |
| Storage format | Lance | Only format with columnar + versioning + Arrow + embedded at once |
| Schema | wide triple/quad table | Every Arrow engine reads it; matches COTTAS/RDF Fusion conventions |
| SPARQL v0 | pyoxigraph over materialized subset | Full SPARQL 1.1 correct day one |
| SPARQL v1 | rdf-fusion on DataFusion | Native columnar execution when ready |
| TTL parsing | pyoxigraph (wraps `oxttl`) | Already maintained, already Pythonic |
| Analytic queries | Polars / DuckDB / DataFusion direct | Lance reads natively from all three |
| Versioning | Lance tags | Agent rollback primitive for free |
| MCP framework | FastMCP | Official SDK path |

## Open questions

1. **Dictionary encoding of IRIs** -- materialize at ingest (smaller, faster joins, opaque to external readers) vs leave to Lance's column-level compression (simpler, portable). MVP: leave to Lance. Revisit at 10M+ triples.
2. **Named graph story** -- `graph` column (current) vs one Lance dataset per named graph (per-graph versioning). MVP: column.
3. **SPARQL UPDATE** -- DELETE/INSERT WHERE needs the WHERE to execute against current state. v0 supports `INSERT DATA` / `DELETE DATA` via pyoxigraph round-trip. Full UPDATE is M6+.
4. **Write-safety for agents** -- every write auto-tagged with `"pre-<timestamp>"` vs opt-in `checkpoint()`? Opt-in is simpler and matches git's model; leaving auto-tagging for a `strict=True` flag later.

## MVP milestones

1. **M1 -- Schema + ingest** ✓
2. **M2 -- SPARQL via pyoxigraph round-trip** ✓
3. **M3 -- Time travel surface (tag, versions, open(tag=...))** ✓
4. **M4 -- Agent read primitive (`entity(iri, hops=N)`)** ✓
5. **M5 -- MCP server (stdio) with sparql, entity, ingest, checkpoint, rollback** ✓
6. **M6 -- DataFusion analytic path (`dataset.sql(q)`)** pending
7. **M7 -- SPARQL route planner (pyoxigraph fallback, DataFusion fast path)** pending
8. **(post-MVP) -- rdf-fusion as executor** once Lance table provider lands

Realistic effort: M1–M5 shipped in this branch. M6 ~1 day. M7 ~3 days.

## Prior art

- **[Lance](https://github.com/lance-format/lance)** -- the file format this whole project piggybacks on.
- **[pyoxigraph / Oxigraph](https://github.com/oxigraph/oxigraph)** -- borrowed as the v0 SPARQL engine and as the Turtle/N-Quads parser.
- **[spargebra](https://crates.io/crates/spargebra)** + **[sparopt](https://crates.io/crates/sparopt)** -- standalone SPARQL parser and optimizer crates; needed if we write the route planner in Rust.
- **[RDF Fusion](https://github.com/tobixdev/rdf-fusion)** (Schwarzinger 2025) -- SPARQL on Apache DataFusion; our v1 target executor.
- **[COTTAS / pycottas](https://sferrada.com/publication/2025-iswc-arenas-guerrero-cottas/)** (ISWC 2025) -- RDF-in-Parquet. Validates the columnar-triples shape; doesn't do versioning or expose an Arrow-native API at the surface.
- **[Grafeo](https://github.com/GrafeoDB/grafeo)** -- embedded Rust graph DB with SPARQL + Cypher + vectors. Closest speed competitor; proprietary format so the LanceDB-philosophy bet (open file readable by any Arrow engine) is not available.
- **[OSTRICH](https://rdfostrich.github.io/article-demo/)** -- versioned RDF triple store on HDT. Has versioning; doesn't have columnar, Arrow, or a modern Python surface.
- **[LanceDB](https://github.com/lancedb/lancedb)** -- the product whose local-agent storage bet this project copies for RDF.
- **[Open Ontologies (fabio-rovai)](https://github.com/fabio-rovai/open-ontologies)** -- Rust MCP server + desktop Studio for AI-native ontology *engineering*. Same ecosystem (Oxigraph + SPARQL + SHACL + MCP), adjacent but distinct slot:

  | Axis | Open Ontologies | turtlelake |
  |---|---|---|
  | Center of gravity | **TBox** -- designing and reasoning over the ontology | **ABox** -- storing and serving the materialized graph |
  | Storage artifact | Oxigraph in-memory + SQLite sidecar for state | One Lance directory -- the whole KG, versions included |
  | OWL2-DL reasoner | **native (SHOIQ tableaux)** | none -- run an external reasoner, ingest its output |
  | File-format bet | `.ttl` + SQLite (two artifacts) | Lance (one Arrow-native artifact any tool reads) |
  | Tool surface | 43 tools (alignment, clinical, digital-twin, ONNX embeddings) | ~10 tools (store/query/version/validate/diff/provenance) |
  | Scale posture | RAM-bound (ontology-sized ≲1M triples) | on-disk Lance, analytic-friendly |

  The intended relationship is a **pipeline**: Open Ontologies is the workbench (design, reason, align, validate); turtlelake is the runtime (store, version, serve to agents, expose to Polars/DuckDB). Reasoned-closure TTL flows from the former into the latter.
