# Requirements & Use Cases

This document exists to keep scope honest. Every requirement traces back to the stated purpose; anything that doesn't trace is declared out of scope so we don't silently accumulate work.

---

## 1. Purpose

> **An embedded knowledge graph format for local agents -- LanceDB's philosophy applied to RDF.**
>
> One on-disk directory is the whole "database". An agent opens it with no server, queries it with SPARQL or DataFrame tools, writes to it under explicit checkpoints, and hands it to another agent by copying the directory.

Three principles derive from the purpose. All other decisions are resolved by consulting them:

- **P1 -- Local and embedded.** No server, no network, no daemon. The unit of deployment is a directory.
- **P2 -- Open file format.** The data is a Lance dataset with a public Arrow schema. Any Arrow-compatible engine (Polars, DuckDB, DataFusion) reads it without turtlelake.
- **P3 -- Reuse, don't reinvent.** SPARQL engines, TTL parsers, columnar executors already exist and are mature. turtlelake ships the glue, schema, and agent surface -- nothing else.

## 2. Actors

| Actor | How they interact | Primary? |
|---|---|---|
| **Local agent** (Claude Code, Cursor, local runtime) | MCP over stdio | **yes** |
| **CI / evaluation harness** | Python API; opens the dataset pinned at a tag | yes |
| **Data/ontology engineer** | Python API + CLI; ingest, inspect, tag | yes |
| **Analytics user** (notebook, BI) | Polars / DuckDB / DataFusion directly against the Lance dataset | secondary |
| **Shared-server deployment** | HTTP MCP + auth layer | out of scope for MVP |

The entire design is resolved in favor of **primary actors**. When secondary and primary interests conflict, primary wins.

---

## 3. Use cases -- in scope

Each use case below maps to one or more functional requirements in §5. If a use case is added here, at least one FR must be added too; otherwise it's aspirational and belongs in the roadmap, not here.

### UC-1 -- Ingest an ontology into a local KG

**Actor:** ontology engineer or agent onboarding a project.
**Goal:** take a TTL/JSON-LD/N-Quads file and produce a local, queryable KG directory.
**Trigger:** `Dataset.open(path).ingest_ttl("schema.ttl")` or the `ingest` MCP tool.
**Main flow:**
1. Parse RDF terms with `oxttl` (via pyoxigraph).
2. Map quads into the Arrow schema (§6.1).
3. Append as a new Lance version.
4. Return the number of quads written and the new version number.
**Acceptance:** ingest of a 100k-triple file completes in under 10s on commodity hardware; result is SPARQL-queryable immediately.

### UC-2 -- Agent asks "what do I know about X?"

**Actor:** local agent via MCP.
**Goal:** fetch everything the KG knows about an IRI in a form an LLM can reason over, without writing SPARQL.
**Trigger:** `entity(iri, hops=1)` MCP tool.
**Main flow:**
1. Execute bounded outbound + inbound triple patterns for the IRI.
2. If `hops > 1`, breadth-first expand to IRI neighbors.
3. Return a structured dict with `outgoing`, `incoming`, and optional `neighbors`.
**Acceptance:** the returned dict is valid JSON, round-trips through `json.dumps`, and includes every triple the pattern matches. No extra triples, no missing ones.

### UC-3 -- Agent reasons, checkpoints, writes, rolls back if wrong

**Actor:** local agent making inferences.
**Goal:** let the agent add inferred triples without risking a poisoned KG.
**Trigger:** `checkpoint("pre-reasoning") → insert(…) → rollback("pre-reasoning")` (or the MCP equivalents).
**Main flow:**
1. Agent tags the current Lance version before writing.
2. Agent writes speculative triples -- new Lance version, old version still intact.
3. Validator judges the result.
4. If bad: `rollback` re-opens the dataset pinned to the earlier tag. Future reads see the old state.
**Acceptance:** after rollback, `dataset.count()` and SPARQL results match the pre-checkpoint state exactly.

### UC-4 -- Ship a KG inside a project / container / SDK

**Actor:** a team that bundles turtlelake with a codebase, container image, or domain SDK.
**Goal:** give every consumer of the artifact the same KG with no provisioning step.
**Trigger:** repo clone / `docker run` / SDK install.
**Main flow:**
1. The KG directory is committed alongside code (or baked into the image).
2. On first open, the agent reads the dataset at its current version.
3. The agent answers queries without any network or credentials.
**Acceptance:** `tar czf kg.tgz kg.turtlelake/ && scp kg.tgz host2 && tar xzf kg.tgz && open + query` on the remote host returns identical results.

### UC-5 -- Reproducible agent evaluation

**Actor:** CI / eval harness.
**Goal:** benchmark an agent against a frozen KG state so variance isn't caused by data drift.
**Trigger:** `Dataset.open(path, tag="eval-v3")`.
**Main flow:**
1. Evaluator opens the KG pinned to a tag.
2. Runs the agent N times against it.
3. Variance is attributable to the agent alone, not the KG.
**Acceptance:** the same dataset path and tag produce byte-identical query results across runs on different machines.

### UC-6 -- Columnar analytics on the same file

**Actor:** analytics user / data engineer.
**Goal:** compute aggregates over triples (counts, groupings, histograms) with DataFrame ergonomics.
**Trigger:** `polars.scan_pylance(dataset.triples_path)` or DuckDB / DataFusion equivalent.
**Main flow:**
1. Reader opens the Lance dataset directly, bypassing turtlelake's Python API.
2. Applies a query over the Arrow schema (`subject`, `predicate`, `object`, `object_kind`, `object_datatype`, `object_lang`, `graph`).
3. Returns a DataFrame or Arrow table; zero-copy where the engine supports it.
**Acceptance:** `SELECT object_kind, COUNT(*) FROM triples GROUP BY 1` runs in DuckDB against the Lance directory without any turtlelake import.

### UC-7 -- MCP integration with Claude Code / Cursor

**Actor:** local agent runtime.
**Goal:** surface all of the above as MCP tools with one install command.
**Trigger:** `claude mcp add turtlelake -- uvx turtlelake-mcp`.
**Main flow:**
1. Agent runtime spawns the MCP server over stdio.
2. Tools registered: `sparql`, `entity`, `ingest`, `checkpoint`, `rollback`, `versions`, `scan`.
3. Every tool call is stateless except writes (§5, FR-6).
**Acceptance:** all seven tools are callable from Claude Code without code changes; help strings are agent-legible.

### UC-8 -- Offline / air-gapped operation

**Actor:** agent on a Flatcar on-prem host, a factory-floor laptop, or a disconnected devkit.
**Goal:** answer KG queries with zero network access.
**Trigger:** normal operation when offline.
**Main flow:** identical to UC-2/UC-3/UC-6. Nothing in turtlelake's runtime path touches the network.
**Acceptance:** `pip install turtlelake` on a pre-fetched wheel, then full agent workflow, passes with the NIC disabled.

### UC-9 -- Validate writes against SHACL shapes before accepting them

**Actor:** local agent or ingest pipeline.
**Goal:** reject KG writes that violate a shapes graph (e.g. "every Device must have a partNumber"); useful as the gate between speculative agent writes and a permanent tag.
**Trigger:** `Dataset.validate(shapes_ttl)` (or the `validate` MCP tool) between `checkpoint` and `tag`.
**Main flow:**
1. The shapes graph is loaded from a user-provided TTL.
2. pySHACL runs against the current dataset version.
3. Validation report is returned as structured JSON (conforms, violations with focus-node + path + value).
4. Agent decides: keep the version or `rollback`.
**Acceptance:** a shapes file that requires every `a Device` to have an `rdfs:label` correctly flags a labelless device and is a no-op when the constraint is satisfied.

### UC-10 -- Provenance per version (who / when / why)

**Actor:** regulated domain (finance, medical, legal) or any agent whose outputs are audited.
**Goal:** for every triple in the KG, recover the version it was written in, the source (TTL file path, URL, agent ID, or "manual"), the author, and the timestamp.
**Trigger:** every ingest/checkpoint records metadata; `Dataset.provenance(version=N)` retrieves it.
**Main flow:**
1. `ingest_ttl(src, *, source=..., author=...)` attaches source URI, author, ISO timestamp to the resulting Lance version as metadata.
2. `Dataset.provenance()` returns a JSON list of `{version, source, author, timestamp, row_delta}`.
3. Audit tools consume this list to build a chain-of-custody report.
**Acceptance:** a mixed ingest sequence (two TTL files + an agent-write) produces three distinct provenance records in order; deleting the files on disk does not erase the provenance.

### UC-11 -- Multi-agent coordination via shared KG state

**Actor:** two or more agents on the same machine (same container / same runner).
**Goal:** let agent A write a triple (e.g. "ticket #42 status=triaged") and have agent B read it without any message bus.
**Trigger:** two processes open the same dataset path.
**Main flow:**
1. Agent A opens `./kg.turtlelake`, ingests or writes a quad, commits a new Lance version.
2. Agent B -- which may have been running with an older version cached -- calls `Dataset.refresh()` (or simply re-opens) and observes the new version.
3. Lance's MVCC means reads are consistent to the version the reader opened; no torn reads.
**Acceptance:** two Python processes demonstrate write-then-read-then-observe on the same dataset with no explicit locking.

### UC-12 -- Diff two KG versions

**Actor:** reviewer of an agent-generated inference batch, or ontology engineer releasing a new version.
**Goal:** see added and removed triples between two Lance versions so the change can be code-reviewed.
**Trigger:** `Dataset.diff(from_version=N, to_version=M)`; or `turtlelake diff --from v1 --to v2` on the CLI.
**Main flow:**
1. Open the dataset pinned at version N and version M.
2. Compute set-difference on quads (hash the 4-tuple to avoid materializing the whole table where possible).
3. Return two Arrow tables: `added`, `removed`.
**Acceptance:** between a version with N triples and one with N+3, `diff` returns exactly those 3 added quads and zero removed.

### UC-13 -- Durable agent memory (personal / conversation KG)

**Actor:** a long-running agent that accumulates facts about its user or environment across sessions.
**Goal:** store facts as triples ("user prefers SPARQL over Cypher"; "project X uses Postgres 16"); retrieve them into the next turn's context.
**Trigger:** agent writes ad-hoc quads via `Store.insert()` or its MCP equivalent; retrieves via `entity(user_iri)`.
**Main flow:**
1. Agent synthesises observations into quads with a stable subject (`user:david`, `project:my-toolkit`).
2. Quads are appended as a new Lance version; no checkpoint needed for low-stakes memory.
3. Next session, agent opens the dataset, calls `entity(user_iri)`, receives the subgraph, folds it into the system prompt.
**Acceptance:** facts written in one process are retrievable by `entity()` in a fresh process on the same path.

### UC-14 -- Publish / distribute a KG as an artifact

**Actor:** ontology maintainer, dataset publisher.
**Goal:** ship a versioned KG to consumers the way LanceDB ships datasets to Hugging Face or S3.
**Trigger:** `Dataset.open("s3://bucket/kg.turtlelake/")` or `Dataset.open("hf://datasets/org/kg")`.
**Main flow:**
1. Author writes the dataset locally, tags a release (`author.tag("v1.0")`).
2. Author uploads the directory to S3 / HF / any object store.
3. Consumers open it with the remote URI; reads go through Lance's remote-storage layer.
**Acceptance:** a dataset written locally, uploaded to an S3-compatible store, then opened via `s3://...` URI returns identical SPARQL results.

### UC-15 -- Privacy-first personal / on-device KG

**Actor:** an individual user's agent on their laptop or phone.
**Goal:** keep all structured knowledge about the user on their device; give no server, no company, no backup provider access to the raw triples.
**Trigger:** the dataset lives under the user's home directory only.
**Main flow:** identical to UC-2/UC-3/UC-13. No code path reaches the network. Encryption at rest is delegated to the filesystem (LUKS/APFS).
**Acceptance:** inspection of outbound packets during a full agent session shows zero egress from turtlelake code.

### UC-16 -- Per-IRI embeddings stored alongside the graph

**Actor:** an agent (or its harness) that has computed vectors for some
or all entities in the KG.
**Goal:** persist those vectors in the same artifact as the triples so
the dataset directory is the whole RAG index -- no second store, no
sync.
**Trigger:** `Dataset.embed(iris, vectors, model_id=...)` or the `embed`
MCP tool.
**Main flow:**
1. Caller supplies pre-computed float32 vectors plus an embedding model
   identifier. turtlelake never loads a model.
2. Vectors are appended to `embeddings.lance/`. First write fixes the
   dimension and records it in `manifest.json`.
3. A new Lance version is produced on the embeddings dataset; the
   triples dataset is unaffected.
**Acceptance:** after `embed`, `embedding_count()` reflects the rows
just written and `embedding_dim()` matches the input. Re-opening the
directory in a fresh process recovers both.

### UC-17 -- GraphRAG retrieval (semantic search → structural expansion)

**Actor:** an agent answering a user question by retrieving relevant
entities and the facts around them.
**Goal:** find the IRIs whose meaning is closest to a query embedding
and pull each one's neighborhood -- the canonical GraphRAG retrieval
shape -- without leaving the dataset directory.
**Trigger:** `Dataset.graph_rag(query_vector, k, hops)` or the
`graph_rag` MCP tool.
**Main flow:**
1. ANN search over `embeddings.lance/` returns the `k` IRIs nearest to
   `query_vector`.
2. For each hit, `entity()` walks the triples dataset to depth `hops`.
3. Hits and entities are returned in one structured response.
**Acceptance:** for a seeded dataset where IRI A's vector is closer to
the query than IRI B's, `graph_rag(..., k=1)` returns A and its
neighborhood. Vector + graph results pin to the same checkpoint when
the call is made under a tagged read.

---

## 4. Use cases -- explicitly out of scope

Declared here so we say "no" fast instead of slowly drifting into them. Each has a suggested alternative.

| Out of scope | Why | Use instead |
|---|---|---|
| Multi-tenant / hosted SPARQL server | Conflicts with P1 (local and embedded) | Oxigraph server, GraphDB |
| OLTP-heavy writes (>1k triples/sec) | Lance is append-optimized; we borrow pyoxigraph for reads | Oxigraph on RocksDB |
| Federated SPARQL across remote endpoints | Requires network and credential management | Apache Jena ARQ |
| Real-time / streaming ingest | v0 is batch append; streaming contradicts the versioning model | Kafka + a streaming consumer that batches into turtlelake |
| Labeled Property Graph / Cypher queries | Different data model (see §6 philosophy) | **kglite**, Neo4j |
| Distributed / cluster query execution | Not a local-agent shape | Trino, DuckDB-on-Ballista, Spark |
| SPARQL UPDATE with arbitrary WHERE clauses in MVP | Requires a full round-trip rewrite | `INSERT DATA` / `DELETE DATA` only in MVP; full UPDATE on roadmap |
| Datasets > 100M triples at v0 | v0 SPARQL materializes into RAM | Wait for M7 (rdf-fusion executor) |
| OWL/RDFS reasoning at query time | Out of scope for this layer; reasoners are separate | Run a reasoner (ELK, Pellet, Open Ontologies' tableaux) and ingest the inferred triples as a new version |
| **Building our own reasoner of any kind** | Reasoners are correctness-critical, mature OSS exists, and owning one would double the project. SPARQL 1.1 property paths cover the transitive cases natively; everything else is an external materialization step. | Open Ontologies (OWL2-DL), `owlrl` (OWL-RL Python), ELK (EL profile), HermiT / Pellet (OWL-DL legacy) |
| **Agentic extraction of triples from unstructured text** | That's an agent's job; turtlelake stores what the agent produces | KG-Orchestra, Graphiti, AutoBioKG |
| **Agent-to-agent messaging / orchestration** | Not a storage concern; solve via MCP-level tools, not the KG | LangGraph, CrewAI, OpenAI Agents SDK |
| **Web UI for visual ontology editing** | We ship a file format, not an editor | Protégé, OntoInk, metaphacts |
| **Authoring SHACL shapes automatically from data** | Learning shapes is a research problem of its own | W3C SHACL shape-learners, Shaclex |
| **Git-style branching of the dataset across agents** | Post-MVP; Lance supports it but adds surface area we're not ready to own | Lance native branch API (direct use) until we wrap it |

---

## 5. Functional requirements

Each FR is testable and maps to at least one use case.

| ID | Requirement | Covers |
|---|---|---|
| FR-1 | `Dataset.open(path, *, tag=None, version=None)` opens or creates a dataset at `path`. When `tag` or `version` is given, subsequent reads are pinned to that point. | UC-1, UC-5 |
| FR-2 | `Dataset.ingest_ttl(src)` accepts TTL, TriG, N-Triples, N-Quads, JSON-LD, and RDF/XML. Each call produces a new Lance version. | UC-1 |
| FR-3 | `Dataset.query(sparql)` executes SPARQL 1.1 SELECT/ASK/CONSTRUCT/DESCRIBE against the currently-open version. Results are JSON-serializable. | UC-2, UC-5, UC-7 |
| FR-4 | `Dataset.entity(iri, hops=N)` returns `{iri, outgoing, incoming, neighbors?}`. Hop count ≥ 1; neighbors are expanded breadth-first only when `hops > 1`. | UC-2, UC-7 |
| FR-5 | `Dataset.tag(name)` / `.tags()` / `.versions()` expose Lance's tag/version surface as a thin pass-through. | UC-3, UC-5 |
| FR-6 | `Dataset.checkpoint(name)` tags the current version; `Dataset.rollback(name)` returns a new handle pinned to that tag. Rollback must not mutate existing versions. | UC-3 |
| FR-7 | The on-disk dataset at `<path>/triples.lance/` is a valid Lance dataset with the schema in §6.1; any Arrow-compatible reader can consume it without importing turtlelake. | UC-4, UC-6 |
| FR-8 | An MCP stdio server exposes `sparql`, `entity`, `ingest`, `checkpoint`, `rollback`, `versions`, `scan`. | UC-7 |
| FR-9 | All runtime code paths function without network access. No telemetry, no update checks, no required external services. | UC-8 |
| FR-10 | Identifier-free dataset portability: `tar` → `cp` → `untar` → `open` yields byte-identical query results on any host with compatible Lance version. | UC-4, UC-8 |
| FR-11 | `Dataset.validate(shapes_ttl)` runs pySHACL against the current version and returns a JSON report (conforms, violations with focus-node + path + value). | UC-9 |
| FR-12 | Every write (`ingest_ttl`, `checkpoint`, explicit quad inserts) records `{source, author, timestamp}` as Lance version metadata; `Dataset.provenance()` returns the ordered list. | UC-10 |
| FR-13 | `Dataset.diff(from_version, to_version)` returns two Arrow tables, `added` and `removed`, with the quad schema from §6.1. | UC-12 |
| FR-14 | `Dataset.open(uri)` accepts remote URIs (`s3://`, `gs://`, `az://`, `hf://`, `file://`) for read; write to remote stores is optional at MVP. | UC-14 |
| FR-15 | `Dataset.refresh()` re-opens the dataset at the latest committed version, enabling reader processes to observe writes made by sibling processes. | UC-11 |
| FR-16 | `Dataset.insert_turtle(ttl_text)` (and `Dataset.insert(quads)`) append quads without requiring a file; records provenance like `ingest_ttl`. Agent-memory write entry point. | UC-13 |
| FR-17 | `Dataset.open(path, sources={graph_iri: file})` attaches read-only external TTLs. Each source is loaded into its own named graph in the cached engine, **never copied into Lance**. mtime changes upstream are picked up on the next query. | UC-M1, UC-M4 |
| FR-18 | When sources are attached and `graph=` is not provided, writes (`insert_turtle`, `insert`, `ingest_ttl`) default to the `turtlelake://agent-overlay` named graph so vendor and agent data stay separable. With no sources, writes go to the default graph (backwards-compatible). | UC-M2 |
| FR-19 | Sources are read-only. Writes never modify the source files; rollback reverts only the overlay. | UC-M3 |
| FR-20 | `Dataset.dump(path, format=, graph=)` serializes the overlay (or one specific named graph) to TTL / N-Quads / N-Triples / RDF-XML / JSON-LD. | UC-M6 |
| FR-21 | `Dataset.sources()` returns attached sources as `[{graph, path, mtime, sha256}]`. `schema()` includes the same list. | UC-M5 |
| FR-22 | `follow_imports=True` (opt-in, local files only) transitively resolves `owl:imports` from each attached source, cycle-safe. | UC-M7 |
| FR-23 | MCP boot honors `TURTLELAKE_SOURCES` (JSON or `graph=path,graph=path` form) to attach sources without Python code. | UC-M8 |
| FR-24 | `query()` uses `use_default_graph_as_union=True` so a pattern like `?s ?p ?o` sees every named graph. Agents can still scope with `GRAPH <iri> { ... }`. | UC-M1..M7 |

---

## 6. Non-functional requirements

| ID | Quality attribute | Target |
|---|---|---|
| NFR-1 | **Startup latency** (open → first SPARQL on ≤1M triples) | < 500 ms |
| NFR-2 | **Ingest throughput** | ≥ 10k quads/sec on commodity hardware for Turtle input |
| NFR-3 | **Portability** | Dataset directory round-trips across Linux/macOS/Windows without byte-level modification |
| NFR-4 | **Dependency surface** | Python 3.11+, three hard dependencies (pyoxigraph, pylance, pyarrow); no system packages required |
| NFR-5 | **Open-format guarantee** | Schema in §6.1 is part of the public API; breaking it is a major version bump |
| NFR-6 | **Read-side safety** | `sparql`, `entity`, `scan`, `versions` never mutate the dataset -- including no temp files inside the dataset directory |
| NFR-7 | **Write-side safety** | Every `ingest`, `checkpoint`, `rollback` is a distinct Lance version -- no silent in-place mutation |
| NFR-8 | **Observability** | Every write logs version number + row count to stderr as structured JSON (for the MCP audit layer) |
| NFR-9 | **Security posture** | MCP server inherits the repo's security base (input scan, audit, timing-safe auth when deployed beyond stdio). **Rate limits are per-tool, calibrated to how the tool is actually used**: high caps for cheap reads (entity, refresh -- 200+/min), moderate for writes (ingest/checkpoint -- 30–120/min), low for destructive ops (rollback -- 10/min), moderate for memory-heavy ops (diff, validate -- 20–30/min). Overridable via `TURTLELAKE_RATE_LIMIT_<TOOL>` env vars; global `TURTLELAKE_RATE_LIMIT` applies to all. Zero disables the limit. |
| NFR-10 | **Remote-storage parity** | A dataset at `s3://` / `hf://` / `gs://` opens, reads, and queries with the same API and comparable semantics to a local path |
| NFR-11 | **Provenance survives copy** | Copying / `tar`-ing a dataset directory preserves per-version metadata (source, author, timestamp) verbatim |
| NFR-12 | **Concurrency** | Multiple processes may open the same dataset read-only concurrently; exactly one writer at a time (Lance's write semantics) |

### 6.1 Triple schema (public API)

```
subject         : string              # IRI or blank-node label
predicate       : string              # IRI
object          : string              # IRI, blank-node label, or literal lexical form
object_kind     : string              # "iri" | "bnode" | "literal"
object_datatype : string (nullable)   # literals only
object_lang     : string (nullable)   # literals only
graph           : string (nullable)   # null = default graph
```

Field names, order, nullability, and semantics are a stability contract. Additions (new columns) are minor-version; removals/renames are major.

---

## 7.1 Known limitations

Decisions that are deliberately out of scope at MVP. Not bugs; documented here so reviewers don't re-raise them.

| Limitation | Why | Workaround |
|---|---|---|
| An IRI whose path starts with `_:` is stored and read back as a blank node | We use the `_:` prefix as the storage convention for blank-node labels; adding a separate `subject_kind` / `graph_kind` column would be a major schema bump. RFC allows such IRIs but they're vanishingly rare in practice. | Avoid IRIs with `_:`-prefixed authorities; or prefix such IRIs with a namespace in your ontology. |
| `diff()` materializes both versions into RAM | v0 SPARQL also forces everything into RAM via pyoxigraph, so this is not an incremental cost. | Don't diff across versions with more triples than fit in memory. Lance-native fragment-level diff lands when the SPARQL route planner lands. |
| Remote URI `_exists()` probes via `lance.dataset(uri)` | Object-store HEADs aren't universally available through pylance; a positive probe is "Lance can open it", not "the bytes are there". | Good enough for a read-after-write on the same agent. For CI, probe the object store directly. |
| `parse_rdf_file` yields inside a `with` block | Safe today because `_append_quads` consumes synchronously. Fragile only if a future change buffers the generator. | Internal callers must stay synchronous; documented invariant. |

## 7. Constraints

Picked deliberately, not negotiable inside this project.

| Area | Choice | Rationale |
|---|---|---|
| Language | Python 3.11+ | Rust engines already ship Python bindings; glue stays Python |
| Storage | Lance | Only embedded format with columnar + Arrow + versioning together |
| SPARQL engine (v0) | pyoxigraph | Full SPARQL 1.1 from day one |
| SPARQL engine (v1) | RDF Fusion on DataFusion | Columnar execution when mature; no API change to callers |
| RDF parser | oxttl via pyoxigraph | Covers every W3C serialization we care about |
| MCP framework | FastMCP | Official SDK path, stdio + HTTP transports |
| Config | Turtle (not YAML) for index rules | Eat our own dogfood where possible |

---

## 8. Acceptance criteria (MVP "done")

turtlelake is MVP-complete when every item below is green:

1. `pip install turtlelake` on a clean Python 3.11+ environment works with no system packages.
2. `quickstart.py` runs end-to-end: ingest → SPARQL → Arrow scan → tag.
3. `agent_workflow.py` runs end-to-end: checkpoint → speculative write → rollback → counts match.
4. All use cases UC-1 through UC-15 have at least one automated test or runnable example. Aspirational UCs (UC-14 remote storage) may be exercised via a local `file://` URI in tests.
5. A dataset directory created on host A opens and returns identical query results on host B.
6. MCP stdio server registers `guide`, `schema`, `sources`, `sparql`, `entity`, `scan`, `explain`, `ingest`, `insert`, `checkpoint`, `rollback`, `versions`, `refresh`, `diff`, `provenance`, `validate`, `dump`, `save_query`, `run_saved` (19 tools) and each is callable from Claude Code. Additions over the 16-tool surface: `explain` returns a query-plan sketch; `save_query` / `run_saved` give agents a parameterized query library (injection-safe via SPARQL substitutions). `sources={graph_iri: url}` also accepts `http(s)://` URLs, fetched once and cached with ETag validation -- simple federation over static remote TTL sources.
7. Schema in §6.1 is readable by DuckDB and Polars without importing turtlelake, and `SELECT object_kind, COUNT(*) FROM triples GROUP BY 1` returns a sensible result.
8. `Dataset.validate(shapes)` returns a non-empty violation report when given a shapes file the data violates, and an empty report when it doesn't.
9. `Dataset.diff(v_old, v_new)` on a two-ingest sequence returns exactly the triples added in the second ingest.
10. `Dataset.provenance()` returns at least one record per write path exercised in §8.2/§8.3.
11. README + ARCHITECTURE + this document agree on naming, scope, and the list of use cases.

Items 1–3 and 6-partial are covered by the code on this branch. Items 4–5, 7–11 are the next focused push.

---

## 9. Research provenance (how this list was derived)

Additions after the initial draft trace to these sources. Kept here so future reviewers can see where each UC came from and whether a signal has since changed.

| UC / FR added | Source that prompted it |
|---|---|
| UC-9 SHACL validation on write | W3C SHACL standard, pySHACL, OntoFlow pipeline, xpSHACL (2026) -- validation as a CI/CD gate is the mainstream ontology-engineering practice |
| UC-10 Provenance per version | 2026 enterprise-RAG consensus that audit trails linking answers to source are a required capability in regulated domains; Glean/ZBrain/NStarX 2026 writeups |
| UC-11 Multi-agent shared state | Multi-agent orchestration literature (AutoBioKG, KG-Orchestra, AGENTiGraph) -- agents' outputs and inputs flow through a shared semantic substrate |
| UC-12 Diff two versions | lakeFS + DVC usage patterns; ontology-engineering workflows use `bubastis` to diff OWL versions (OntoFlow) |
| UC-13 Durable agent memory | LanceDB Cognee customer pattern (durable AI memory); local-first AI thesis (2026 Edge-AI-Vision, Programming Insider) |
| UC-14 Publish / distribute KG artifact | Lance × Hugging Face Hub release (Feb 2026); LanceDB remote-storage read paths |
| UC-15 Privacy-first on-device KG | Local-first AI drivers (privacy, latency, availability, cost); GAIA, local LLM movement 2026 |
| Out-of-scope additions | Reasoners (Open Ontologies -- fabio-rovai, a direct competitor with an in-memory Oxigraph + tableaux OWL reasoner); extraction (Graphiti, KG-Orchestra -- agents build the KG, we store it); UI (Protégé, OntoInk); shape-learning (academic, not productized) |

Open Ontologies (fabio-rovai on GitHub) warrants a direct comparison: same ecosystem (Oxigraph, SPARQL, SHACL, MCP server), different bet -- they are reasoner-first in-memory; turtlelake is storage-first persistent with an open Arrow file format. We expect many users to run both (turtlelake stores, Open Ontologies reasons over a materialized read).
