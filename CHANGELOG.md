# Changelog

All notable changes to turtlelake are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

Repo-hygiene polish: this changelog, a contributor guide, and a
pre-commit config. No code changes.

### Added
- `CHANGELOG.md` (this file).
- `CONTRIBUTING.md` -- dev setup, testing, linting, conventions for
  adding domain demos and benchmarks, PR guidelines.
- `.pre-commit-config.yaml` -- ruff (lint + format) plus a small set
  of file-hygiene checks (trailing whitespace, end-of-file fixer,
  large-file guard, valid YAML/TOML).

---

## [0.0.1] -- 2026-05-04

The "GraphRAG pivot" release. Repositions turtlelake from "embedded
RDF lakehouse" to **embedded graph + vector store for ontology-grounded
agents**.

### Added -- vector layer

- Sibling `embeddings.lance/` dataset alongside `triples.lance/`.
- `Dataset.embed(iris, vectors, *, model_id, author=None)` -- append
  per-IRI embeddings. Validates NaN/Inf, dimension consistency,
  `model_id` whitelist, and per-call row caps
  (`TURTLELAKE_MAX_VECTORS_PER_EMBED`, default 1M) and dim caps
  (`TURTLELAKE_MAX_EMBEDDING_DIM`, default 16k).
- `Dataset.vector_search(query, *, k, model_id, nprobes,
  refine_factor, in_memory)` -- Lance ANN with optional in-RAM
  matmul fast path.
- `Dataset.vector_search_batch(queries, *, k, in_memory)` -- one
  BLAS GEMM call vs N Python-level Lance calls.
- `Dataset.graph_rag(query, *, k, hops, model_id)` -- vector
  retrieval + structural expansion in one call.
- `Dataset.graph_rag_ppr(query, *, k, seed_k, damping, iterations,
  edge_predicates)` -- HippoRAG-style Personalized PageRank over the
  entity graph.
- `Dataset.bm25_search(text, *, k)` -- TF-IDF lexical search over
  cached label/definition predicates.
- `Dataset.hybrid_search(text, vec, *, k, rrf_k)` -- BM25 + vector
  fused via Reciprocal Rank Fusion.
- `Dataset.entity(iri, *, hops, similar, model_id)` -- `similar > 0`
  appends nearest-neighbor IRIs by stored vector.

### Added -- caching and tuning

- `Dataset.preload_vectors(*, model_id)` -- load all embeddings into
  a contiguous numpy matrix kept in process memory. Bypasses Lance
  for sub-millisecond per-query search; 18.8× faster than Lance on
  batched 50k workloads.
- `Dataset.preload_text_index(*, predicates, max_features)` -- fit
  TF-IDF over chosen literal predicates.
- `Dataset.has_warm_cache(*, model_id)` -- cache state introspection.
- `Dataset.tune_nprobes(*, target_recall, sample_queries, seed)` --   binary-search the smallest `nprobes` that hits a target recall on
  a sampled query set.
- `Dataset.build_vector_index(*, index_type="auto", metric,
  num_partitions, num_sub_vectors)` -- auto-policy picks by row count
  (skip below 10k, IVF_FLAT below 1M, IVF_PQ at scale).

### Added -- versioning + recovery

- Crash-safe paired checkpoint: `manifest.json` records a
  `pending_checkpoint` write-ahead-log marker before either Lance
  tag is created, then clears it once both succeed. A crash
  mid-checkpoint is reconciled on the next `Dataset.open(...)` --   forward-rolls in either direction (triples-tagged-but-embeddings-not
  or vice versa).
- Manifest writes use tmp + `os.replace` + `fsync` so a killed
  writer cannot leave torn JSON on disk.
- `Dataset.compact()` -- merge small Lance fragments on both datasets.
- `Dataset.prune_versions(*, keep_versions)` -- drop old versions on
  both datasets, preserve tagged ones. Returns a stable per-side
  summary shape across pylance versions.
- `Dataset.embedding_versions()` -- surface that the two Lance
  datasets advance independently between checkpoints.

### Added -- tooling

- `Dataset.checkpoint(name)` and `rollback(name)` are now
  paired-atomic across triples and embeddings: rollback restores
  both sides.
- 25 MCP tools (was 19): added `embed`, `vector_search`, `graph_rag`,
  `build_vector_index`, `compact`, `prune_versions`. Each has
  per-tool rate limits in `security.py`.
- `@turtlelake/client` npm package
  ([`clients/typescript/`](./clients/typescript/)) -- spawns
  `turtlelake-mcp` over stdio, completes MCP `initialize`, exposes
  typed wrappers for all 25 tools. Detects the `@secure` decorator's
  `{"error": ...}` envelope and converts to rejected promises.
- `.owl` file extension support in `ingest_ttl` (treated as RDF/XML).

### Added -- benchmarks

- `scripts/benchmark_graphrag.py` -- synthetic 2-hop authorship task;
  isolates the retrieval-shape lift between flat search and graph_rag.
- `scripts/benchmarks/musique.py` -- multi-hop QA against the
  HippoRAG-sampled MuSiQue subset (1k questions, 11.6k paragraphs).
  Scores flat / graph / hybrid / ppr.
- `scripts/benchmarks/ann_sift1m.py` -- ANN-Benchmarks-style recall@10
  + QPS at 1M × 128 (Gaussian-on-sphere when canonical SIFT is
  unreachable).
- `scripts/benchmarks/operational.py` -- five capability benchmarks
  where turtlelake wins by design: open-to-first-query (16 ms cold,
  13 ms warm), update-visibility (2 ms median), reproducibility at
  tag (byte-identical), crash-recovery convergence, in-RAM speedup.
- `scripts/benchmarks/precache_st_model.py` -- Tier-3 prep: download
  sentence-transformers offline.
- `scripts/benchmarks/openie_extract.py` -- Tier-3 prep: LLM-extracted
  entity-relation triples for HippoRAG-density graphs.

### Added -- domain demos

Four end-to-end demos against real public ontologies, each printing
"naive LLM vs ontology-grounded agent" comparisons inline.

- `examples/demo_legal_lkif.py` -- LKIF Core (~1200 quads).
- `examples/demo_medical_doid.py` -- Disease Ontology cancer slim
  (~14k quads, 729 disease terms).
- `examples/demo_science_go.py` -- full Gene Ontology (48k terms),
  BFS-subset around a seed.
- `examples/demo_gov_dcat.py` -- W3C DCAT 3 (~1700 quads).
- `examples/_demo_runner.py` -- shared download cache + naive/grounded
  printer.

### Added -- concurrency

- Multi-process write contract documented in `ARCHITECTURE.md`.
  Concurrent appends from N=2..3 spawn-mode workers all succeed;
  reads during writes are snapshot-consistent; `compact()` runs
  concurrently with readers without corruption.
- `tests/test_concurrent_writes.py` -- five integration tests that
  pin the contract.
- `tests/_concurrent_helpers.py` -- module-level worker functions
  required by spawn-mode (Lance is not fork-safe).

### Changed

- `vector_search` gained `nprobes`, `refine_factor`, and `in_memory`
  parameters (the standard ANN tunables, plus the in-RAM matmul
  fast path).
- `entity` gained `similar` and `model_id` parameters.
- `_similar_to_iri` resolution is now deterministic -- picks the
  most recent `(created_at, model_id)` match rather than relying on
  Lance scan order.
- `_sql_escape` hardened: rejects null bytes, ASCII control chars,
  and Unicode bidi overrides; doubles backslash *before* quotes so
  `\'` cannot become an unescaped `'`.
- `prune_versions` returns a stable shape across pylance versions
  via `_summarize_cleanup` / `_empty_cleanup_summary` helpers.
- README rewritten for impact (Demos promoted, comparison table
  expanded, voice tightened).

### Removed

- Altera/FPGA-themed examples and the `examples/altera/` directory
  (replaced by the four domain demos).
- `examples/comparison.py`, `examples/COMPARISON.md`,
  `examples/agent_workflow.py`, `examples/full_agent_demo.py`,
  `examples/ontology.ttl`.
- Author section from README.
- `Hybrid graph + vector in one store` from `REQUIREMENTS.md`'s
  out-of-scope table -- it is now in scope.

### Fixed

- `prune_versions` datetime tz-naive vs tz-aware subtraction edge
  case.
- `build_vector_index(index_type="IVF_PQ")` on datasets with fewer
  than 256 rows now raises a clear turtlelake-level error pointing
  at `IVF_FLAT`/`auto`, instead of Lance's underlying Rust panic.

### Tests

- 229 tests passing (was ~80 before this release). New suites:
  `test_in_memory_search.py` (12 tests), `test_concurrent_writes.py`
  (5), `test_crash_recovery.py` (6), `test_maintenance.py` (13),
  `test_vector_validation.py` (14), `test_vectors.py` (13),
  `test_graph_rag.py` (5), `test_atomic_rollback.py` (4).

---

## [pre-0.0.1] -- earlier

- Initial RDF + SPARQL + Lance versioning + MCP server (19 tools).
- README + ARCHITECTURE + REQUIREMENTS + TEST_SCENARIOS as written
  during the original MVP push.
