# Contributing to turtlelake

Thanks for your interest. turtlelake is a small project trying to do
one specific thing well -- embedded graph + vector retrieval for local
AI agents -- and we welcome help that makes it sharper at that.

This guide covers dev setup, the conventions we follow, and how to
contribute the kinds of changes that come up most often.

---

## Quick start

```bash
git clone https://github.com/archloom/turtlelake.git
cd turtlelake
uv sync                                       # installs core + dev + all extras
uv run pytest                                 # 229 tests should pass
uv run ruff check src tests examples scripts  # lint
```

If you don't have [uv](https://github.com/astral-sh/uv) yet, the same
commands work with `pip install -e ".[dev,mcp,shacl,sql,metrics]"` and
`python -m pytest`.

---

## Pre-commit (recommended)

We ship a `.pre-commit-config.yaml` that runs ruff + a few file-hygiene
checks on every commit. To enable it:

```bash
uv pip install pre-commit
uv run pre-commit install
```

Now `git commit` will lint your changes locally before you push.

---

## Tests

```bash
uv run pytest                                # full suite
uv run pytest tests/test_vectors.py -v       # one file
uv run pytest -k "rollback"                  # filter by name
```

The test suite covers concurrency (spawn-mode multi-process workers),
crash recovery (kills a subprocess mid-checkpoint, validates WAL
recovery), hybrid retrieval, PPR, in-RAM cache invariants, and the
full MCP tool surface. New code should be tested.

Two slow tests are deselected by default (`test_query_timeout_raises_cleanly`,
`test_speedup`) -- they run fine but take significant wall time. Run
them explicitly when relevant:

```bash
uv run pytest tests/test_advanced_features.py::test_query_timeout_raises_cleanly
uv run pytest tests/test_speedup.py
```

---

## Lint and format

```bash
uv run ruff check src tests examples scripts          # lint
uv run ruff check --fix src tests examples scripts    # auto-fix
uv run ruff format src tests examples scripts         # format
```

Settings live in `pyproject.toml` under `[tool.ruff]`. We aim for
clean lint on everything in `src/`, `tests/`, `examples/`, and
`scripts/`.

---

## Project conventions

### Code style

- Type hints on every public function. Internal helpers prefer them
  too.
- Docstrings on every public method, with the first line being a
  short imperative sentence. Multi-paragraph docstrings are fine for
  methods that have non-obvious semantics or trade-offs.
- Comments explain the *why*, not the *what*. If a comment restates
  the code, delete it; if it captures a non-obvious invariant or
  trade-off, keep it.
- Avoid emojis in code, comments, commit messages, and docs unless
  the surrounding context already uses them.

### Commit messages

Keep them short and structural. Convention:

```
<area>: <one-line summary in imperative mood>

<optional body explaining why, citing benchmarks / tests / links
where helpful>
```

Examples:

```
feat: add per-IRI vector layer + GraphRAG retrieval
harden: WAL checkpoints, ANN index, maintenance ops, input validation
docs: rewrite README for impact (421 lines, down from 505)
```

### Branches

Topic branches off `main`. The convention this repo currently uses
is `claude/<short-description>-<random-suffix>` for AI-assisted work,
but a plain descriptive name is fine:

```
git checkout -b feat/streaming-ingest
git checkout -b fix/prune-versions-tz-naive
```

### Pull requests

- Open against `main`.
- Title: short imperative summary that reads well in the merge log.
- Body: a *Summary* (1-3 bullets), then *Test plan* (bulleted
  checklist), then any benchmark numbers or design notes that help
  reviewers. Larger or capability-introducing PRs should also include
  what changed in `ARCHITECTURE.md` or the relevant test matrix
  entries; small bug-fix PRs deserve short bodies.
- Tests must pass and lint must be clean before review.
- One concern per PR. If a refactor touches three subsystems, split it.

---

## Adding things

### A new domain demo

Each demo follows the same shape -- see
[`examples/demo_legal_lkif.py`](./examples/demo_legal_lkif.py) for a
representative one.

1. Pick a public ontology that's reachable and small enough to ingest
   in seconds (or use a slim subset). Prefer GitHub-hosted mirrors.
2. Create `examples/demo_<domain>_<ontology>.py` and import from
   `examples/_demo_runner.py` for the shared download cache and the
   naive/grounded printer.
3. Walk through 4-6 steps that each highlight one turtlelake
   primitive (`bm25_search`, `entity`, SPARQL property paths,
   `validate`, `provenance`, etc.).
4. Print a `naive("...")` line for each step *before* the
   `grounded(...)` line so the value prop is visible without reading
   code.
5. Run the demo end-to-end on a fresh clone before opening a PR.
6. Add a one-line entry to the demos table in `README.md`.

### A new benchmark

See [`scripts/README.md`](./scripts/README.md) for the methodology
notes. New benchmarks live in `scripts/benchmarks/` and reuse
`_common.py` for the embedder factory and dataset download cache.
Run on a single machine, capture the configuration, write the
numbers into `scripts/README.md` with full disclosure of constraints.
We avoid made-up numbers -- if a dataset isn't reachable in your
environment, document the substitution and what it preserves vs
changes.

### A new MCP tool

1. Add the method on `Dataset` first, with tests.
2. Wire it through `src/turtlelake/mcp_server.py` using `@secure(...)`.
3. Add a per-tool rate limit in `src/turtlelake/security.py`'s
   `DEFAULT_RATE_LIMITS` dict.
4. Update the tool count in `tool_names()`,
   `tests/test_mcp_tools.py::test_tool_names_is_stable_and_complete`,
   and the README's MCP table.
5. Re-run `tests/test_mcp_e2e.py` to confirm the new tool registers
   end-to-end against the real binary.

### A new test

Tests live in `tests/`. Conventions:

- One test file per concern (`test_<concern>.py`).
- Use pytest fixtures over setup/teardown classes.
- Use `tmp_path` for any filesystem state -- the test must clean up
  after itself.
- For multi-process tests, use `multiprocessing.get_context("spawn")`
  with workers defined in `tests/_concurrent_helpers.py`. Lance is
  explicitly not fork-safe.

---

## Out of scope (please don't open PRs for these)

- Built-in OWL reasoner -- delegated to Open Ontologies / `owlrl` on
  purpose. See `ARCHITECTURE.md` for the rationale.
- Triple-store server / wire protocol -- embedded is the whole pitch.
- Distributed query execution.
- Custom SPARQL parser/optimizer/executor -- we reuse pyoxigraph and
  plan to swap in `rdf-fusion` later. We will not maintain our own.

---

## Reporting issues

- Bug reports: include Python version, pylance version, OS, the
  smallest reproducer you can manage, and the exact error/traceback.
- Feature requests: explain the use case first, the proposed API
  second. We say "no" fast to anything that doesn't trace back to
  the use cases listed in `REQUIREMENTS.md` -- that's a feature, not
  a bug, and it keeps the surface focused.

---

## License

By contributing, you agree your contributions are licensed under
[Apache 2.0](./LICENSE) (or the equivalent header in `pyproject.toml`).
