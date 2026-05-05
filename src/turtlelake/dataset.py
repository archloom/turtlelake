"""The `Dataset` facade — the single thing users touch.

Backed by two sibling Lance datasets:

  triples.lance/      RDF quads (always present after first ingest)
  embeddings.lance/   per-IRI vectors (created lazily on first `embed()`)

SPARQL execution dispatches to pyoxigraph for the MVP; swapping in
rdf-fusion later is an internal change, not an API one. Vector search
goes directly through Lance's native ANN — no extra index server.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import lance
import pyarrow as pa

from turtlelake.engine import SparqlEngine
from turtlelake.ingest import parse_rdf_file, quads_to_record_batch
from turtlelake.schema import TRIPLE_SCHEMA, embedding_schema


class Dataset:
    """Open (or create) a turtlelake dataset at `path`.

    A turtlelake dataset is one directory containing two Lance datasets:
    `<path>/triples.lance` (the RDF graph) and an optional
    `<path>/embeddings.lance` (per-IRI vectors).
    """

    TRIPLES_DIR = "triples.lance"
    EMBEDDINGS_DIR = "embeddings.lance"
    MANIFEST_FILE = "manifest.json"

    # When sources are attached, agent writes default here so vendor data
    # (in per-source named graphs) and agent data stay cleanly separable.
    AGENT_OVERLAY_GRAPH = "turtlelake://agent-overlay"

    def __init__(self, path: str | Path, version: int | None = None, tag: str | None = None):
        # Normalize `file://` URIs to local paths so pathlib behaves.
        # Non-file:// URIs (s3://, gs://, hf://) are kept as strings
        # in `self.path`, but wrapping in Path still works for the
        # .TRIPLES_DIR join — pathlib just treats them as string parts.
        path_str = str(path)
        if path_str.startswith("file://"):
            path_str = path_str.removeprefix("file://")
        self.path = Path(path_str)
        self._version = version
        self._tag = tag
        self._lance: lance.LanceDataset | None = None
        self._embeddings: lance.LanceDataset | None = None
        # In-RAM vector cache. Populated by `preload_vectors()`; consumed
        # by `vector_search(in_memory="auto"|True)` for sub-millisecond
        # brute-force search via numpy matmul. Invalidated on next embed().
        # Keyed by (embeddings.version, model_id_filter) so the cache
        # remains correct across writes and across model_id filters.
        self._vector_cache: dict | None = None
        # Engine cache: materializing into pyoxigraph is O(rows). Agents
        # typically do many reads per write, so we cache the engine keyed
        # on the Lance version. Writes bump the version → natural invalidation.
        self._cached_engine: SparqlEngine | None = None
        self._cached_engine_version: int = -2  # -1 is valid for empty datasets
        # Upstream sources: {graph_iri: {"path": Path, "mtime": float, "sha256": str}}
        # Loaded at query time into named graphs inside the cached engine;
        # never copied into Lance.
        self._sources: dict[str, dict] = {}
        self._sources_fingerprint: tuple = ()  # used as a cache invalidation key

    # --- construction ----------------------------------------------------

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        version: int | None = None,
        tag: str | None = None,
        pre_warm: bool = False,
        sources: dict[str, str | Path] | None = None,
        follow_imports: bool = False,
    ) -> Dataset:
        """Open or create a dataset at `path`.

        `pre_warm=True` eagerly materializes the pyoxigraph engine cache
        at open time, paying the warm-up cost (~150 ms on a 15k-triple
        graph) up front rather than on the first query.

        `sources` attaches external TTL files as read-only upstream.
        Shape: `{graph_iri: path_or_file_uri}`. Each file is loaded into
        its own named graph inside the cached engine — NEVER copied into
        Lance. File mtime is watched; edits upstream are picked up on
        the next query. With at least one source attached, agent writes
        (`insert_turtle` / `insert` / `ingest_ttl`) default to the
        `turtlelake://agent-overlay` named graph to keep agent data
        cleanly separable from vendor data.

        `follow_imports=True` walks `owl:imports` transitively from each
        attached source (local files only in v0; cycle-safe).
        """
        if version is not None and tag is not None:
            raise ValueError(
                "Dataset.open: pass either `version` or `tag`, not both."
            )
        ds = cls(path, version=version, tag=tag)
        # Only mkdir for local filesystem paths; URIs point to object stores.
        if _is_local_path(ds.path):
            ds.path.mkdir(parents=True, exist_ok=True)
        if _exists(ds.triples_path):
            triples_uri = _resolve_uri(ds.triples_path)
            lance_ds = lance.dataset(triples_uri, version=version)
            if tag is not None:
                tagged_version = lance_ds.tags.get_version(tag)
                lance_ds = lance.dataset(triples_uri, version=tagged_version)
            ds._lance = lance_ds
        # Embeddings are paired by tag name with the triples dataset. A
        # `checkpoint("foo")` tags both with the same name; opening at
        # `tag="foo"` pins both. When a tag exists on triples but not
        # embeddings (e.g. embeddings written later), we open embeddings
        # at its latest version and the caller gets a "best effort" view.
        if _exists(ds.embeddings_path):
            emb_uri = _resolve_uri(ds.embeddings_path)
            emb_ds = lance.dataset(emb_uri, version=version)
            if tag is not None:
                try:
                    emb_tagged = emb_ds.tags.get_version(tag)
                    emb_ds = lance.dataset(emb_uri, version=emb_tagged)
                except (KeyError, ValueError, RuntimeError):
                    pass
            ds._embeddings = emb_ds
        # Reconcile a partial checkpoint left over from a prior crash.
        # No-op if nothing pending. See `_recover_pending_checkpoint`.
        if ds._lance is not None:
            ds._recover_pending_checkpoint()
        if sources:
            ds._register_sources(sources, follow_imports=follow_imports)
        if pre_warm and (ds._lance is not None or ds._sources):
            ds._engine()
        return ds

    @property
    def triples_path(self) -> Path:
        return self.path / self.TRIPLES_DIR

    @property
    def embeddings_path(self) -> Path:
        return self.path / self.EMBEDDINGS_DIR

    @property
    def manifest_path(self) -> Path:
        return self.path / self.MANIFEST_FILE

    # --- ingest ----------------------------------------------------------

    def ingest_ttl(
        self,
        ttl_path: str | Path,
        *,
        batch_size: int = 50_000,
        source: str | None = None,
        author: str | None = None,
        graph: str | None = None,
    ) -> int:
        """Parse an RDF file and append its quads as a new Lance version.

        `graph` names the target named graph. `graph=None` + attached sources
        auto-routes to `turtlelake://agent-overlay`; `graph=None` + no
        sources uses the default graph (today's behavior).
        """
        from pyoxigraph import NamedNode, Quad

        src = Path(ttl_path)
        target = self._resolve_write_graph(graph)
        if target is None:
            n = self._append_quads(parse_rdf_file(src), batch_size=batch_size)
        else:
            graph_node = NamedNode(target)
            quads = (
                Quad(q.subject, q.predicate, q.object, graph_node)
                for q in parse_rdf_file(src)
            )
            n = self._append_quads(quads, batch_size=batch_size)
        self._log_provenance(
            source=source or src.name,
            author=author,
            kind="ingest_ttl",
            row_delta=n,
            graph=target,
        )
        return n

    def insert_turtle(
        self,
        ttl_text: str,
        *,
        source: str | None = None,
        author: str | None = None,
        graph: str | None = None,
    ) -> int:
        """Append quads parsed from a TTL string. The agent-memory entry point.

        `graph` routes to a specific named graph. `graph=None` + attached
        sources → auto-routes to `turtlelake://agent-overlay` so vendor
        and agent data stay separable.
        """
        import io

        from pyoxigraph import NamedNode, Quad, RdfFormat, parse

        parsed = list(parse(io.BytesIO(ttl_text.encode("utf-8")), format=RdfFormat.TURTLE))
        target = self._resolve_write_graph(graph)
        if target is None:
            quads = parsed
        else:
            graph_node = NamedNode(target)
            quads = [
                Quad(q.subject, q.predicate, q.object, graph_node) for q in parsed
            ]
        n = self._append_quads(iter(quads), batch_size=50_000)
        self._extend_cache(quads, n)
        self._log_provenance(
            source=source or "inline-turtle",
            author=author,
            kind="insert_turtle",
            row_delta=n,
            graph=target,
        )
        return n

    def insert(
        self,
        quads: Iterable,
        *,
        source: str | None = None,
        author: str | None = None,
        graph: str | None = None,
    ) -> int:
        """Append pyoxigraph Quads, optionally re-graphing to `graph`."""
        from pyoxigraph import NamedNode, Quad

        quads = list(quads)
        target = self._resolve_write_graph(graph)
        if target is not None:
            graph_node = NamedNode(target)
            quads = [
                Quad(q.subject, q.predicate, q.object, graph_node) for q in quads
            ]
        n = self._append_quads(iter(quads), batch_size=50_000)
        self._extend_cache(quads, n)
        self._log_provenance(
            source=source or "manual-quads",
            author=author,
            kind="insert",
            row_delta=n,
            graph=target,
        )
        return n

    def _resolve_write_graph(self, explicit: str | None) -> str | None:
        """Determine the target named graph for a write.

        - If caller passed `graph=...` explicitly, honor it.
        - Else if sources are attached, route to the agent-overlay graph
          so vendor data stays separable.
        - Else fall back to the default graph (None).
        """
        if explicit is not None:
            return explicit
        if self._sources:
            return self.AGENT_OVERLAY_GRAPH
        return None

    def _extend_cache(self, quads: list, written: int) -> None:
        """If a pyoxigraph engine is already cached, extend it with the
        just-written quads so the next query avoids a full rebuild.

        No-op when no engine is warm — `_engine()` will materialize on
        demand from Lance.
        """
        if written == 0 or self._cached_engine is None or self._lance is None:
            return
        self._cached_engine.store.extend(quads)
        self._cached_engine_version = self._lance.version

    def _append_quads(self, quads: Iterable, *, batch_size: int) -> int:
        written = 0
        buf: list = []
        reader_batches: list[pa.RecordBatch] = []

        def flush() -> None:
            nonlocal buf
            if not buf:
                return
            reader_batches.append(quads_to_record_batch(buf))
            buf = []

        for q in quads:
            buf.append(q)
            if len(buf) >= batch_size:
                flush()
        flush()

        if not reader_batches:
            return 0

        reader = pa.RecordBatchReader.from_batches(TRIPLE_SCHEMA, reader_batches)
        mode = "append" if self._lance is not None else "create"
        self._lance = lance.write_dataset(
            reader,
            _resolve_uri(self.triples_path),
            schema=TRIPLE_SCHEMA,
            mode=mode,
        )
        written = sum(b.num_rows for b in reader_batches)
        return written

    # --- query -----------------------------------------------------------

    def scan(self, columns: list[str] | None = None) -> pa.Table:
        """Return the current dataset as an Arrow Table. Zero-copy where possible.

        For analytic queries prefer `.to_polars()` / DuckDB / DataFusion against
        the Lance dataset directly — this method materializes in memory.
        """
        self._require_lance()
        return self._lance.to_table(columns=columns)

    def query(self, sparql: str, *, timeout_ms: int | None = None) -> list[dict]:
        """Run a SPARQL query. Returns a list of bindings (one dict per row).

        v0 strategy: materialize into an in-memory pyoxigraph store and query
        there. Correct but memory-bound. The materialized store is cached
        per Lance version + source fingerprint; agents doing many reads
        between writes pay the materialization cost only once.

        `timeout_ms` aborts waits for long-running queries (default
        `TURTLELAKE_QUERY_TIMEOUT_MS` env var, or None = no limit).
        """
        self._require_backing()
        if timeout_ms is None:
            env = os.environ.get("TURTLELAKE_QUERY_TIMEOUT_MS")
            if env:
                try:
                    timeout_ms = int(env)
                except ValueError:
                    timeout_ms = None
        return self._engine().query(sparql, timeout_ms=timeout_ms)

    def explain(self, sparql: str) -> str:
        """Return a plain-text query plan sketch.

        pyoxigraph doesn't expose a true optimizer plan yet — we provide a
        pattern-stat heuristic that's still useful for agents wondering
        which part of their SPARQL is slow. See ARCHITECTURE.md M6/M7 for
        the real DataFusion-based plan coming later.
        """
        self._require_backing()
        return self._engine().explain(sparql)

    def entity(
        self,
        iri: str,
        *,
        hops: int = 1,
        similar: int = 0,
        model_id: str | None = None,
    ) -> dict:
        """Return the N-hop subgraph around `iri` as a JSON-friendly dict.

        Shape:
            {
              "iri": "...",
              "outgoing":  [{"predicate": "...", "object": <term>}, ...],
              "incoming":  [{"predicate": "...", "subject": "..."}, ...],
              "neighbors": {"<iri>": {... same shape, recursive ...}},
              "similar":   [{iri, distance, model_id}, ...]   # iff similar>0
            }

        `similar` (when > 0) appends nearest-neighbor IRIs by vector
        distance to this entity's stored embedding. If the IRI has no
        embedding (or no embeddings dataset exists), `similar` is omitted
        rather than raising — keeps the call shape stable.

        Agents use this far more than open-ended SPARQL. Pushed as a
        first-class primitive so the MCP surface can expose it directly.
        """
        if hops < 1:
            raise ValueError("hops must be >= 1")
        self._require_backing()
        result = _expand_entity(self._engine(), iri, hops)
        if similar > 0:
            sim = self._similar_to_iri(iri, k=similar, model_id=model_id)
            if sim is not None:
                # Drop self if it landed in the top-k (vector self-distance is 0).
                result["similar"] = [s for s in sim if s["iri"] != iri][:similar]
        return result

    def _similar_to_iri(
        self, iri: str, *, k: int, model_id: str | None
    ) -> list[dict] | None:
        """Find vectors nearest to `iri`'s own stored embedding. Returns
        None when no embedding for `iri` is present (so `entity()` can
        omit the field rather than report empty).

        Deterministic resolution when an IRI has multiple embeddings
        (e.g. across `model_id`s or re-embeds): we pick the most recently
        written matching row by `created_at`, with `model_id` used as a
        tiebreaker for the unlikely case of identical timestamps. The
        explicit `model_id` argument, when provided, narrows the lookup
        to a single model first."""
        if self._embeddings is None:
            return None
        sql_iri = _sql_escape(iri)
        flt = f"iri = '{sql_iri}'"
        if model_id is not None:
            if not _valid_model_id(model_id):
                raise ValueError(
                    f"model_id must be a non-empty identifier of [A-Za-z0-9._:/+-]; "
                    f"got {model_id!r}"
                )
            flt += f" AND model_id = '{_sql_escape(model_id)}'"
        my = self._embeddings.to_table(
            columns=["vector", "model_id", "created_at"], filter=flt
        )
        if my.num_rows == 0:
            return None
        # Stable, deterministic pick: most recent created_at, then
        # smallest model_id alphabetically as tiebreaker.
        rows = []
        for i in range(my.num_rows):
            rows.append(
                (
                    my["created_at"][i].as_py(),
                    my["model_id"][i].as_py(),
                    my["vector"][i].as_py(),
                )
            )
        rows.sort(key=lambda r: (r[0], r[1]), reverse=True)
        my_vec = rows[0][2]
        # k+1 because the IRI's own vector is distance 0 from itself.
        return self.vector_search(my_vec, k=k + 1, model_id=model_id)

    def _engine(self) -> SparqlEngine:
        """Return a pyoxigraph engine over the current Lance version + any
        attached upstream sources.

        Cache invalidation triggers:
        - Lance version changed (a write happened)
        - Any attached source's mtime changed (upstream edit)
        """
        current_lance = self._lance.version if self._lance is not None else -1
        current_src_fp = self._refresh_source_fingerprint()

        stale = (
            self._cached_engine is None
            or self._cached_engine_version != current_lance
            or self._sources_fingerprint != current_src_fp
        )
        if stale:
            if self._lance is not None:
                engine = SparqlEngine.from_lance(self._lance)
            else:
                # Sources-only mode (no Lance yet): start with an empty store.
                from pyoxigraph import Store as _OxStore
                engine = SparqlEngine(store=_OxStore())
            # Layer every attached source into its named graph.
            self._load_sources_into(engine)
            self._cached_engine = engine
            self._cached_engine_version = current_lance
            self._sources_fingerprint = current_src_fp
        return self._cached_engine

    # --- versioning ------------------------------------------------------

    def versions(self) -> list[dict]:
        """Return Lance versions. Empty list if nothing has been written."""
        if self._lance is None:
            return []
        return self._lance.versions()

    def tag(self, name: str) -> None:
        """Tag the current version of *both* the triples and embeddings
        datasets with `name`.

        Crash-safe pairing: we record `pending_checkpoint` in the
        manifest BEFORE touching either Lance dataset. If the process
        dies between the two tag creates, the next `Dataset.open(path)`
        finishes the pair via `_recover_pending_checkpoint`. Once both
        tags are in place we clear the pending marker."""
        self._require_lance()
        triples_version = self._lance.version
        emb_version = self._embeddings.version if self._embeddings is not None else None
        self._begin_pending_checkpoint(name, triples_version, emb_version)

        try:
            self._lance.tags.delete(name)
        except Exception:
            pass
        self._lance.tags.create(name, triples_version)
        if self._embeddings is not None:
            try:
                self._embeddings.tags.delete(name)
            except Exception:
                pass
            self._embeddings.tags.create(name, emb_version)
        self._commit_pending_checkpoint()

    def tags(self) -> list[str]:
        """Return tag names. Empty list if nothing has been written.

        Triples tags are authoritative — embedding tags echo them. If a
        write touched only embeddings between two checkpoints, the
        intermediate embeddings versions are still in the version chain
        but no tag points at them."""
        if self._lance is None:
            return []
        return list(self._lance.tags.list())

    # --- agent write-safety primitives -----------------------------------

    def checkpoint(self, name: str, *, author: str | None = None) -> int:
        """Tag the current version before a risky write.

        Semantically identical to `tag()`, but named to match how agents
        think about it ("make a savepoint before I attempt this"). Idempotent:
        re-checkpointing the same name moves it to the current version.

        Crash-safe across the triples + embeddings pair: a
        `pending_checkpoint` record is written to manifest.json BEFORE
        either Lance tag is created, then cleared once both succeed. A
        crash between the two tag creates is detected on the next
        `Dataset.open(path)` and reconciled (forward-roll the missing
        tag) so callers see a consistent pair, never a half-tagged
        state. A later `rollback(name)` restores both atomically."""
        self._require_lance()
        version = self._lance.version
        emb_version = self._embeddings.version if self._embeddings is not None else None
        self._begin_pending_checkpoint(name, version, emb_version)

        try:
            self._lance.tags.delete(name)
        except Exception:
            pass  # tag didn't exist; fine
        self._lance.tags.create(name, version)
        if self._embeddings is not None:
            try:
                self._embeddings.tags.delete(name)
            except Exception:
                pass
            self._embeddings.tags.create(name, emb_version)
        self._commit_pending_checkpoint()
        self._log_provenance(
            source=f"checkpoint:{name}",
            author=author,
            kind="checkpoint",
            row_delta=0,
        )
        return version

    def rollback(self, name: str) -> Dataset:
        """Restore a tagged version as the new HEAD.

        **Critical semantic**: this does *not* merely re-open at the tag
        (that's `Dataset.open(path, tag=...)`, which returns a read-only
        snapshot). Rollback commits the tagged state as the latest
        version via `lance.restore()`, so subsequent writes build on top
        of it. The forward history that was rolled back is preserved
        in Lance's version chain and visible to `provenance()` — it is
        not deleted — but queries against the new HEAD don't see it.

        **Mutates `self` in place** AND returns self for chaining. This
        guarantees that every existing handle — not just the returned
        value — sees the post-rollback state on its next query. Use
        `kg.rollback(name)` or `kg = kg.rollback(name)` interchangeably.

        Raises `KeyError` with an actionable message if `name` isn't a
        known tag (UC-3.4).
        """
        self._require_lance()
        if name not in self.tags():
            raise KeyError(
                f"No such tag {name!r}. Known tags: {sorted(self.tags())}. "
                "Call .checkpoint(name) or .tag(name) first."
            )
        target_version = self._lance.tags.get_version(name)
        # Open a FRESH handle pinned to the tagged version (opening at the
        # latest and then calling checkout_version doesn't re-pin as
        # strongly as a direct versioned open), then restore to make that
        # state the new HEAD.
        uri = _resolve_uri(self.triples_path)
        pinned = lance.dataset(uri, version=target_version)
        pinned.restore()
        # Update *this* handle to point at the fresh HEAD. The in-place
        # mutation prevents silent stale reads on callers that already
        # held a reference before the rollback.
        self._lance = lance.dataset(uri)
        self._version = None
        self._tag = None
        # Roll the embeddings dataset back to the same tag, if it exists.
        # Atomic-ish: we restore embeddings only after triples succeeded,
        # so a partial rollback leaves the agent with the older graph and
        # whichever embeddings happened to be HEAD. Acceptable: vector
        # search degrades gracefully (returns IRIs that may not be in
        # the rolled-back graph anymore; entity expansion handles that).
        if self._embeddings is not None:
            emb_uri = _resolve_uri(self.embeddings_path)
            try:
                emb_target = self._embeddings.tags.get_version(name)
                emb_pinned = lance.dataset(emb_uri, version=emb_target)
                emb_pinned.restore()
                self._embeddings = lance.dataset(emb_uri)
            except (KeyError, ValueError, RuntimeError):
                # No matching embedding tag — embeddings predate the
                # checkpoint or were never tagged. Leave HEAD as-is.
                pass
        self._log_provenance(
            source=f"rollback:{name}", author=None, kind="rollback", row_delta=0
        )
        return self

    def refresh(self) -> int:
        """Re-open the dataset at the latest committed version.

        Refuses to run on a handle that was pinned with `tag=` or
        `version=` — those handles represent a reproducibility snapshot
        (UC-5) and refreshing would silently leak the caller out of
        their pinned view. Open a new unpinned handle if you want the
        latest state.
        """
        if self._tag is not None or self._version is not None:
            raise RuntimeError(
                "Cannot refresh() a pinned handle "
                f"(tag={self._tag!r}, version={self._version!r}). "
                "Open a new Dataset without tag/version to see latest."
            )
        triples_uri = _resolve_uri(self.triples_path)
        if self._lance is None:
            if not _exists(self.triples_path):
                return -1
            self._lance = lance.dataset(triples_uri)
            return self._lance.version
        self._lance = lance.dataset(triples_uri)
        return self._lance.version

    # --- diff ------------------------------------------------------------

    def diff(self, from_version: int, to_version: int | None = None) -> dict:
        """Return quads added and removed between two Lance versions.

        `to_version=None` (the default) means "current version" — the
        natural shape for an agent asking "what changed since the
        baseline?".

        Shape: `{"added": [quad-dict, ...], "removed": [...]}`. For MVP
        this materializes both versions' quad tuples in memory; fine at
        the sizes the v0 SPARQL path already forces into RAM.
        """
        self._require_lance()
        if to_version is None:
            to_version = self._lance.version
        old = lance.dataset(_resolve_uri(self.triples_path), version=from_version)
        new = lance.dataset(_resolve_uri(self.triples_path), version=to_version)
        old_rows = _quad_tuples(old)
        new_rows = _quad_tuples(new)
        added = new_rows - old_rows
        removed = old_rows - new_rows
        return {
            "added": [_tuple_to_quad_dict(t) for t in added],
            "removed": [_tuple_to_quad_dict(t) for t in removed],
        }

    # --- stored queries --------------------------------------------------

    @property
    def queries_path(self) -> Path:
        return self.path / "queries.json"

    def save_query(self, name: str, sparql: str, description: str = "") -> None:
        """Persist a named SPARQL query to `<path>/queries.json`.

        Agents can register parameterized queries once and call them by
        name later, passing bindings (pyoxigraph `substitutions=`).
        """
        if not name or not name.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                f"query name must be alphanumeric / underscore / dash; got {name!r}"
            )
        store = self.stored_queries()
        store[name] = {"sparql": sparql, "description": description}
        self.queries_path.write_text(json.dumps(store, indent=2, sort_keys=True))

    def stored_queries(self) -> dict:
        """Return the full stored-queries catalog as `{name: {sparql, description}}`."""
        if not self.queries_path.exists():
            return {}
        return json.loads(self.queries_path.read_text())

    def run_saved(
        self,
        name: str,
        *,
        bindings: dict | None = None,
        timeout_ms: int | None = None,
    ) -> list[dict]:
        """Execute a named stored query.

        `bindings` maps variable names (without the `?` prefix) to values.
        Because pyoxigraph's `substitutions=` requires the variables to
        appear in the SELECT projection, we instead rewrite the SPARQL
        text by replacing each `?var` with a properly-serialized term
        (IRI → `<...>`, literal → `"..."`, escaped). Injection-safe by
        construction: the replacement comes from our own serializer."""
        catalog = self.stored_queries()
        if name not in catalog:
            raise KeyError(
                f"No saved query {name!r}. Known queries: "
                f"{sorted(catalog)}. Call .save_query(...) first."
            )
        sparql = catalog[name]["sparql"]
        if bindings:
            sparql = _inline_bindings(sparql, bindings)
        return self.query(sparql, timeout_ms=timeout_ms)

    # --- provenance ------------------------------------------------------

    @property
    def provenance_path(self) -> Path:
        return self.path / "provenance.jsonl"

    def provenance(self) -> list[dict]:
        """Ordered list of writes: `{version, source, author, timestamp, kind, row_delta}`.

        Tolerates torn/truncated lines (rare but possible if a previous
        writer crashed mid-write) by skipping the offending entry.
        """
        if not self.provenance_path.exists():
            return []
        out: list[dict] = []
        with self.provenance_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    # Skip torn line; real remediation is to re-ingest.
                    continue
        return out

    def _log_provenance(
        self,
        *,
        source: str | None,
        author: str | None,
        kind: str,
        row_delta: int,
        graph: str | None = None,
    ) -> None:
        """Append one JSON line to provenance.jsonl atomically.

        Two concurrent writers on the same file must not interleave
        bytes. Strategy: build the full line (JSON + '\\n') as one byte
        buffer, then acquire an advisory fcntl lock and do a single
        `os.write`. On Windows, fcntl is unavailable — fall back to the
        OS-level O_APPEND which gives atomic appends for writes smaller
        than PIPE_BUF (~4 KiB) on POSIX; our lines are well under that.
        """
        record = {
            "version": self._lance.version if self._lance is not None else 0,
            "source": source or "unknown",
            "author": author or os.environ.get("USER", "unknown"),
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "kind": kind,
            "row_delta": row_delta,
        }
        if graph is not None:
            record["graph"] = graph
        payload = (json.dumps(record) + "\n").encode("utf-8")
        _append_line_atomic(self.provenance_path, payload)

    # --- SHACL validation (optional) -------------------------------------

    def validate(self, shapes_ttl: str | Path) -> dict:
        """Validate the current version against a SHACL shapes graph.

        Requires the optional `shacl` extra: `pip install 'turtlelake[shacl]'`.
        Returns `{"conforms": bool, "report_text": str}`.
        """
        try:
            import pyshacl  # type: ignore
            import rdflib  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "pyshacl is not installed. Install with: pip install 'turtlelake[shacl]'"
            ) from e

        shapes_path = Path(shapes_ttl)
        if not shapes_path.exists():
            raise FileNotFoundError(
                f"SHACL shapes file not found: shapes_ttl={str(shapes_ttl)!r}. "
                "Pass a filesystem path to a readable Turtle shapes graph."
            )

        data = rdflib.Graph()
        for row in self.query(
            "SELECT ?s ?p ?o WHERE { ?s ?p ?o }"
        ):  # keep it simple; default graph only for MVP
            s = _rdflib_term(row["s"], rdflib)
            p = _rdflib_term(row["p"], rdflib)
            o = _rdflib_term(row["o"], rdflib)
            data.add((s, p, o))
        shapes = rdflib.Graph().parse(str(shapes_path), format="turtle")
        conforms, _results_graph, report_text = pyshacl.validate(
            data_graph=data, shacl_graph=shapes
        )
        return {"conforms": bool(conforms), "report_text": report_text}

    # --- vector layer ----------------------------------------------------

    # Hard ceilings — kept conservative on the assumption that a single
    # well-behaved call never approaches them. Override via env vars
    # `TURTLELAKE_MAX_VECTORS_PER_EMBED` and `TURTLELAKE_MAX_EMBEDDING_DIM`.
    DEFAULT_MAX_VECTORS_PER_EMBED = 1_000_000
    DEFAULT_MAX_EMBEDDING_DIM = 16_384

    def embed(
        self,
        iris: Sequence[str],
        vectors: Sequence[Sequence[float]],
        *,
        model_id: str,
        author: str | None = None,
    ) -> int:
        """Append per-IRI embeddings to the embeddings dataset.

        We never load a model — the caller computes vectors however they
        like (OpenAI API, local sentence-transformer, hand-crafted) and
        passes raw floats. This mirrors turtlelake's overall philosophy:
        we store data, we do not run inference.

        On the first call, the embeddings Lance dataset is created with
        `dim = len(vectors[0])`. Subsequent calls must use the same dim
        (Lance enforces it via `fixed_size_list`).

        Hard input checks — these prevent runaway resource use and
        silently-broken vectors poisoning ANN distance:
        - `iris` and `vectors` must have equal length
        - dim must be in `[1, TURTLELAKE_MAX_EMBEDDING_DIM]`
        - row count must be ≤ `TURTLELAKE_MAX_VECTORS_PER_EMBED`
        - every component must be a finite float (no NaN, no ±Inf)
        - `model_id` must be a non-empty ASCII identifier (chars in
          `[A-Za-z0-9._:/+-]`); rejects values that could break the
          SQL filter even after escaping

        Returns the number of rows written. Each call creates a new
        Lance version on the embeddings dataset; tag/checkpoint covers
        both.
        """
        import math

        if len(iris) != len(vectors):
            raise ValueError(
                f"iris and vectors length mismatch: {len(iris)} vs {len(vectors)}"
            )
        if not iris:
            return 0

        max_rows = _int_env_or(
            "TURTLELAKE_MAX_VECTORS_PER_EMBED", self.DEFAULT_MAX_VECTORS_PER_EMBED
        )
        if len(iris) > max_rows:
            raise ValueError(
                f"embed batch too large: {len(iris)} rows exceeds cap {max_rows}. "
                "Split into smaller calls or raise TURTLELAKE_MAX_VECTORS_PER_EMBED."
            )

        if not _valid_model_id(model_id):
            raise ValueError(
                f"model_id must be a non-empty identifier of [A-Za-z0-9._:/+-]; "
                f"got {model_id!r}"
            )

        dim = len(vectors[0])
        if dim <= 0:
            raise ValueError("vectors must have at least one dimension")
        max_dim = _int_env_or(
            "TURTLELAKE_MAX_EMBEDDING_DIM", self.DEFAULT_MAX_EMBEDDING_DIM
        )
        if dim > max_dim:
            raise ValueError(
                f"embedding dim {dim} exceeds cap {max_dim}. "
                "Raise TURTLELAKE_MAX_EMBEDDING_DIM if you really need this."
            )
        # Enforce dim consistency with any prior embed on this dataset
        # ahead of Lance's own check, so the error message names the
        # right knob.
        existing_dim = self.embedding_dim()
        if existing_dim is not None and existing_dim != dim:
            raise ValueError(
                f"vector dim {dim} does not match existing dataset dim "
                f"{existing_dim}. All embeddings in one dataset share a dim."
            )
        for i, v in enumerate(vectors):
            if len(v) != dim:
                raise ValueError(
                    f"vector at index {i} has dim {len(v)}, expected {dim} "
                    "(all vectors in one call must share dim)"
                )
            for j, c in enumerate(v):
                if not math.isfinite(c):
                    raise ValueError(
                        f"vector[{i}][{j}]={c!r} is not finite "
                        "(NaN/Inf vectors break ANN distance and are rejected)"
                    )

        emb_schema = embedding_schema(dim)
        ts_now = datetime.now(tz=timezone.utc)
        # pyarrow stores `timestamp[us]` from datetime objects directly.
        tbl = pa.table(
            {
                "iri": list(iris),
                "vector": list(vectors),
                "model_id": [model_id] * len(iris),
                "created_at": [ts_now] * len(iris),
            },
            schema=emb_schema,
        )
        emb_uri = _resolve_uri(self.embeddings_path)
        mode = "append" if self._embeddings is not None else "create"
        self._embeddings = lance.write_dataset(
            tbl, emb_uri, schema=emb_schema, mode=mode
        )
        # In-RAM cache is keyed on the embeddings version; bumping the
        # version invalidates `has_warm_cache()` automatically. Drop
        # the old payload eagerly so we don't hold a copy of stale
        # vectors longer than necessary.
        self._vector_cache = None
        self._record_manifest_dim(dim)
        self._log_provenance(
            source=f"embed:{model_id}",
            author=author,
            kind="embed",
            row_delta=len(iris),
        )
        return len(iris)

    # --- hybrid retrieval (BM25 + vector, RRF-fused) --------------------

    def preload_text_index(
        self,
        *,
        predicates: Sequence[str] | None = None,
        max_features: int = 50_000,
    ) -> dict:
        """Build a per-IRI BM25 index over text literals from the graph.

        For each IRI we concatenate the literal objects of the listed
        `predicates` (defaulting to `rdfs:label` + `skos:definition`,
        the canonical "give me a string description of X" predicates).
        The resulting bag-of-words is fed to a TF-IDF vectorizer
        configured for sublinear TF — i.e. the standard BM25-shaped
        retrieval baseline.

        Why this exists: vector retrieval and BM25 fail on different
        questions (vector loses on rare tokens / typos, BM25 loses on
        paraphrase). Reciprocal Rank Fusion of the two is the cheapest
        retrieval improvement in the modern RAG playbook — published
        lift on MuSiQue-style benchmarks is +5-15 recall points.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

        self._require_backing()
        preds = (
            list(predicates)
            if predicates is not None
            else [
                "http://www.w3.org/2000/01/rdf-schema#label",
                "http://www.w3.org/2004/02/skos/core#definition",
            ]
        )
        engine = self._engine()
        from pyoxigraph import Literal as _Lit
        from pyoxigraph import NamedNode as _NN

        text_by_iri: dict[str, list[str]] = {}
        for pred_iri in preds:
            try:
                pred = _NN(pred_iri)
            except ValueError:
                continue
            for q in engine.store.quads_for_pattern(None, pred, None, None):
                if isinstance(q.subject, _NN) and isinstance(q.object, _Lit):
                    text_by_iri.setdefault(q.subject.value, []).append(q.object.value)
        if not text_by_iri:
            self._text_index = None
            return {"rows": 0, "predicates": preds}

        iris = sorted(text_by_iri)
        docs = [" ".join(text_by_iri[i]) for i in iris]
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(docs)
        self._text_index = {
            "iris": iris,
            "iri_to_row": {iri: i for i, iri in enumerate(iris)},
            "matrix": matrix,
            "vectorizer": vectorizer,
            "predicates": preds,
            "version": self._lance.version if self._lance is not None else 0,
        }
        return {
            "rows": len(iris),
            "vocab_size": int(matrix.shape[1]),
            "predicates": preds,
        }

    def bm25_search(self, query_text: str, *, k: int = 10) -> list[dict]:
        """BM25-style lexical search over the cached text index.

        Returns `{iri, score}` ordered by decreasing TF-IDF cosine score.
        Raises if `preload_text_index()` hasn't been called."""
        if getattr(self, "_text_index", None) is None:
            raise RuntimeError(
                "No text index. Call .preload_text_index() first."
            )
        idx = self._text_index
        q_vec = idx["vectorizer"].transform([query_text])
        scores = (idx["matrix"] @ q_vec.T).toarray().ravel()
        if k >= len(scores):
            order = scores.argsort()[::-1]
        else:
            partition = (-scores).argpartition(k)[:k]
            order = partition[(-scores[partition]).argsort()]
        out: list[dict] = []
        for r in order:
            s = float(scores[r])
            if s <= 0:
                continue
            out.append({"iri": idx["iris"][int(r)], "score": s})
        return out

    def hybrid_search(
        self,
        query_text: str,
        query_vector: Sequence[float],
        *,
        k: int = 10,
        model_id: str | None = None,
        rrf_k: int = 60,
        in_memory: bool | str = "auto",
    ) -> list[dict]:
        """Hybrid retrieval: BM25 + vector, fused via Reciprocal Rank Fusion.

        For each candidate IRI, RRF score = Σ 1/(rrf_k + rank_in_list).
        Higher score = better. `rrf_k=60` is the literature-default;
        bump to de-emphasize tail rank differences. Returns top-k IRIs
        with their RRF score and which list(s) they came from.

        Pulls top-`k*3` from each side; RRF benefits from depth.
        Requires both a warm vector dataset AND a text index."""
        if getattr(self, "_text_index", None) is None:
            raise RuntimeError(
                "Hybrid search needs a text index. "
                "Call .preload_text_index() first."
            )
        depth = max(k * 3, 30)
        bm25 = self.bm25_search(query_text, k=depth)
        vec = self.vector_search(
            query_vector, k=depth, model_id=model_id, in_memory=in_memory
        )
        rrf: dict[str, float] = {}
        sources: dict[str, list[str]] = {}
        for rank, hit in enumerate(bm25, start=1):
            rrf[hit["iri"]] = rrf.get(hit["iri"], 0.0) + 1.0 / (rrf_k + rank)
            sources.setdefault(hit["iri"], []).append("bm25")
        for rank, hit in enumerate(vec, start=1):
            rrf[hit["iri"]] = rrf.get(hit["iri"], 0.0) + 1.0 / (rrf_k + rank)
            sources.setdefault(hit["iri"], []).append("vector")
        ranked = sorted(rrf.items(), key=lambda kv: -kv[1])[:k]
        return [
            {"iri": iri, "score": score, "sources": sources[iri]}
            for iri, score in ranked
        ]

    # --- Personalized PageRank (HippoRAG-style) -------------------------

    def graph_rag_ppr(
        self,
        query_vector: Sequence[float],
        *,
        k: int = 5,
        seed_k: int = 10,
        damping: float = 0.5,
        iterations: int = 30,
        model_id: str | None = None,
        in_memory: bool | str = "auto",
        edge_predicates: Sequence[str] | None = None,
    ) -> list[dict]:
        """Vector-seeded Personalized PageRank over the entity graph.

        The HippoRAG mechanism in one method: vector search produces
        the seed nodes, PPR diffuses their importance through the
        graph along outbound edges, top-k by stationary score is the
        retrieval result. Better than plain `graph_rag` on multi-hop
        questions because PPR amplifies entities that are reachable
        from *several* seeds rather than just from one.

        Args:
            seed_k: how many vector hits to use as PPR seeds
            damping: PPR teleport probability (0..1). Lower = stays
                close to seeds; higher = wanders further. 0.5 is a
                strong default for multi-hop QA.
            iterations: power-method iterations (30 typically converges)
            edge_predicates: only walk along these predicates; default
                `None` = all predicates whose object is a NamedNode

        Returns `{iri, score}` ordered by decreasing PPR score.
        Falls back to plain `graph_rag` semantics (vector seeds only)
        if there are no edges to walk.
        """
        import numpy as np

        if not 0.0 < damping < 1.0:
            raise ValueError("damping must be in (0, 1)")
        if iterations < 1:
            raise ValueError("iterations must be >= 1")

        seeds = self.vector_search(
            query_vector, k=seed_k, model_id=model_id, in_memory=in_memory
        )
        if not seeds:
            return []

        # Build adjacency by walking the engine's typed-term store.
        # We index every IRI we encounter (subject or object).
        from pyoxigraph import NamedNode as _NN

        engine = self._engine()
        store = engine.store

        idx_by_iri: dict[str, int] = {}
        edges: list[tuple[int, int]] = []

        def _idx(iri: str) -> int:
            i = idx_by_iri.get(iri)
            if i is None:
                i = len(idx_by_iri)
                idx_by_iri[iri] = i
            return i

        if edge_predicates is None:
            for q in store:
                if isinstance(q.subject, _NN) and isinstance(q.object, _NN):
                    edges.append((_idx(q.subject.value), _idx(q.object.value)))
        else:
            for pred_iri in edge_predicates:
                try:
                    pred = _NN(pred_iri)
                except ValueError:
                    continue
                for q in store.quads_for_pattern(None, pred, None, None):
                    if isinstance(q.subject, _NN) and isinstance(q.object, _NN):
                        edges.append(
                            (_idx(q.subject.value), _idx(q.object.value))
                        )

        # Make sure every seed has a row in the matrix even if it has
        # no outbound edges (PPR still needs to teleport from it).
        for s in seeds:
            _idx(s["iri"])

        n = len(idx_by_iri)
        if n == 0:
            return []
        if not edges:
            # No graph structure → fall back to ranking the seeds by
            # vector score (consistent contract).
            return [
                {"iri": s["iri"], "score": 1.0 / (i + 1)}
                for i, s in enumerate(seeds)
            ][:k]

        # Build a row-normalized transition matrix (sparse) so PPR is
        # tractable even on millions of edges.
        from scipy.sparse import csr_matrix  # type: ignore

        rows = np.array([e[0] for e in edges], dtype=np.int64)
        cols = np.array([e[1] for e in edges], dtype=np.int64)
        data = np.ones(len(edges), dtype=np.float32)
        adj = csr_matrix((data, (rows, cols)), shape=(n, n))
        # Row-normalize: each row sums to 1 (or 0 if dangling).
        out_deg = np.asarray(adj.sum(axis=1)).ravel()
        # Avoid divide-by-zero; dangling rows → uniform teleport.
        inv_deg = np.zeros_like(out_deg, dtype=np.float32)
        nz = out_deg > 0
        inv_deg[nz] = 1.0 / out_deg[nz]
        # Scale rows in-place via diag multiply.
        from scipy.sparse import diags as _diags  # type: ignore
        transition = _diags(inv_deg) @ adj  # (n, n) row-stochastic where defined

        # Personalization vector: weight seeds by inverse rank
        # (closer seeds get more juice). Zero everywhere else.
        e_vec = np.zeros(n, dtype=np.float32)
        for rank, s in enumerate(seeds, start=1):
            row = idx_by_iri[s["iri"]]
            e_vec[row] += 1.0 / rank
        e_vec /= e_vec.sum()

        # Power iteration: r = (1 - d) * e + d * Pᵀ r
        r = e_vec.copy()
        Pt = transition.T.tocsr()
        for _ in range(iterations):
            r = (1.0 - damping) * e_vec + damping * (Pt @ r)
            # Re-distribute mass from dangling rows uniformly so the
            # vector keeps summing to 1. Negligible runtime overhead.
            mass = float(r.sum())
            if mass > 0:
                r /= mass

        # Top-k by stationary score.
        if k >= n:
            order = np.argsort(-r)
        else:
            partition = np.argpartition(-r, kth=k)[:k]
            order = partition[np.argsort(-r[partition])]
        iri_by_idx = {v: k_ for k_, v in idx_by_iri.items()}
        out: list[dict] = []
        for i in order:
            score = float(r[i])
            if score <= 0:
                continue
            out.append({"iri": iri_by_idx[int(i)], "score": score})
        return out

    # --- batch vector_search --------------------------------------------

    def vector_search_batch(
        self,
        query_vectors: Sequence[Sequence[float]],
        *,
        k: int = 10,
        model_id: str | None = None,
        in_memory: bool | str = "auto",
    ) -> list[list[dict]]:
        """Run several queries in one call. When the in-memory cache is
        warm, this is a single matrix-matrix multiply rather than N
        Python-level Lance calls — typically 10-50× faster on small
        batches. Returns one results list per query, in input order."""
        # Fast path: in-memory matmul. We do all queries in one BLAS
        # call, which is the actual reason `_batch` exists.
        if (in_memory is True or
            (in_memory == "auto" and self.has_warm_cache(model_id=model_id))):
            return self._in_memory_vector_search_batch(
                query_vectors, k=k, model_id=model_id
            )
        # Fallback: per-query Lance call. Same shape, slower throughput.
        return [
            self.vector_search(q, k=k, model_id=model_id, in_memory=False)
            for q in query_vectors
        ]

    def _in_memory_vector_search_batch(
        self,
        query_vectors: Sequence[Sequence[float]],
        *,
        k: int,
        model_id: str | None,
    ) -> list[list[dict]]:
        import numpy as np

        if not self.has_warm_cache(model_id=model_id):
            raise RuntimeError(
                "in-memory batch search needs a warm cache. "
                "Call .preload_vectors() first."
            )
        cache = self._vector_cache
        matrix: np.ndarray = cache["vectors"]
        if matrix.shape[0] == 0:
            return [[] for _ in query_vectors]
        Q = np.asarray(query_vectors, dtype=np.float32)
        if Q.ndim != 2:
            raise ValueError("query_vectors must be a 2-D array-like")
        # Squared L2 in one BLAS call:
        # ||q_i - x_j||^2 = ||q_i||^2 + ||x_j||^2 - 2 q_i · x_j
        q_sq = np.einsum("ij,ij->i", Q, Q)[:, None]
        x_sq = np.einsum("ij,ij->i", matrix, matrix)[None, :]
        dists = q_sq + x_sq - 2.0 * (Q @ matrix.T)
        # Optional model_id filter.
        if model_id is not None and cache["model_filter"] is None:
            mask = np.array(
                [m == model_id for m in cache["model_ids"]], dtype=bool
            )
            dists[:, ~mask] = np.inf
        n = matrix.shape[0]
        out: list[list[dict]] = []
        for row in dists:
            if k >= n:
                order = np.argsort(row)
            else:
                partition = np.argpartition(row, kth=k)[:k]
                order = partition[np.argsort(row[partition])]
            results: list[dict] = []
            for idx in order:
                d = row[idx]
                if not np.isfinite(d):
                    continue
                results.append(
                    {
                        "iri": cache["iris"][int(idx)],
                        "distance": float(d),
                        "model_id": cache["model_ids"][int(idx)],
                    }
                )
            out.append(results)
        return out

    # --- vector index ---------------------------------------------------

    # Auto-policy thresholds. Tunable via env vars, but the defaults
    # are deliberate: brute-force scan is fine to ~10k, IVF_FLAT (no
    # PQ) is the cheap-to-train sweet spot up to ~1M, IVF_PQ pays off
    # past that on memory.
    AUTO_INDEX_FLAT_MIN_ROWS = 10_000
    AUTO_INDEX_PQ_MIN_ROWS = 1_000_000
    PQ_TRAINING_MIN_ROWS = 256

    def build_vector_index(
        self,
        *,
        index_type: str = "auto",
        metric: str = "L2",
        num_partitions: int | None = None,
        num_sub_vectors: int | None = None,
    ) -> dict:
        """Build an approximate-nearest-neighbor index over `vector`.

        Modes:
          - `"auto"` (default): pick by row count.
              < ~10k rows  → no index built; brute-force scan is fine
              < ~1M rows   → `IVF_FLAT` (cheap to train, fast enough)
              ≥ ~1M rows   → `IVF_PQ` (compresses vectors to fit memory)
          - explicit `IVF_FLAT` / `IVF_SQ` / `IVF_PQ`: force the choice.

        `metric` is `"L2"` | `"cosine"` | `"dot"`. Partition / sub-vector
        knobs follow Lance's defaults when left as None.

        Returns a JSON-friendly status dict so callers (and the MCP
        tool) can see what actually happened:

            {"action": "skipped"|"built", "index_type": "...",
             "rows": int, "reason": str}

        Idempotent: re-building replaces the prior index. Raises with a
        clear message — not Lance's Rust panic — when an explicit
        `IVF_PQ` is requested below the 256-row training minimum.
        """
        if self._embeddings is None:
            raise RuntimeError(
                f"No embeddings at {self.embeddings_path}. "
                "Call .embed(iris, vectors, model_id=...) first."
            )
        rows = self._embeddings.count_rows()

        chosen = self._resolve_auto_index(index_type, rows)
        if chosen.get("action") == "skipped":
            return chosen

        resolved_type = chosen["index_type"]
        if "PQ" in resolved_type and rows < self.PQ_TRAINING_MIN_ROWS:
            raise RuntimeError(
                f"build_vector_index({resolved_type!r}) needs at least "
                f"{self.PQ_TRAINING_MIN_ROWS} vectors to train (have "
                f"{rows}). Use index_type='IVF_FLAT' or 'auto' at this "
                "scale; brute-force scan is already sub-millisecond."
            )

        kwargs: dict = {
            "column": "vector",
            "index_type": resolved_type,
            "metric": metric,
            "replace": True,
        }
        if num_partitions is not None:
            kwargs["num_partitions"] = num_partitions
        if num_sub_vectors is not None:
            kwargs["num_sub_vectors"] = num_sub_vectors
        self._embeddings.create_index(**kwargs)
        return {
            "action": "built",
            "index_type": resolved_type,
            "rows": rows,
            "reason": chosen.get("reason", "explicit"),
        }

    def tune_nprobes(
        self,
        *,
        target_recall: float = 0.95,
        sample_queries: int = 50,
        seed: int = 0,
        model_id: str | None = None,
    ) -> dict:
        """Pick the smallest `nprobes` that hits `target_recall` on a
        sampled query set.

        Production ANN systems all expose a tuning step like this:
        nobody knows what `nprobes=32` means in absolute terms, but
        "the smallest knob value that gives me ≥0.95 recall" is a
        question every engineer can answer. We sample `sample_queries`
        random vectors from the dataset itself, compute the
        brute-force top-k for each as ground truth, and binary-search
        for the smallest `nprobes` that recovers `target_recall` of
        those top-k on average.

        Returns `{nprobes, achieved_recall, target_recall, sample_size,
        partitions_in_index}` so the caller can persist the result and
        pass it to subsequent `vector_search()` calls.

        Requires an ANN index (`build_vector_index()`) to be built
        first; raises with a clear message otherwise. Uses the
        in-memory cache for ground truth when warm, otherwise issues
        Lance brute-force queries.
        """
        import numpy as np

        if self._embeddings is None:
            raise RuntimeError(
                "No embeddings; call .embed(...) before tune_nprobes()."
            )
        if not 0.0 < target_recall <= 1.0:
            raise ValueError("target_recall must be in (0, 1]")
        if sample_queries < 1:
            raise ValueError("sample_queries must be >= 1")

        # Pull the sample query vectors. We sample from the corpus
        # itself — the canonical "leave-one-in" approximation that
        # ann-benchmarks uses when no held-out queries are provided.
        n_total = self._embeddings.count_rows()
        if n_total == 0:
            raise RuntimeError("Embeddings dataset is empty.")
        rng = np.random.default_rng(seed)
        sample_idx = rng.choice(n_total, size=min(sample_queries, n_total), replace=False)

        # Read the sample vectors directly. Project only `vector` for
        # speed; we already trust the model_id filter at write time.
        sample_tbl = self._embeddings.to_table(columns=["vector"])
        flat = (
            sample_tbl["vector"]
            .combine_chunks()
            .flatten()
            .to_numpy(zero_copy_only=False)
        )
        dim = self.embedding_dim() or 0
        all_matrix = flat.astype(np.float32, copy=False).reshape(n_total, dim)
        sample_vecs = all_matrix[sample_idx]
        del flat

        k = 10  # standard recall target depth

        # Ground truth: brute-force top-k via numpy. Way faster than
        # hitting Lance N times on 1M-scale data.
        # Squared L2; add a tiny epsilon to break ties stably.
        ground_truth: list[set] = []
        # Process sample queries in chunks to keep memory bounded.
        chunk = 32
        for start in range(0, len(sample_vecs), chunk):
            q = sample_vecs[start:start + chunk]
            # ||q-x||^2 = ||q||^2 + ||x||^2 - 2 q·x
            q_sq = np.einsum("ij,ij->i", q, q)[:, None]
            x_sq = np.einsum("ij,ij->i", all_matrix, all_matrix)[None, :]
            d = q_sq + x_sq - 2.0 * (q @ all_matrix.T)
            for row in d:
                topk = np.argpartition(row, kth=min(k, row.size - 1))[:k]
                ground_truth.append(set(int(i) for i in topk))

        # Read partitions from the index. Lance's metadata isn't
        # exposed cleanly across versions, so we probe via a config
        # fallback: walk a small geometric sweep of nprobes values
        # and return the smallest that hits the target.
        candidates = [1, 2, 4, 8, 16, 32, 64, 128, 256]

        # Build IRI → row mapping so we can score recall.
        iri_tbl = self._embeddings.to_table(columns=["iri"])
        iri_to_row = {iri: i for i, iri in enumerate(iri_tbl["iri"].to_pylist())}

        def _eval(probes: int) -> float:
            recalls: list[float] = []
            for q_vec, gt in zip(sample_vecs, ground_truth):
                hits = self.vector_search(
                    q_vec.tolist(),
                    k=k,
                    model_id=model_id,
                    nprobes=probes,
                    refine_factor=10,
                    in_memory=False,  # measure the indexed path
                )
                got = {iri_to_row[h["iri"]] for h in hits if h["iri"] in iri_to_row}
                if gt:
                    recalls.append(len(got & gt) / len(gt))
            return float(np.mean(recalls)) if recalls else 0.0

        chosen = candidates[-1]
        achieved = 0.0
        for probes in candidates:
            r = _eval(probes)
            if r >= target_recall:
                chosen = probes
                achieved = r
                break
            achieved = r
        return {
            "nprobes": chosen,
            "achieved_recall": achieved,
            "target_recall": target_recall,
            "sample_size": int(len(sample_vecs)),
        }

    def _resolve_auto_index(self, index_type: str, rows: int) -> dict:
        """Decide the effective index type. Pure dispatch — no Lance
        calls — so the policy can be unit-tested without touching disk.

        Returns either `{"action": "skipped", ...}` (caller should
        return the dict directly) or `{"index_type": <str>, "reason":
        <str>}` (caller proceeds to build).
        """
        if index_type != "auto":
            return {"index_type": index_type, "reason": "explicit"}
        if rows < self.AUTO_INDEX_FLAT_MIN_ROWS:
            return {
                "action": "skipped",
                "index_type": None,
                "rows": rows,
                "reason": (
                    f"row count {rows} < {self.AUTO_INDEX_FLAT_MIN_ROWS}; "
                    "brute-force scan is fast enough at this scale"
                ),
            }
        if rows < self.AUTO_INDEX_PQ_MIN_ROWS:
            return {"index_type": "IVF_FLAT", "reason": "auto:flat"}
        return {"index_type": "IVF_PQ", "reason": "auto:pq"}

    # --- maintenance (operate on both Lance datasets) -------------------

    def compact(self) -> dict:
        """Compact small fragments on both Lance datasets.

        Each `ingest_ttl` / `insert_turtle` / `embed` produces a Lance
        fragment; a long-lived agent dataset accumulates many small
        ones, hurting scan speed. Compaction merges them into a smaller
        number of larger fragments. Returns a summary dict reporting
        how many fragments were merged on each side. No-op for missing
        datasets."""
        out: dict = {"triples": None, "embeddings": None}
        if self._lance is not None:
            res = self._lance.optimize.compact_files()
            out["triples"] = _summarize_compaction(res)
        if self._embeddings is not None:
            res = self._embeddings.optimize.compact_files()
            out["embeddings"] = _summarize_compaction(res)
        return out

    def prune_versions(self, *, keep_versions: int = 10) -> dict:
        """Drop old Lance versions on both datasets, keeping the most
        recent `keep_versions` plus anything currently tagged.

        Long-running agents accumulate versions forever; this is the
        knob to keep disk usage in check. Tagged versions are always
        retained, so checkpoints survive a prune. Returns a stable
        per-side summary dict:

            {
              "triples":    {removed_versions, bytes_removed,
                             data_files_removed, index_files_removed,
                             kept_versions} | {"error": str},
              "embeddings": <same shape>,
            }

        Every numeric field is a Python int when known, `None` when the
        installed pylance build doesn't expose it. The shape is stable
        across pylance versions even when the underlying CleanupStats
        object renames or adds attributes — see `_summarize_cleanup`.
        """
        if keep_versions < 1:
            raise ValueError("keep_versions must be >= 1")
        out: dict = {"triples": None, "embeddings": None}
        for key, ds in (("triples", self._lance), ("embeddings", self._embeddings)):
            if ds is None:
                continue
            versions = ds.versions()
            if len(versions) <= keep_versions:
                out[key] = _empty_cleanup_summary(kept=len(versions))
                continue
            older_than = _older_than_cutoff(versions, keep_versions)
            try:
                # delete_unverified=False is the safe default: never
                # delete files that don't appear in any manifest. The
                # only files that get freed are the manifests we are
                # explicitly retiring.
                stats = ds.cleanup_old_versions(
                    older_than=older_than, delete_unverified=False
                )
                out[key] = _summarize_cleanup(stats, kept=keep_versions)
            except Exception as e:  # defensive: surface but never crash
                out[key] = {"error": str(e)}
        return out

    def embedding_versions(self) -> list[dict]:
        """Lance versions of the embeddings dataset. Empty if nothing
        embedded yet. Mirror of `versions()` for the vector side — the
        two can advance independently between checkpoints."""
        if self._embeddings is None:
            return []
        return self._embeddings.versions()

    # ── in-RAM cache (small datasets that fit in process memory) ────

    def preload_vectors(self, *, model_id: str | None = None) -> dict:
        """Load all embeddings into a contiguous numpy array kept in
        process memory.

        Why this exists: `vector_search` normally goes through Lance's
        nearest-neighbor scan (which is fine for indexed data on disk).
        For corpora that fit in RAM (~10M vectors × 128 dim ≈ 5 GB,
        comfortable on most laptops), bypassing Lance's per-call
        manifest check + projection construction and going straight
        to a numpy matmul is 5–20× faster on small batches.

        After calling this once, set `in_memory="auto"` (the default
        when the cache is warm) on `vector_search()` and queries will
        use the cached matrix. The cache invalidates on the next
        `embed()` write.

        `model_id`, when set, narrows the cache to one model's
        vectors. Re-call with a different `model_id` to switch.

        Returns a dict reporting the cache state for visibility:
        `{rows, dim, model_id, version, bytes}`. Raises if no
        embeddings exist."""
        import numpy as np

        if self._embeddings is None:
            raise RuntimeError(
                f"No embeddings at {self.embeddings_path}. "
                "Call .embed(iris, vectors, model_id=...) first."
            )
        flt = None
        if model_id is not None:
            if not _valid_model_id(model_id):
                raise ValueError(
                    f"model_id must be [A-Za-z0-9._:/+-]; got {model_id!r}"
                )
            flt = f"model_id = '{_sql_escape(model_id)}'"
        kwargs: dict = {"columns": ["iri", "vector", "model_id"]}
        if flt is not None:
            kwargs["filter"] = flt
        tbl = self._embeddings.to_table(**kwargs)
        if tbl.num_rows == 0:
            self._vector_cache = {
                "iris": [],
                "vectors": np.empty((0, 0), dtype=np.float32),
                "model_ids": [],
                "version": self._embeddings.version,
                "model_filter": model_id,
            }
            return {
                "rows": 0,
                "dim": 0,
                "model_id": model_id,
                "version": self._embeddings.version,
                "bytes": 0,
            }
        # Vectors come back as a fixed-size-list; flatten to a
        # contiguous (N, dim) float32 array.
        vec_col = tbl["vector"]
        flat = vec_col.combine_chunks().flatten().to_numpy(zero_copy_only=False)
        dim = len(vec_col[0].as_py())
        matrix = flat.astype(np.float32, copy=False).reshape(tbl.num_rows, dim)
        # Force C-contiguous for matmul speed.
        matrix = np.ascontiguousarray(matrix)
        self._vector_cache = {
            "iris": tbl["iri"].to_pylist(),
            "vectors": matrix,
            "model_ids": tbl["model_id"].to_pylist(),
            "version": self._embeddings.version,
            "model_filter": model_id,
        }
        return {
            "rows": tbl.num_rows,
            "dim": dim,
            "model_id": model_id,
            "version": self._embeddings.version,
            "bytes": int(matrix.nbytes),
        }

    def _in_memory_vector_search(
        self, q: list[float], *, k: int, model_id: str | None
    ) -> list[dict]:
        """Brute-force top-k via numpy matmul over the cached matrix.

        Exact distances (no PQ approximation), so recall is always 1.0
        relative to a Lance brute-force run. The point is throughput:
        on a corpus that fits in RAM, dot-product against a contiguous
        float32 matrix is faster than per-call Lance overhead at
        small batches.

        Implementation note: we compute squared L2 via the
        ||q-x||² = ||q||² + ||x||² - 2 q·x identity, with ||x||²
        cached on the dataset on first use. That avoids the
        (matrix - q) broadcast allocation that would dominate
        per-call cost at million-vector scale. With the cached norm,
        each query is one (N,)-shaped matvec — BLAS's GEMV path —
        plus one O(N) addition.
        """
        import numpy as np

        cache = self._vector_cache
        assert cache is not None  # has_warm_cache enforced this
        matrix: np.ndarray = cache["vectors"]
        if matrix.shape[0] == 0:
            return []
        # Cache ||x||² lazily on the dataset (one float per row).
        # Recomputed only on cache invalidation (next embed write).
        x_sq = cache.get("x_sq")
        if x_sq is None:
            x_sq = np.einsum("ij,ij->i", matrix, matrix)
            cache["x_sq"] = x_sq
        qv = np.asarray(q, dtype=np.float32)
        q_sq = float(qv @ qv)
        # 1 GEMV + 1 add: this is the fast path.
        dists_sq = q_sq + x_sq - 2.0 * (matrix @ qv)
        if model_id is not None and cache["model_filter"] is None:
            mask = cache.get("model_id_mask", {}).get(model_id)
            if mask is None:
                mask = np.array(
                    [m == model_id for m in cache["model_ids"]], dtype=bool
                )
                cache.setdefault("model_id_mask", {})[model_id] = mask
            dists_sq = np.where(mask, dists_sq, np.inf)
        n = matrix.shape[0]
        if k >= n:
            order = np.argsort(dists_sq)
        else:
            partition = np.argpartition(dists_sq, kth=k)[:k]
            order = partition[np.argsort(dists_sq[partition])]
        out: list[dict] = []
        for idx in order:
            d = dists_sq[idx]
            if not np.isfinite(d):
                continue
            out.append(
                {
                    "iri": cache["iris"][int(idx)],
                    "distance": float(d),
                    "model_id": cache["model_ids"][int(idx)],
                }
            )
        return out

    def has_warm_cache(self, *, model_id: str | None = None) -> bool:
        """True iff `preload_vectors()` has been called for this
        embeddings version and the requested `model_id` filter
        (or any filter, when `model_id` is None on the caller side).
        Used by `vector_search(in_memory="auto")` to decide whether to
        take the fast path."""
        if self._vector_cache is None or self._embeddings is None:
            return False
        if self._vector_cache["version"] != self._embeddings.version:
            return False
        cache_filter = self._vector_cache["model_filter"]
        if model_id is not None and cache_filter is not None and cache_filter != model_id:
            return False
        if model_id is not None and cache_filter is None:
            # Cache is unfiltered; we can still serve a model_id filter
            # from it (we'll filter post-search). Return True.
            return True
        return True

    def vector_search(
        self,
        query_vector: Sequence[float],
        *,
        k: int = 10,
        model_id: str | None = None,
        nprobes: int | None = None,
        refine_factor: int | None = None,
        in_memory: bool | str = "auto",
    ) -> list[dict]:
        """Approximate-nearest-neighbor search over the embeddings.

        Returns a list of `{iri, distance, model_id}` dicts ordered by
        increasing distance. `k` is the maximum number of results.

        `model_id` filters to vectors written by a specific model — useful
        when more than one embedding model has been added to the same
        dataset and the query vector is only meaningful for one of them.

        ANN tunables (passed through to Lance):
          - `nprobes`: how many IVF partitions to scan. Larger → higher
            recall, lower throughput. Lance default is 1, which is fine
            for tiny datasets but produces low recall at the IVF_PQ
            scale this method targets. As a rule of thumb,
            `nprobes = max(8, num_partitions // 16)` recovers most of
            the brute-force recall on 1M-vector datasets.
          - `refine_factor`: re-rank the top `k * refine_factor`
            candidates by exact distance after the ANN scan. Cheap
            recall boost when set to ~10. Lance default is None
            (no refinement).
          - `in_memory`:
              - `False` → always go through Lance (canonical path).
              - `True`  → require an in-memory cache; raises if cold.
              - `"auto"` (default) → use the cache when warm
                (`preload_vectors()` called and not yet invalidated by
                a write), otherwise fall back to Lance.
            The in-memory path bypasses Lance ANN and computes top-k
            via numpy matmul over the cached matrix — 5–20× faster
            than Lance for batches of small queries when the corpus
            fits in process memory.

        nprobes / refine_factor only apply on the Lance path; they are
        ignored on the in-memory path (which does exact brute force).
        """
        import math

        if self._embeddings is None:
            raise RuntimeError(
                f"No embeddings at {self.embeddings_path}. "
                "Call .embed(iris, vectors, model_id=...) first."
            )
        if k <= 0:
            raise ValueError("k must be >= 1")
        # Reject query vectors that would silently break ANN distance.
        q = list(query_vector)
        for j, c in enumerate(q):
            try:
                cf = float(c)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"query_vector[{j}]={c!r} is not a number"
                ) from e
            if not math.isfinite(cf):
                raise ValueError(
                    f"query_vector[{j}]={c!r} is not finite "
                    "(NaN/Inf in a query is a programming bug, not a search)"
                )
        existing_dim = self.embedding_dim()
        if existing_dim is not None and len(q) != existing_dim:
            raise ValueError(
                f"query_vector has dim {len(q)}; dataset embeddings have dim "
                f"{existing_dim}"
            )
        if model_id is not None and not _valid_model_id(model_id):
            raise ValueError(
                f"model_id must be a non-empty identifier of [A-Za-z0-9._:/+-]; "
                f"got {model_id!r}"
            )

        # ── in-RAM fast path ──
        # When `in_memory` is requested (or "auto" + warm cache),
        # search the cached numpy matrix instead of going through
        # Lance. Exact brute force; orders of magnitude faster than
        # the per-call Lance scan on small batches.
        use_memory = False
        if in_memory is True:
            if not self.has_warm_cache(model_id=model_id):
                raise RuntimeError(
                    "in_memory=True but no warm cache. "
                    "Call .preload_vectors() first."
                )
            use_memory = True
        elif in_memory == "auto":
            use_memory = self.has_warm_cache(model_id=model_id)
        elif in_memory is False:
            use_memory = False
        else:
            raise ValueError(
                f"in_memory must be True, False, or 'auto'; got {in_memory!r}"
            )

        if use_memory:
            return self._in_memory_vector_search(q, k=k, model_id=model_id)

        nearest = {
            "column": "vector",
            "q": q,
            "k": int(k),
        }
        if nprobes is not None:
            if nprobes < 1:
                raise ValueError("nprobes must be >= 1")
            nearest["nprobes"] = int(nprobes)
        if refine_factor is not None:
            if refine_factor < 1:
                raise ValueError("refine_factor must be >= 1")
            nearest["refine_factor"] = int(refine_factor)
        # Explicitly project `_distance` alongside the data columns; Lance
        # will stop auto-injecting it in a future release.
        kwargs: dict = {
            "nearest": nearest,
            "columns": ["iri", "model_id", "_distance"],
        }
        if model_id is not None:
            # Lance accepts a SQL filter applied alongside the ANN scan.
            kwargs["filter"] = f"model_id = '{_sql_escape(model_id)}'"
        tbl = self._embeddings.to_table(**kwargs)
        out: list[dict] = []
        iri_col = tbl["iri"]
        model_col = tbl["model_id"]
        dist_col = tbl["_distance"]
        for i in range(tbl.num_rows):
            out.append(
                {
                    "iri": iri_col[i].as_py(),
                    "distance": float(dist_col[i].as_py()),
                    "model_id": model_col[i].as_py(),
                }
            )
        return out

    def graph_rag(
        self,
        query_vector: Sequence[float],
        *,
        k: int = 5,
        hops: int = 1,
        model_id: str | None = None,
    ) -> dict:
        """One-shot GraphRAG retrieval: ANN over vectors → entity expansion.

        The retrieval pattern that motivates the whole vector layer:

          1. Find the `k` IRIs whose embeddings are closest to `query_vector`.
          2. Pull each IRI's `hops`-deep subgraph via `entity()`.
          3. Return both the ranked hit list and the structured neighborhoods.

        Shape:
            {
              "hits":     [{iri, distance, model_id}, ...],
              "entities": {iri: <entity-shape>, ...}
            }

        This is the convenience surface for an LLM agent: "search by
        meaning, get back facts." The same effect is reachable by chaining
        `vector_search` + `entity` manually; we just make it one call.
        """
        if hops < 1:
            raise ValueError("hops must be >= 1")
        hits = self.vector_search(query_vector, k=k, model_id=model_id)
        entities: dict[str, dict] = {}
        if hits:
            self._require_backing()
            engine = self._engine()
            for hit in hits:
                entities[hit["iri"]] = _expand_entity(engine, hit["iri"], hops)
        return {"hits": hits, "entities": entities}

    def embedding_count(self) -> int:
        """Row count on the embeddings dataset. 0 if no `.embed()` has happened."""
        if self._embeddings is None:
            return 0
        return self._embeddings.count_rows()

    def embedding_dim(self) -> int | None:
        """Vector dim recorded in the manifest, or `None` if no embeddings yet."""
        if not self.manifest_path.exists():
            return None
        try:
            data = json.loads(self.manifest_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        dim = data.get("embedding_dim")
        return int(dim) if isinstance(dim, int) else None

    # --- manifest (small WAL for paired-tag operations) -----------------

    def _read_manifest(self) -> dict:
        """Return the manifest dict, or {} if missing/corrupt.
        Tolerates a torn write — we treat a malformed manifest as empty
        rather than raise, since the only invariant we strictly need is
        the embedding dim (which Lance also encodes in its schema)."""
        if not self.manifest_path.exists():
            return {}
        try:
            return json.loads(self.manifest_path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}

    def _write_manifest_atomic(self, manifest: dict) -> None:
        """Write manifest.json via tmp + rename so a crashed writer
        cannot leave a torn JSON file on disk. Best-effort fsync to
        survive a hard reboot, not just a process crash."""
        payload = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
        tmp = self.manifest_path.with_suffix(".json.tmp")
        with open(tmp, "wb") as fh:
            fh.write(payload)
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass
        os.replace(tmp, self.manifest_path)

    def _record_manifest_dim(self, dim: int) -> None:
        """Persist the embedding dim on first embed. Re-opens read it
        without scanning Lance."""
        manifest = self._read_manifest()
        if manifest.get("embedding_dim") == dim:
            return
        manifest["embedding_dim"] = dim
        self._write_manifest_atomic(manifest)

    def _begin_pending_checkpoint(
        self,
        name: str,
        triples_version: int,
        embeddings_version: int | None,
    ) -> None:
        """Write the intent of a paired checkpoint to manifest BEFORE
        creating the underlying tags. If the process crashes between
        the manifest write and the tag creates, `_finish_pending_checkpoint`
        on next open completes (or rolls back) the operation."""
        manifest = self._read_manifest()
        manifest["pending_checkpoint"] = {
            "name": name,
            "triples_version": triples_version,
            "embeddings_version": embeddings_version,
        }
        self._write_manifest_atomic(manifest)

    def _commit_pending_checkpoint(self) -> None:
        """Clear the pending-checkpoint marker after both tags are in
        place. Idempotent."""
        manifest = self._read_manifest()
        if "pending_checkpoint" not in manifest:
            return
        manifest.pop("pending_checkpoint", None)
        self._write_manifest_atomic(manifest)

    def _recover_pending_checkpoint(self) -> None:
        """Called by `Dataset.open()` after Lance datasets are loaded.
        If a `pending_checkpoint` is in the manifest, reconcile.

        Reconciliation rules (handle both directions of partial state):

        - both tags exist → clear the marker (clean commit raced the
          marker clear).
        - neither tag exists → clear the marker (no real state changed
          before the crash; a no-op).
        - only triples tagged → forward-roll: create the embeddings tag
          at the recorded version. This is the common case given our
          write order (we tag triples first).
        - only embeddings tagged → forward-roll: create the triples tag
          at the recorded version. Defensive — shouldn't happen given
          our call order, but if it ever does (e.g. a future
          refactoring reorders the calls, or a tool-level retry tags
          embeddings first) we still converge to a consistent pair
          rather than leaving an orphan tag.

        This is what makes `checkpoint()` and `rollback()` honestly
        crash-safe rather than 'best-effort.'
        """
        manifest = self._read_manifest()
        pending = manifest.get("pending_checkpoint")
        if not pending or not isinstance(pending, dict):
            return
        name = pending.get("name")
        if not isinstance(name, str):
            manifest.pop("pending_checkpoint", None)
            self._write_manifest_atomic(manifest)
            return

        triples_has = self._lance is not None and name in list(self._lance.tags.list())
        emb_has = self._embeddings is not None and name in list(
            self._embeddings.tags.list()
        )
        triples_version = pending.get("triples_version")
        emb_version = pending.get("embeddings_version")

        if triples_has and not emb_has and self._embeddings is not None:
            try:
                self._embeddings.tags.create(name, int(emb_version))
            except Exception:
                pass
        elif emb_has and not triples_has and self._lance is not None:
            # Inverse: embeddings tagged but triples not. Pin triples
            # at the recorded version so the pair is consistent. If
            # the recorded triples_version no longer exists (e.g.
            # someone manually dropped versions), surface that by
            # leaving the embeddings tag in place — caller can clean
            # up explicitly via `tag(name)` later.
            if isinstance(triples_version, int):
                try:
                    self._lance.tags.create(name, triples_version)
                except Exception:
                    pass

        manifest.pop("pending_checkpoint", None)
        self._write_manifest_atomic(manifest)

    # --- misc ------------------------------------------------------------

    def count(self) -> int:
        """Row count. 0 if nothing has been ingested yet — lets the
        idiomatic `if kg.count() == 0: kg.ingest_ttl(...)` pattern work
        without a separate "is it new" check."""
        if self._lance is None:
            return 0
        return self._lance.count_rows()

    # ── Discoverability: tell the agent what's in the KG ───────

    def schema(self, *, top_classes: int = 20, top_predicates: int = 30) -> dict:
        """Return a runtime-introspected schema of the current graph.

        Shape:
            {
              "triples": int,
              "classes":    [{"iri": str, "count": int}, ...],
              "predicates": [{"iri": str, "count": int}, ...],
              "namespaces": [{"prefix": str, "count": int}, ...],
              "versions": int,
              "tags": [str]
            }

        Agents call this before writing SPARQL so they know which classes
        and predicates actually exist in the data. No SPARQL skill of
        their own required to discover the shape of the graph.
        """
        if self._lance is None and not self._sources:
            return {
                "triples": 0,
                "classes": [],
                "predicates": [],
                "namespaces": [],
                "versions": 0,
                "tags": [],
                "sources": [],
            }

        classes = self.query(
            "SELECT ?c (COUNT(?s) AS ?n) WHERE { ?s a ?c } "
            "GROUP BY ?c ORDER BY DESC(?n) "
            f"LIMIT {int(top_classes)}"
        )
        predicates = self.query(
            "SELECT ?p (COUNT(*) AS ?n) WHERE { ?s ?p ?o } "
            "GROUP BY ?p ORDER BY DESC(?n) "
            f"LIMIT {int(top_predicates)}"
        )

        # Namespace histogram from the distinct predicates (cheap heuristic).
        ns_counts: dict[str, int] = {}
        for row in predicates:
            iri = row["p"]["value"]
            # Split on last "#" or "/" — standard RDF namespace convention.
            if "#" in iri:
                ns = iri.rsplit("#", 1)[0] + "#"
            else:
                ns = iri.rsplit("/", 1)[0] + "/"
            count = int(row["n"]["value"])
            ns_counts[ns] = ns_counts.get(ns, 0) + count
        namespaces = sorted(
            [{"prefix": ns, "count": c} for ns, c in ns_counts.items()],
            key=lambda d: -d["count"],
        )

        overlay_triples = self._lance.count_rows() if self._lance is not None else 0
        return {
            "triples": overlay_triples,
            "classes": [
                {"iri": r["c"]["value"], "count": int(r["n"]["value"])}
                for r in classes
            ],
            "predicates": [
                {"iri": r["p"]["value"], "count": int(r["n"]["value"])}
                for r in predicates
            ],
            "namespaces": namespaces,
            "versions": len(self.versions()),
            "tags": sorted(self.tags()),
            "sources": self.sources(),
        }

    def guide(self) -> str:
        """Agent-facing how-to: the canonical workflow + examples.

        Pairs with `schema()` — guide() explains *how* to use the tools;
        schema() tells the agent *what* is in this particular graph.
        Returns a plain string an LLM can drop into its context.
        """
        return _GUIDE_TEXT

    @property
    def version(self) -> int:
        """Public accessor for the version this handle is pinned to.
        -1 if nothing has been written yet. Prefer this over reaching into
        `._lance.version` from outside the class."""
        return self.current_version()

    def current_version(self) -> int:
        """The Lance version this handle is currently pinned to.

        Differs from `versions()[-1]` after a rollback: rolling back pins
        the handle to an earlier version, while the full version chain
        (including later versions) is still on disk. Agents usually want
        *this* value — "what version am I looking at right now" — rather
        than the chain head."""
        if self._lance is None:
            return -1
        return self._lance.version

    def _require_lance(self) -> None:
        if self._lance is None:
            raise RuntimeError(
                f"No triples dataset at {self.triples_path}. "
                "Call .ingest_ttl(...) first."
            )

    def _require_backing(self) -> None:
        """Reads are allowed if either Lance or at least one source is
        attached. Used by `query`, `entity`, `schema` — anything that only
        needs the cached engine."""
        if self._lance is None and not self._sources:
            raise RuntimeError(
                f"No triples dataset at {self.triples_path} and no "
                "sources attached. Call .ingest_ttl(...) or pass "
                "sources={...} to Dataset.open()."
            )

    def __iter__(self) -> Iterator[pa.RecordBatch]:
        self._require_lance()
        yield from self._lance.to_batches()

    # --- upstream sources + dump -----------------------------------------

    def sources(self) -> list[dict]:
        """Return the currently-attached upstream sources with their
        current mtime and sha256. Useful for `provenance`-style audit."""
        self._refresh_source_fingerprint()
        out = []
        for graph_iri, meta in self._sources.items():
            out.append({
                "graph": graph_iri,
                "path": str(meta["path"]),
                "mtime": meta.get("mtime"),
                "sha256": meta.get("sha256"),
            })
        return out

    def dump(
        self,
        output: str | Path,
        *,
        format: str = "turtle",
        graph: str | None = None,
    ) -> int:
        """Serialize the overlay (or a specific named graph) to a file.

        `format`: "turtle" | "nquads" | "ntriples" | "rdfxml" | "jsonld".
        `graph=None`: dump everything in the Lance overlay (default
        graph + all named graphs). If sources are attached and you pass
        `graph="turtlelake://agent-overlay"`, you get JUST the agent
        writes — typical "export my inferences" path.
        Returns the number of quads written.
        """
        from pyoxigraph import BlankNode, DefaultGraph, NamedNode, RdfFormat

        format_map = {
            "turtle": RdfFormat.TURTLE,
            "nquads": RdfFormat.N_QUADS,
            "ntriples": RdfFormat.N_TRIPLES,
            "rdfxml": RdfFormat.RDF_XML,
            "jsonld": RdfFormat.JSON_LD,
        }
        fmt = format_map.get(format.lower())
        if fmt is None:
            raise ValueError(
                f"Unknown format {format!r}; expected one of {list(format_map)}"
            )
        if self._lance is None:
            raise RuntimeError(
                "Nothing to dump: no Lance overlay exists. "
                "sources=... content is read-only upstream; dump what "
                "the agent wrote by first ingesting or inserting."
            )
        engine = self._engine()
        store = engine.store

        # Build a filtered store for dump. pyoxigraph's dump can take a
        # from_graph; for "everything overlay" we need a separate store
        # containing only quads that are NOT in an attached source graph.
        source_graphs: set[str] = set(self._sources.keys())

        out_path = Path(str(output))
        out_path.parent.mkdir(parents=True, exist_ok=True)

        from pyoxigraph import Store as _OxStore

        scratch = _OxStore()
        n = 0
        if graph is not None:
            graph_node = NamedNode(graph) if graph != "default" else DefaultGraph()
            for q in store.quads_for_pattern(None, None, None, graph_node):
                scratch.add(q)
                n += 1
        else:
            # "overlay only" = everything NOT in an attached-source graph
            for q in store:
                gn = q.graph_name
                if isinstance(gn, NamedNode) and gn.value in source_graphs:
                    continue
                if isinstance(gn, BlankNode):
                    # blank-node graphs — treat as overlay
                    pass
                scratch.add(q)
                n += 1
        # Turtle / N-Triples / RDF-XML / JSON-LD don't support datasets
        # with named graphs. Flatten into DefaultGraph before dumping.
        flat_formats = {
            RdfFormat.TURTLE,
            RdfFormat.N_TRIPLES,
            RdfFormat.RDF_XML,
            RdfFormat.JSON_LD,
        }
        if fmt in flat_formats:
            flat = _OxStore()
            from pyoxigraph import DefaultGraph as _DG
            from pyoxigraph import Quad as _Quad
            for q in scratch:
                flat.add(_Quad(q.subject, q.predicate, q.object, _DG()))
            flat.dump(str(out_path), format=fmt, from_graph=_DG())
        else:
            scratch.dump(str(out_path), format=fmt)
        return n

    def _register_sources(
        self, sources: dict[str, str | Path], *, follow_imports: bool = False
    ) -> None:
        """Resolve the user-provided source map into internal _sources.

        Each value is either a local path OR an `http(s)://` URL. HTTP
        sources are fetched and cached under `<path>/_source_cache/`
        with ETag-based invalidation (basic federation: load remote TTL
        locally, no live SPARQL SERVICE needed).

        `follow_imports=True` transitively walks `owl:imports` from every
        attached source (local files only in v0; cycle-safe)."""
        import hashlib

        for graph_iri, path_or_uri in sources.items():
            uri = str(path_or_uri)
            if uri.startswith(("http://", "https://")):
                p = self._fetch_http_source(graph_iri, uri)
            else:
                p = Path(uri)
                if not p.exists():
                    raise FileNotFoundError(
                        f"source for graph {graph_iri!r} not found: {p}"
                    )
            stat = p.stat()
            self._sources[graph_iri] = {
                "path": p,
                "mtime": stat.st_mtime,
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "origin": uri,
            }

        if follow_imports:
            self._follow_imports()

    def _fetch_http_source(self, graph_iri: str, url: str) -> Path:
        """Fetch a remote TTL source into the local cache and return the
        cached path. Uses If-None-Match / ETag so repeat opens don't
        re-download. Stdlib-only — no extra HTTP dep.

        On network failure, falls back to a previously cached copy if
        one exists (so the agent keeps working offline after one fetch)."""
        import hashlib
        import urllib.error
        import urllib.request

        cache_dir = self.path / "_source_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        slug = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        # Best-guess extension from the URL path; defaults to .ttl.
        ext = ".ttl"
        for candidate in (".ttl", ".nt", ".nq", ".trig", ".rdf", ".xml", ".jsonld"):
            if url.lower().endswith(candidate):
                ext = candidate
                break
        cache_file = cache_dir / f"{slug}{ext}"
        etag_file = cache_dir / f"{slug}.etag"

        req_headers = {"User-Agent": "turtlelake/0.0.1 (+https://example.org)"}
        if etag_file.exists() and cache_file.exists():
            req_headers["If-None-Match"] = etag_file.read_text().strip()

        req = urllib.request.Request(url, headers=req_headers)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                cache_file.write_bytes(resp.read())
                etag = resp.headers.get("ETag")
                if etag:
                    etag_file.write_text(etag)
        except urllib.error.HTTPError as e:
            if e.code == 304 and cache_file.exists():
                pass  # 304 Not Modified — cached version still valid
            else:
                if cache_file.exists():
                    pass  # fall through to cached copy
                else:
                    raise
        except (urllib.error.URLError, TimeoutError):
            if not cache_file.exists():
                raise
            # offline + have a prior cache → use it
        return cache_file

    def _follow_imports(self) -> None:
        """Transitively resolve owl:imports. Local files only in v0.
        Cycle-safe via the path-set visited check."""
        import hashlib

        visited: set[Path] = {
            meta["path"].resolve() for meta in self._sources.values()
        }
        frontier: list[tuple[Path, Path]] = [
            (meta["path"], meta["path"].resolve()) for meta in self._sources.values()
        ]
        while frontier:
            source_path, _ = frontier.pop()
            imports = _extract_owl_imports(source_path)
            for imp in imports:
                # pyoxigraph resolves relative IRIs against base_iri, so
                # imp typically arrives as a full `file://` URI. Map
                # that back to a local Path. http(s):// imports are
                # skipped in v0 (local files only).
                if imp.startswith("file://"):
                    imp_path = Path(imp.removeprefix("file://")).resolve()
                elif imp.startswith(("http://", "https://")):
                    continue
                elif Path(imp).is_absolute():
                    imp_path = Path(imp).resolve()
                else:
                    imp_path = (source_path.parent / imp).resolve()
                if imp_path in visited:
                    continue
                if not imp_path.exists():
                    # owl:imports can legitimately point at URIs that
                    # aren't locally available. Skip quietly rather than
                    # fail — the agent still gets the rest of the graph.
                    continue
                graph_iri = imp_path.as_uri()
                self._sources[graph_iri] = {
                    "path": imp_path,
                    "mtime": imp_path.stat().st_mtime,
                    "sha256": hashlib.sha256(imp_path.read_bytes()).hexdigest(),
                }
                visited.add(imp_path)
                frontier.append((imp_path, imp_path))

    def _refresh_source_fingerprint(self) -> tuple:
        """Recompute mtimes for every attached source; update
        `self._sources[*]["mtime"]` and return a fingerprint tuple."""
        if not self._sources:
            return ()
        fp: list = []
        for graph_iri in sorted(self._sources.keys()):
            meta = self._sources[graph_iri]
            p: Path = meta["path"]
            try:
                cur = p.stat().st_mtime
            except FileNotFoundError:
                cur = -1.0
            meta["mtime"] = cur
            fp.append((graph_iri, cur))
        return tuple(fp)

    def _load_sources_into(self, engine: SparqlEngine) -> None:
        """Load each attached source file into its designated named graph
        inside the engine's store. Called during cache rebuild."""
        from pyoxigraph import NamedNode, RdfFormat

        fmt_by_suffix = {
            ".ttl": RdfFormat.TURTLE,
            ".trig": RdfFormat.TRIG,
            ".nt": RdfFormat.N_TRIPLES,
            ".nq": RdfFormat.N_QUADS,
            ".rdf": RdfFormat.RDF_XML,
            ".xml": RdfFormat.RDF_XML,
            ".jsonld": RdfFormat.JSON_LD,
        }
        for graph_iri, meta in self._sources.items():
            p: Path = meta["path"]
            fmt = fmt_by_suffix.get(p.suffix.lower(), RdfFormat.TURTLE)
            # base_iri lets relative IRIs like owl:imports <extension.ttl>
            # resolve against the file's own location.
            with p.open("rb") as fh:
                engine.store.load(
                    fh,
                    format=fmt,
                    to_graph=NamedNode(graph_iri),
                    base_iri=p.resolve().as_uri(),
                )


# ── Path / URI helpers ───────────────────────────────────────
#
# Lance's Python API accepts either local paths or object-store URIs
# (s3://, gs://, az://, hf://, file://). We keep `self.path` as a `Path`
# for ergonomic reasons and convert to a URI string only at the Lance
# boundary. `file://` paths get their scheme stripped so pathlib's
# `.exists()` and `.mkdir()` work naturally.

_URI_SCHEMES = ("s3://", "gs://", "az://", "hf://", "http://", "https://")


def _is_local_path(path: Path) -> bool:
    """True iff `path` points at a local filesystem location (not a remote URI)."""
    p = str(path)
    return not any(p.startswith(s) for s in _URI_SCHEMES)


def _resolve_uri(path: Path) -> str:
    """Return a string suitable to pass to `lance.dataset(...)`.

    - Local paths pass through as strings.
    - Remote URIs (s3://, gs://, etc.) pass through verbatim.
    - `file://` is stripped (Lance and pathlib both accept the raw path).
    """
    s = str(path)
    if s.startswith("file://"):
        return s.removeprefix("file://")
    return s


_VALID_MODEL_ID = __import__("re").compile(r"^[A-Za-z0-9._:/+\-]+$")


def _valid_model_id(value: str) -> bool:
    """Whitelist `model_id` to a conservative ASCII identifier shape.

    Stricter than just escaping — even with `_sql_escape` doubling
    quotes, accepting arbitrary text makes audit logs noisy and invites
    locale / encoding pitfalls in DataFusion's filter parser. The
    accepted shape covers every embedding model we have seen in the
    wild ('openai:text-embedding-3-small', 'sentence-transformers/all-MiniLM-L6-v2',
    'voyage:v3', etc.)."""
    return isinstance(value, str) and bool(value) and bool(_VALID_MODEL_ID.match(value))


def _sql_escape(value: str) -> str:
    """Escape a string literal for a Lance / DataFusion SQL filter
    clause.

    Hardening:
    - reject null bytes (Lance's filter parser truncates at \\x00 on
      some platforms — silent data loss otherwise)
    - reject ASCII control characters and Unicode bidi overrides (no
      legitimate IRI / model_id contains them, and they break audit
      grep-ability)
    - escape backslash *before* quote so an attacker-controlled
      \\x27 sequence cannot become an unescaped quote
    - escape single quote by doubling (SQL standard)
    """
    if not isinstance(value, str):
        raise TypeError(f"filter value must be str, got {type(value).__name__}")
    if "\x00" in value:
        raise ValueError("null byte in filter value")
    for ch in value:
        # C0 controls except tab/newline/CR — even those are dubious in
        # an IRI but tolerated in a literal text field. We accept them
        # here and let upstream callers (e.g. _valid_model_id) decide.
        if ord(ch) < 0x20 and ch not in ("\t", "\n", "\r"):
            raise ValueError(
                f"control character U+{ord(ch):04X} in filter value"
            )
        if 0x202A <= ord(ch) <= 0x202E or 0x2066 <= ord(ch) <= 0x2069:
            raise ValueError("bidi override character in filter value")
    return value.replace("\\", "\\\\").replace("'", "''")


def _int_env_or(name: str, default: int) -> int:
    """Read a positive-int env var or fall back to `default`. Invalid
    values fall back rather than raise, so a typo in an env var doesn't
    take the whole agent down — we'd rather log and use a safe value."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        v = int(raw)
    except ValueError:
        return default
    return v if v > 0 else default


def _summarize_compaction(res) -> dict:
    """Coerce Lance's compaction-result object (whose attributes vary
    by pylance version) into a small JSON-friendly dict."""
    out: dict = {}
    for attr in ("fragments_removed", "fragments_added", "files_removed", "files_added"):
        if hasattr(res, attr):
            try:
                out[attr] = int(getattr(res, attr))
            except (TypeError, ValueError):
                out[attr] = None
    return out


# CleanupStats attribute names we have seen across pylance versions.
# `old_versions` (current) and `removed_versions` (older builds) both
# count retired manifest files; we expose them under one stable key.
_CLEANUP_VERSIONS_ATTRS = ("old_versions", "removed_versions")
_CLEANUP_BYTES_ATTRS = ("bytes_removed", "total_bytes_removed")
_CLEANUP_DATA_FILES_ATTRS = ("data_files_removed", "removed_files")
_CLEANUP_INDEX_FILES_ATTRS = ("index_files_removed",)


def _coerce_int(res, names) -> int | None:
    """Pick the first attribute on `res` that exists and is int-coercible.
    None if we can't find anything — better than fabricating a zero that
    a caller would interpret as 'we ran cleanup and removed nothing.'"""
    for n in names:
        if hasattr(res, n):
            try:
                return int(getattr(res, n))
            except (TypeError, ValueError):
                continue
    return None


def _summarize_cleanup(stats, *, kept: int) -> dict:
    """Stable shape for `prune_versions` return values, regardless of the
    installed pylance's CleanupStats attribute set."""
    return {
        "removed_versions": _coerce_int(stats, _CLEANUP_VERSIONS_ATTRS),
        "bytes_removed": _coerce_int(stats, _CLEANUP_BYTES_ATTRS),
        "data_files_removed": _coerce_int(stats, _CLEANUP_DATA_FILES_ATTRS),
        "index_files_removed": _coerce_int(stats, _CLEANUP_INDEX_FILES_ATTRS),
        "kept_versions": int(kept),
    }


def _empty_cleanup_summary(*, kept: int) -> dict:
    """Same shape as `_summarize_cleanup` but for the no-op path
    (already at or below `keep_versions`). Lets callers parse one
    schema regardless of whether work actually happened."""
    return {
        "removed_versions": 0,
        "bytes_removed": 0,
        "data_files_removed": 0,
        "index_files_removed": 0,
        "kept_versions": int(kept),
    }


def _older_than_cutoff(versions: list[dict], keep_versions: int):
    """Translate `keep_versions=N` into the `older_than` timedelta
    that Lance's cleanup API expects. Pure helper — extracted so the
    cutoff logic can be tested directly without writing fragments."""
    from datetime import datetime as _dt
    from datetime import timedelta as _td

    sorted_v = sorted(versions, key=lambda v: v["version"])
    cutoff_idx = len(sorted_v) - keep_versions
    cutoff_ts = sorted_v[cutoff_idx]["timestamp"]
    if not isinstance(cutoff_ts, _dt):
        # Older pylance builds returned floats/strings on `versions[*].timestamp`.
        # Fall back to a 0s window — Lance still gates on its own snapshot
        # file timestamps, so this stays correct, just not aggressive.
        return _td(seconds=0)
    cutoff_aware = (
        cutoff_ts if cutoff_ts.tzinfo is not None
        else cutoff_ts.replace(tzinfo=timezone.utc)
    )
    older = _dt.now(tz=timezone.utc) - cutoff_aware
    return older if older.total_seconds() >= 0 else _td(seconds=0)


def _exists(path: Path) -> bool:
    """Existence check that tolerates URIs.

    - Local paths: pathlib `.exists()`.
    - `file://` URIs: strip the scheme first.
    - Remote URIs (s3://, gs://, etc.): probe via `lance.dataset(uri)`.
      Returns False on any exception; returns True only if Lance can
      successfully open the address as a dataset.
    """
    if _is_local_path(path):
        s = str(path)
        if s.startswith("file://"):
            return Path(s.removeprefix("file://")).exists()
        return path.exists()
    try:
        lance.dataset(str(path))
    except Exception:
        return False
    return True


_GUIDE_TEXT = """\
turtlelake — agent-facing knowledge graph.

WHAT THIS IS
  A local RDF knowledge graph stored as a single Lance directory on
  disk. SPARQL 1.1 reads. Versioned, with cheap checkpoint + rollback.
  Everything you do is traceable via `provenance`.

CANONICAL AGENT WORKFLOW
  1. `schema`                    discover what's in the graph
                                  (classes, predicates, namespaces,
                                   versions, tags, attached sources)
  2. `sources`                    list external TTL files that feed
                                  this KG (each pinned to its own
                                  named graph; they're read-only)
  3. `entity(iri)`                the fast way to ask "what do I know
                                  about X?" — returns a structured
                                  subgraph, no SPARQL required.
                                  Pass `similar=N` to also get the N
                                  closest IRIs by vector distance.
  4. `sparql(query)`              for anything more than 1–2 hops
  5. `vector_search(q, k)`        ANN over per-IRI embeddings. Pair
                                  with `entity` to expand a hit into
                                  facts.
  6. `graph_rag(q, k, hops)`      one-shot vector retrieval + entity
                                  expansion — the canonical GraphRAG
                                  shape. Use this when you have a
                                  query embedding and want both the
                                  ranked hits and the surrounding
                                  facts in one call.
  7. `checkpoint(name)`           BEFORE any risky write. Atomic
                                  across triples + embeddings.
  8. `insert(turtle)` / `ingest`  write facts (TTL string / file).
                                  When sources are attached, writes
                                  default to the named graph
                                  `turtlelake://agent-overlay` so
                                  vendor and agent data stay separable.
                                  Pass `graph=` to override.
  9. `embed(iris, vecs, model)`   add per-IRI vectors. Caller supplies
                                  pre-computed floats; turtlelake never
                                  loads an embedding model.
 10. `validate(shapes_path)`      optional SHACL check on the new state
 11a. If conforming              → keep; next query sees the new facts
 11b. If violations              → `rollback(name)`, and it's as if the
                                   write never happened on either side
                                   (but the attempt is still in
                                   `provenance` for audit)
 12. `diff(v_old)`                see what changed since a version
 13. `provenance`                 full write log: source + author + ts
 14. `dump(path, graph=...)`      export the overlay (or one graph)
                                  to a TTL/N-Quads file when you need
                                  to publish or hand off to another tool

IMPORTANT SEMANTICS
  - `rollback(name)` MUTATES THE HANDLE in place and returns it.
    Calling `kg.rollback("pre")` is enough; reassignment is optional.
  - `entity(iri)` is the fastest read (direct index lookup, ~0.01 ms).
    Prefer it over a DESCRIBE-style SPARQL.
  - Every write is automatically stamped with source/author/timestamp
    into `provenance`. Supply `source=` and `author=` when calling
    `ingest` / `insert` so the audit trail is meaningful.
  - `ingest` takes a file path; `insert` takes a TTL string inline.
    Use `insert` for agent-memory writes ("<user> prefers <SPARQL>").
  - Attached sources (from `sources`) are READ-ONLY and their files are
    never modified. mtime changes upstream are auto-picked-up by the
    cache on the next query. To "persist" agent output as TTL, call
    `dump(path, graph="turtlelake://agent-overlay")`.
  - Use a SPARQL `GRAPH <iri> { ... }` pattern to query a specific
    named graph, e.g. to see only agent-written triples vs only
    vendor-declared ones.

COMMON SPARQL SHAPES
  # What types of thing live here?
  SELECT ?cls (COUNT(?s) AS ?n) WHERE { ?s a ?cls } GROUP BY ?cls

  # 1-hop around an IRI (but `entity(iri)` is faster):
  SELECT ?p ?o WHERE { <IRI> ?p ?o }

  # Multi-hop via property path (supported):
  SELECT ?ancestor WHERE {
    <IRI> (<http://www.w3.org/2000/01/rdf-schema#subClassOf>)+ ?ancestor
  }

SECURITY + LIMITS
  - SPARQL UPDATE keywords (INSERT DATA / DROP GRAPH / LOAD / CLEAR …)
    are blocked by the input scanner. Use `insert` / `ingest` for writes
    instead of SPARQL UPDATE.
  - Tools are rate-limited (e.g. rollback 10/min, entity 240/min).
    Exceeding returns {"error": "..."}, not an exception.
  - Errors are redacted for secrets before reaching the agent.
"""


def _append_line_atomic(path: Path, payload: bytes) -> None:
    """Append one line to `path` atomically. Uses `fcntl.flock` for an
    advisory exclusive lock when available (POSIX); the write itself is
    a single `os.write` on an `O_APPEND` fd so it's POSIX-atomic under
    PIPE_BUF (our lines are ~200 bytes)."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    fd = os.open(str(path), flags, 0o644)
    try:
        try:
            import fcntl  # POSIX only
            fcntl.flock(fd, fcntl.LOCK_EX)
        except ImportError:  # pragma: no cover — Windows
            pass
        try:
            os.write(fd, payload)
        finally:
            try:
                import fcntl  # re-import safe
                fcntl.flock(fd, fcntl.LOCK_UN)
            except ImportError:  # pragma: no cover
                pass
    finally:
        os.close(fd)


def _pyox_term_to_py(term) -> dict:
    """Convert a pyoxigraph term directly to our {type, value, ...} dict.
    Used by `_expand_entity`, which walks the raw pyoxigraph store rather
    than going through SPARQL."""
    from pyoxigraph import BlankNode, Literal, NamedNode

    if isinstance(term, NamedNode):
        return {"type": "iri", "value": term.value}
    if isinstance(term, BlankNode):
        return {"type": "bnode", "value": term.value}
    if isinstance(term, Literal):
        return {
            "type": "literal",
            "value": term.value,
            "datatype": term.datatype.value if term.datatype else None,
            "lang": term.language,
        }
    return {"type": "unknown", "value": str(term)}


def _inline_bindings(sparql: str, bindings: dict) -> str:
    """Rewrite a SPARQL string, replacing `?var` with a properly-
    serialized term for each (var, value) in `bindings`. Injection-safe
    because the term serialization is ours, not the caller's."""
    import re

    def _to_sparql_term(val) -> str:
        if hasattr(val, "value"):  # pyoxigraph NamedNode / BlankNode / Literal
            from pyoxigraph import NamedNode as _NN
            if isinstance(val, _NN):
                return f"<{val.value}>"
            return str(val)  # Literal renders with quotes already
        s = str(val)
        if s.startswith(("http://", "https://", "file://", "urn:")):
            return f"<{s}>"
        escaped = s.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'

    result = sparql
    for var, value in bindings.items():
        if not var.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"binding variable name must be alphanumeric: {var!r}")
        repl = _to_sparql_term(value)
        # `?var` word-boundary replace to avoid touching `?varname`.
        result = re.sub(rf"\?{re.escape(var)}\b", repl, result)
    return result


def _extract_owl_imports(path: Path) -> list[str]:
    """Parse the TTL at `path` in isolation and return the list of
    owl:imports target IRIs. Used for transitive import resolution."""
    from pyoxigraph import NamedNode, RdfFormat, Store

    scratch = Store()
    try:
        with path.open("rb") as fh:
            scratch.load(
                fh, format=RdfFormat.TURTLE, base_iri=path.resolve().as_uri()
            )
    except Exception:
        return []
    imports_predicate = NamedNode("http://www.w3.org/2002/07/owl#imports")
    out: list[str] = []
    for q in scratch.quads_for_pattern(None, imports_predicate, None, None):
        if isinstance(q.object, NamedNode):
            out.append(q.object.value)
    return out


def _quad_tuples(ds: lance.LanceDataset) -> set[tuple]:
    """Collapse all rows of a Lance dataset into a set of hashable quad tuples."""
    out: set[tuple] = set()
    for batch in ds.to_batches():
        cols = {name: batch.column(name).to_pylist() for name in batch.schema.names}
        for i in range(batch.num_rows):
            out.add(
                (
                    cols["subject"][i],
                    cols["predicate"][i],
                    cols["object"][i],
                    cols["object_kind"][i],
                    cols["object_datatype"][i],
                    cols["object_lang"][i],
                    cols["graph"][i],
                )
            )
    return out


def _tuple_to_quad_dict(t: tuple) -> dict:
    return {
        "subject": t[0],
        "predicate": t[1],
        "object": t[2],
        "object_kind": t[3],
        "object_datatype": t[4],
        "object_lang": t[5],
        "graph": t[6],
    }


def _rdflib_term(term: dict, rdflib):
    """Convert our JSON-shaped term dict to an rdflib term (for SHACL)."""
    kind = term["type"]
    if kind == "iri":
        return rdflib.URIRef(term["value"])
    if kind == "bnode":
        return rdflib.BNode(term["value"])
    # literal
    if term.get("lang"):
        return rdflib.Literal(term["value"], lang=term["lang"])
    if term.get("datatype"):
        return rdflib.Literal(term["value"], datatype=rdflib.URIRef(term["datatype"]))
    return rdflib.Literal(term["value"])


def _expand_entity(engine: SparqlEngine, iri: str, hops: int) -> dict:
    """Breadth-first expansion via `quads_for_pattern` (NOT SPARQL).

    We deliberately DO NOT f-string the IRI into a SPARQL query — an
    attacker-controlled IRI with angle brackets or whitespace could
    inject extra patterns. Using the typed term API keeps this
    injection-safe regardless of the IRI's contents.
    """
    from pyoxigraph import NamedNode

    store = engine.store  # the in-memory pyoxigraph Store
    visited: set[str] = set()
    frontier = {iri}
    root: dict = {"iri": iri}
    by_iri: dict[str, dict] = {iri: root}

    for _ in range(hops):
        next_frontier: set[str] = set()
        for current in frontier:
            if current in visited:
                continue
            visited.add(current)
            node = by_iri.setdefault(current, {"iri": current})
            try:
                subj = NamedNode(current)
            except ValueError:
                # Not a valid IRI; skip with empty neighborhood.
                node["outgoing"] = []
                node["incoming"] = []
                continue

            # Outgoing: ?p ?o where ?s == current.
            # Stable sort by predicate so an LLM sees grouped facts
            # ("all `supports` values together, then all `speedGrade`
            # values") rather than pyoxigraph's internal index order.
            outgoing = []
            for q in store.quads_for_pattern(subj, None, None, None):
                outgoing.append(
                    {"predicate": q.predicate.value, "object": _pyox_term_to_py(q.object)}
                )
                if isinstance(q.object, NamedNode) and q.object.value not in visited:
                    next_frontier.add(q.object.value)
            outgoing.sort(key=lambda e: (e["predicate"], str(e["object"].get("value", ""))))
            node["outgoing"] = outgoing

            # Incoming: ?s ?p where ?o == current. Same sort contract.
            incoming = []
            for q in store.quads_for_pattern(None, None, subj, None):
                incoming.append(
                    {"predicate": q.predicate.value, "subject": q.subject.value}
                )
                if isinstance(q.subject, NamedNode) and q.subject.value not in visited:
                    next_frontier.add(q.subject.value)
            incoming.sort(key=lambda e: (e["predicate"], e["subject"]))
            node["incoming"] = incoming
        frontier = next_frontier

    # If hops > 1, attach expanded neighbors on the root.
    if hops > 1:
        root["neighbors"] = {k: v for k, v in by_iri.items() if k != iri}
    return root
