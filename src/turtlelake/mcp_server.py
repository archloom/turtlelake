"""FastMCP server exposing a turtlelake Dataset to a local agent.

Every tool is wrapped with the turtlelake security stack:
  rate limit → input scan → audit log → execute → metrics → redacted errors.

This mirrors SecureFluidTopicsTool in the parent repo's src/tools/.

Tools registered (22 total):

  discovery   : guide, schema, sources     -- tell the agent what the KG
                                              is, what's in it, and which
                                              external sources contribute
  read        : sparql, entity, scan, explain
  write       : ingest, insert, checkpoint, rollback
  versioning  : versions, refresh, diff
  audit       : provenance
  quality     : validate  (requires `turtlelake[shacl]`)
  export      : dump                       -- serialize overlay (or a
                                              specific graph) back to TTL
  saved       : save_query, run_saved      -- parameterized query library
  vectors     : embed, vector_search, graph_rag, build_vector_index
                                            -- store per-IRI vectors and
                                              do semantic + structural
                                              retrieval in one shot
  maintenance : compact, prune_versions     -- keep dataset disk usage
                                              and scan speed in check
                                              on long-lived projects

Env vars:
  TURTLELAKE_PATH           dataset directory (default ./.turtlelake)
  TURTLELAKE_RATE_LIMIT     requests per minute per tool (default 30)
  TURTLELAKE_MAX_INGEST_BYTES  size cap per ingest call (default 256 MiB)
  TURTLELAKE_AUDIT          1 to log to stderr (default 1)
"""

from __future__ import annotations

import json
import os
import signal
import sys
from pathlib import Path

from turtlelake import Dataset
from turtlelake.config import RuntimeConfig
from turtlelake.observability import metrics
from turtlelake.security import audit_log
from turtlelake.secure_tool import secure


def tool_names() -> list[str]:
    """Canonical tool-set contract. Tests assert equality with the
    server's registered tools."""
    return [
        "guide",
        "schema",
        "sources",
        "sparql",
        "entity",
        "scan",
        "explain",
        "ingest",
        "insert",
        "checkpoint",
        "rollback",
        "versions",
        "refresh",
        "diff",
        "provenance",
        "validate",
        "dump",
        "save_query",
        "run_saved",
        "embed",
        "vector_search",
        "graph_rag",
        "build_vector_index",
        "compact",
        "prune_versions",
    ]


def build_server(store_path: Path | None = None, config: RuntimeConfig | None = None):
    """Construct and return a configured FastMCP server. Split from `main()`
    so tests can introspect the registered tool set without booting stdio."""
    try:
        from fastmcp import FastMCP
    except ImportError as e:  # pragma: no cover
        raise SystemExit("fastmcp is not installed. Run `uv sync --extra mcp`.") from e

    cfg = config or RuntimeConfig.from_env()
    path = store_path or cfg.store_path

    # Parse TURTLELAKE_SOURCES -- either JSON ({graph: path, ...}) or
    # comma-separated graph=path pairs for shell friendliness.
    sources_env = os.environ.get("TURTLELAKE_SOURCES")
    sources_map: dict[str, str] | None = None
    if sources_env:
        sources_env = sources_env.strip()
        if sources_env.startswith("{"):
            sources_map = json.loads(sources_env)
        else:
            sources_map = {}
            for pair in sources_env.split(","):
                if "=" not in pair:
                    continue
                g, p = pair.split("=", 1)
                sources_map[g.strip()] = p.strip()

    audit_log(
        "boot",
        {
            "store_path": str(path),
            "sources": list(sources_map.keys()) if sources_map else [],
            "rate_limit": cfg.rate_limit_max_per_minute,
        },
    )

    state = {"dataset": Dataset.open(path, sources=sources_map)}
    mcp = FastMCP("turtlelake")

    def ds() -> Dataset:
        return state["dataset"]

    def _record_version() -> None:
        v = state["dataset"].version
        if v >= 0:
            metrics.dataset_version.labels(path=str(path)).set(v)

    # --- discovery (call these FIRST when you find a new KG) ------------

    @mcp.tool()
    @secure("guide")
    def guide() -> str:
        """How to use this knowledge graph. Returns a short
        plain-text walkthrough of the canonical agent workflow:
        schema → entity / sparql → checkpoint → insert → validate →
        rollback-or-keep → diff → provenance. Call this ONCE at the
        start of a session -- it tells you which tools to call and
        in what order. Pairs with `schema` which tells you what's
        actually in THIS graph."""
        return ds().guide()

    @mcp.tool()
    @secure("schema")
    def schema(top_classes: int = 20, top_predicates: int = 30) -> str:
        """Runtime introspection of the current graph. Returns JSON:
        {triples, classes[], predicates[], namespaces[], versions, tags, sources[]}.
        Each class/predicate entry has {iri, count}. Use this to learn
        what classes and predicates exist before writing SPARQL --         avoids asking for a predicate the graph doesn't have.
        `sources` lists external TTL files feeding this KG (with mtime)."""
        return json.dumps(
            ds().schema(top_classes=top_classes, top_predicates=top_predicates),
            default=str,
        )

    @mcp.tool()
    @secure("sources")
    def sources() -> str:
        """List external TTL sources attached to this dataset as JSON.
        Each entry: {graph, path, mtime, sha256}. Returns [] when no
        sources are attached (the Lance overlay is the only data)."""
        return json.dumps(ds().sources(), default=str)

    # --- read -----------------------------------------------------------

    @mcp.tool()
    @secure("sparql")
    def sparql(query: str, timeout_ms: int | None = None) -> str:
        """Run a SPARQL 1.1 query against the current version. Returns a
        JSON array of binding dicts. For "what do I know about X?"
        questions prefer `entity(iri)` -- it's faster and returns a
        structured subgraph without needing SPARQL skill.

        `timeout_ms` aborts long-running queries (default from
        TURTLELAKE_QUERY_TIMEOUT_MS env var, else no limit)."""
        return json.dumps(ds().query(query, timeout_ms=timeout_ms), default=str)

    @mcp.tool()
    @secure("explain")
    def explain(query: str) -> str:
        """Return a plain-text query-plan sketch. Useful when a SPARQL
        is slow: the output shows which triple patterns the engine will
        scan, in the order it sees them. Put the most-selective pattern
        first to speed things up."""
        return ds().explain(query)

    @mcp.tool()
    @secure("entity")
    def entity(
        iri: str,
        hops: int = 1,
        similar: int = 0,
        model_id: str | None = None,
    ) -> str:
        """Return everything known about `iri` within N hops as JSON.
        Shape: {iri, outgoing, incoming, neighbors?, similar?}.

        `similar > 0` appends the top-`similar` IRIs by vector distance
        (requires embeddings to be present and `iri` itself to have one)."""
        return json.dumps(
            ds().entity(iri, hops=hops, similar=similar, model_id=model_id),
            default=str,
        )

    @mcp.tool()
    @secure("scan")
    def scan(limit: int = 100) -> str:
        """Return the first `limit` quads as a JSON array. Debug only --         prefer `sparql` for structured queries."""
        tbl = ds().scan()
        return json.dumps(tbl.slice(0, limit).to_pylist(), default=str)

    # --- write ----------------------------------------------------------

    @mcp.tool()
    @secure("ingest")
    def ingest(
        path: str,
        source: str | None = None,
        author: str | None = None,
        graph: str | None = None,
    ) -> str:
        """Parse a TTL / N-Quads / JSON-LD / RDF-XML file and append as a
        new Lance version. Records provenance. `graph` routes the import
        into a specific named graph; when sources are attached and
        graph is omitted, writes default to `turtlelake://agent-overlay`
        so vendor and agent data stay separable."""
        _enforce_ingest_policy(Path(path), cfg.max_ingest_bytes, cfg.ingest_root)
        n = ds().ingest_ttl(path, source=source, author=author, graph=graph)
        metrics.ingest_rows.labels(kind="ingest_ttl").inc(n)
        _record_version()
        return f"ingested {n} quads; dataset now has {ds().count()} rows"

    @mcp.tool()
    @secure("insert")
    def insert(
        turtle: str,
        source: str | None = None,
        author: str | None = None,
        graph: str | None = None,
    ) -> str:
        """Append quads parsed from a TTL snippet string. Agent-memory
        entry point -- add facts without a file. `graph` routes the
        write into a specific named graph (defaults to
        `turtlelake://agent-overlay` when sources are attached)."""
        if len(turtle.encode("utf-8")) > cfg.max_ingest_bytes:
            raise ValueError(
                f"insert payload too large (>{cfg.max_ingest_bytes} bytes)."
            )
        n = ds().insert_turtle(turtle, source=source, author=author, graph=graph)
        metrics.ingest_rows.labels(kind="insert_turtle").inc(n)
        _record_version()
        return f"inserted {n} quads; dataset now has {ds().count()} rows"

    # --- versioning -----------------------------------------------------

    @mcp.tool()
    @secure("checkpoint")
    def checkpoint(name: str, author: str | None = None) -> str:
        """Tag the current version. Call before any risky write so
        `rollback(name)` can undo it. Idempotent."""
        v = ds().checkpoint(name, author=author)
        _record_version()
        return f"tagged version {v} as {name!r}"

    @mcp.tool()
    @secure("rollback")
    def rollback(name: str) -> str:
        """Re-open the dataset at a previously tagged version. Subsequent
        reads see the old state; subsequent writes fork from it."""
        state["dataset"] = ds().rollback(name)
        _record_version()
        return f"rolled back to {name!r} (version {state['dataset'].version})"

    @mcp.tool()
    @secure("versions")
    def versions() -> str:
        """List available Lance versions and tags as JSON."""
        return json.dumps(
            {"versions": ds().versions(), "tags": ds().tags()}, default=str
        )

    @mcp.tool()
    @secure("refresh")
    def refresh() -> str:
        """Re-open the dataset at the latest committed version. Use after
        another process has written to the same path."""
        v = ds().refresh()
        metrics.dataset_version.labels(path=str(path)).set(v)
        return f"refreshed to version {v}"

    @mcp.tool()
    @secure("diff")
    def diff(from_version: int, to_version: int) -> str:
        """Return quads added and removed between two Lance versions.
        Shape: {added: [...], removed: [...]}."""
        return json.dumps(ds().diff(from_version, to_version), default=str)

    # --- audit ----------------------------------------------------------

    @mcp.tool()
    @secure("provenance")
    def provenance() -> str:
        """Ordered list of writes with {version, source, author,
        timestamp, kind, row_delta}. Use to trace triple origin."""
        return json.dumps(ds().provenance(), default=str)

    # --- quality --------------------------------------------------------

    @mcp.tool()
    @secure("validate")
    def validate(shapes_path: str) -> str:
        """Validate the current dataset against a SHACL shapes TTL file.
        Returns {conforms, report_text}. Requires the optional `shacl`
        extra: `pip install 'turtlelake[shacl]'`."""
        return json.dumps(ds().validate(shapes_path), default=str)

    # --- export ---------------------------------------------------------

    @mcp.tool()
    @secure("dump")
    def dump(path: str, format: str = "turtle", graph: str | None = None) -> str:
        """Serialize the Lance overlay (or a specific named graph) to a
        file. `format`: turtle | nquads | ntriples | rdfxml | jsonld.
        `graph=None` → dump all overlay quads (vendor sources excluded).
        Pass `graph="turtlelake://agent-overlay"` to export just the
        agent's writes."""
        n = ds().dump(path, format=format, graph=graph)
        return f"wrote {n} quads to {path}"

    # --- saved queries --------------------------------------------------

    @mcp.tool()
    @secure("save_query")
    def save_query(name: str, sparql: str, description: str = "") -> str:
        """Save a named SPARQL query for later reuse. Name must be
        alphanumeric + _/-. Use `run_saved(name, bindings={...})` to
        execute with agent-supplied variable bindings (injection-safe
        via pyoxigraph substitutions -- no string concat)."""
        ds().save_query(name, sparql, description)
        return f"saved query {name!r}"

    @mcp.tool()
    @secure("run_saved")
    def run_saved(
        name: str,
        bindings: dict | None = None,
        timeout_ms: int | None = None,
    ) -> str:
        """Run a previously-saved query. `bindings` is {var_name: value};
        values looking like http(s):// / urn: / file:// are bound as
        IRIs, everything else as literals. Returns the usual SPARQL
        binding-dict JSON."""
        return json.dumps(
            ds().run_saved(name, bindings=bindings, timeout_ms=timeout_ms),
            default=str,
        )

    # --- vectors --------------------------------------------------------

    @mcp.tool()
    @secure("embed")
    def embed(
        iris: list[str],
        vectors: list[list[float]],
        model_id: str,
        author: str | None = None,
    ) -> str:
        """Append per-IRI embedding vectors. The agent (or its caller)
        supplies pre-computed vectors -- turtlelake never loads a model.
        All vectors in one call must share the same dimension; that
        dimension becomes fixed for this dataset.

        `model_id` identifies the embedding model (e.g.
        `openai:text-embedding-3-small`). Multiple model_ids can coexist
        in the same dataset; pass `model_id=` to `vector_search` /
        `graph_rag` to disambiguate.

        Returns a short status string with the row count and the
        embedding dataset's new total."""
        n = ds().embed(iris, vectors, model_id=model_id, author=author)
        metrics.ingest_rows.labels(kind="embed").inc(n)
        _record_version()
        return (
            f"embedded {n} vectors with model {model_id!r}; "
            f"embeddings dataset now has {ds().embedding_count()} rows "
            f"(dim={ds().embedding_dim()})"
        )

    @mcp.tool()
    @secure("vector_search")
    def vector_search(
        query_vector: list[float],
        k: int = 10,
        model_id: str | None = None,
    ) -> str:
        """Approximate-nearest-neighbor search over the embeddings.
        Returns a JSON array of `{iri, distance, model_id}` ordered by
        increasing distance. Pair with `entity(iri)` to expand the
        winning IRI into a structured subgraph, or use `graph_rag` to
        do both in one call."""
        return json.dumps(
            ds().vector_search(query_vector, k=k, model_id=model_id), default=str
        )

    @mcp.tool()
    @secure("build_vector_index")
    def build_vector_index(
        index_type: str = "auto",
        metric: str = "L2",
        num_partitions: int | None = None,
        num_sub_vectors: int | None = None,
    ) -> str:
        """Build an ANN index over the embeddings dataset for sub-linear
        `vector_search`.

        `index_type='auto'` (default) picks by row count: skip below
        ~10k (brute-force is already fast), `IVF_FLAT` to ~1M, then
        `IVF_PQ`. Pass an explicit `IVF_FLAT`/`IVF_SQ`/`IVF_PQ` to
        override. Returns a status JSON
        `{action: 'built'|'skipped', index_type, rows, reason}`."""
        return json.dumps(
            ds().build_vector_index(
                index_type=index_type,
                metric=metric,
                num_partitions=num_partitions,
                num_sub_vectors=num_sub_vectors,
            ),
            default=str,
        )

    # --- maintenance ----------------------------------------------------

    @mcp.tool()
    @secure("compact")
    def compact() -> str:
        """Compact small Lance fragments on both the triples and the
        embeddings datasets. Long-lived agent projects accumulate many
        small fragments (one per write); compacting merges them into a
        smaller number of larger fragments and speeds up scans."""
        return json.dumps(ds().compact(), default=str)

    @mcp.tool()
    @secure("prune_versions")
    def prune_versions(keep_versions: int = 10) -> str:
        """Drop old Lance versions on both datasets, keeping the most
        recent `keep_versions`. Tagged versions are always retained, so
        checkpoints survive a prune. Returns a JSON summary
        `{triples: {...}, embeddings: {...}}` reporting bytes/files
        freed per side."""
        return json.dumps(ds().prune_versions(keep_versions=keep_versions), default=str)

    @mcp.tool()
    @secure("graph_rag")
    def graph_rag(
        query_vector: list[float],
        k: int = 5,
        hops: int = 1,
        model_id: str | None = None,
    ) -> str:
        """Vector retrieval + entity expansion in one call. Returns
        `{hits: [...], entities: {iri: <subgraph>, ...}}`. Use this when
        an agent has a query embedding and wants both the ranked hits
        and the surrounding facts to feed back into its prompt -- the
        canonical GraphRAG retrieval shape."""
        return json.dumps(
            ds().graph_rag(query_vector, k=k, hops=hops, model_id=model_id),
            default=str,
        )

    _install_signal_handlers()
    return mcp


def _enforce_ingest_policy(path: Path, cap_bytes: int, root: Path | None) -> None:
    """Validate an ingest target against the configured policy.

    - Path must exist.
    - Resolved path must be under `root` (if set) -- blocks prompt-injected
      attempts to read files outside the agent's working area.
    - File size must be under the byte cap.
    """
    if not path.exists():
        raise FileNotFoundError(f"ingest path does not exist: {path}")
    if root is not None:
        resolved = path.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as e:
            raise PermissionError(
                f"ingest path {resolved} escapes TURTLELAKE_INGEST_ROOT {root}. "
                "This is a security-policy rejection, not a bug."
            ) from e
    size = path.stat().st_size
    if size > cap_bytes:
        raise ValueError(
            f"ingest file too large: {size} bytes exceeds cap {cap_bytes}."
        )


def _install_signal_handlers() -> None:
    """Graceful shutdown -- mirror the parent repo's behavior by logging
    a shutdown event to audit stderr so operators can correlate."""

    def handler(signum, _frame):  # pragma: no cover
        audit_log("shutdown", {"signal": signum})
        sys.exit(0)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):
            # Signals may not be settable in non-main threads; that's fine.
            pass


def main() -> None:
    # FastMCP 3.x dropped the stdio default and added a startup banner that
    # writes to stdout -- both break MCP clients that talk JSON-RPC over the
    # process's stdout. Pin transport explicitly and silence the banner.
    build_server().run(transport="stdio", show_banner=False)


if __name__ == "__main__":
    main()
