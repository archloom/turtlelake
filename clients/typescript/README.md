# @turtlelake/client

TypeScript client for the [turtlelake](../..) MCP server. Spawn
`turtlelake-mcp` over stdio and call its 25 tools -- `graph_rag`,
`vector_search`, `sparql`, `ingest`, `checkpoint`, `rollback`, and
the rest -- from any Node 20+ runtime.

## Why a client wrapper rather than a Rust port

The MCP server is the canonical agent-facing surface. Claude Code,
Cursor, and other MCP-capable runtimes already speak it. A native
Rust port would duplicate the surface and lock you into a binary
distribution; this client gives you the same capabilities by reusing
the Python implementation as a subprocess.

## Install

```bash
npm install @turtlelake/client
pip install "turtlelake[mcp]"     # provides the turtlelake-mcp binary
```

The Python package puts a `turtlelake-mcp` script on `PATH`. The
client spawns that binary by default; pass `binPath` to override.

## Quickstart

```ts
import { TurtleLake } from "@turtlelake/client";

const kg = await TurtleLake.spawn({ storePath: "./my_kg" });

// Ingest an ontology.
await kg.ingest("./examples/ontology.ttl");

// Embed every entity (you compute the vectors).
await kg.embed(iris, vectors, "openai:text-embedding-3-small");

// Build the ANN index. Below ~10k vectors it is a no-op (auto policy).
await kg.buildVectorIndex();

// Retrieve by meaning, get back facts.
const out = await kg.graphRag(queryVector, { k: 5, hops: 1 });
for (const hit of out.hits) {
  console.log(hit.iri, hit.distance);
  console.log(out.entities[hit.iri].outgoing);
}

// Crash-safe checkpoint across triples + embeddings.
await kg.checkpoint("baseline");

await kg.close();
```

## API

| Method | Returns | Notes |
| --- | --- | --- |
| `TurtleLake.spawn(opts)` | `Promise<TurtleLake>` | Spawns + handshakes |
| `kg.guide()` | `string` | Canonical agent workflow |
| `kg.schema(opts?)` | `unknown` | Classes, predicates, namespaces |
| `kg.sparql(query, opts?)` | `unknown[]` | SPARQL 1.1 bindings |
| `kg.entity(iri, opts?)` | `EntityResult` | N-hop subgraph + optional `similar` |
| `kg.ingest(path, opts?)` | `string` | Append a TTL/N-Quads/JSON-LD/RDF-XML file |
| `kg.insert(turtle, opts?)` | `string` | Append a TTL string |
| `kg.checkpoint(name, opts?)` | `string` | Tag both datasets atomically |
| `kg.rollback(name)` | `string` | Restore both datasets |
| `kg.embed(iris, vectors, modelId, opts?)` | `string` | Append per-IRI vectors |
| `kg.vectorSearch(qVec, opts?)` | `VectorHit[]` | ANN search |
| `kg.graphRag(qVec, opts?)` | `GraphRagResult` | Vector hits + structural expansion |
| `kg.buildVectorIndex(opts?)` | `BuildIndexResult` | Auto / IVF_FLAT / IVF_PQ |
| `kg.compact()` | `unknown` | Merge small fragments |
| `kg.pruneVersions(opts?)` | `unknown` | Drop old versions, keep tagged |
| `kg.versions()` | `VersionsResult` | Versions + tags |
| `kg.provenance()` | `unknown[]` | Per-write audit log |
| `kg.callText(tool, args)` | `string` | Escape hatch: any tool |
| `kg.callJson(tool, args)` | `unknown` | Escape hatch with JSON parsing |
| `kg.close()` | `void` | Clean shutdown |

## Running the test

```bash
cd clients/typescript
npm install
pip install "turtlelake[mcp]"
npm test
```

The integration test is skipped automatically if `turtlelake-mcp` is
not on `PATH`.

## Notes

- All tool calls go through MCP's `tools/call` JSON-RPC method. The
  client emits `notification` events for any server-sent
  notifications you want to handle.
- Errors from the server (e.g. SPARQL parse failures, missing
  embeddings) come back as rejected promises with the redacted error
  message the server already sanitized.
- Each `TurtleLake.spawn(...)` starts a fresh `turtlelake-mcp`
  process. For long-running services you'll want one shared client;
  for short scripts, spawning per-task is fine.
