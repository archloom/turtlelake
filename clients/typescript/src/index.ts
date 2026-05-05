/**
 * TypeScript client for the turtlelake MCP server.
 *
 * The Python `turtlelake[mcp]` package ships a CLI (`turtlelake-mcp`)
 * that speaks JSON-RPC over stdio. This client spawns that process,
 * negotiates an MCP session, and exposes typed methods for every
 * registered tool — graph_rag, vector_search, sparql, ingest,
 * checkpoint, rollback, and the rest.
 *
 * Why JSON-RPC over stdio rather than a Rust FFI port: the MCP server
 * is the canonical agent-facing surface, and Claude Code / Cursor
 * already speak it. A native client doesn't gain you anything you
 * can't get from spawning the server.
 *
 * Quickstart:
 *
 *   import { TurtleLake } from "@turtlelake/client";
 *   const kg = await TurtleLake.spawn({ storePath: "./my_kg" });
 *   await kg.ingest("./ontology.ttl");
 *   await kg.embed(iris, vectors, "openai:text-embedding-3-small");
 *   const out = await kg.graphRag(queryVector, { k: 5, hops: 1 });
 *   await kg.close();
 */

import { spawn, ChildProcessWithoutNullStreams } from "node:child_process";
import { EventEmitter } from "node:events";

// ── JSON-RPC plumbing ─────────────────────────────────────────────

interface JsonRpcRequest {
  jsonrpc: "2.0";
  id: number;
  method: string;
  params?: unknown;
}

interface JsonRpcResponse {
  jsonrpc: "2.0";
  id: number;
  result?: unknown;
  error?: { code: number; message: string; data?: unknown };
}

interface JsonRpcNotification {
  jsonrpc: "2.0";
  method: string;
  params?: unknown;
}

type IncomingMessage = JsonRpcResponse | JsonRpcNotification;

// ── public types ──────────────────────────────────────────────────

export interface SpawnOptions {
  /** Local directory where the dataset lives. Maps to TURTLELAKE_PATH. */
  storePath: string;
  /** Override the binary path. Defaults to `turtlelake-mcp` on PATH. */
  binPath?: string;
  /** Extra env vars passed to the child process. Merged with process.env. */
  env?: Record<string, string>;
  /** Dial-up for verbose diagnostics on stderr (default false). */
  debug?: boolean;
}

export interface VectorHit {
  iri: string;
  distance: number;
  model_id: string;
}

export interface EntityResult {
  iri: string;
  outgoing: { predicate: string; object: unknown }[];
  incoming: { predicate: string; subject: string }[];
  neighbors?: Record<string, EntityResult>;
  similar?: VectorHit[];
}

export interface GraphRagResult {
  hits: VectorHit[];
  entities: Record<string, EntityResult>;
}

export interface VersionsResult {
  versions: { version: number; timestamp: string }[];
  tags: string[];
}

export interface BuildIndexResult {
  action: "built" | "skipped";
  index_type: string | null;
  rows: number;
  reason: string;
}

// ── client ────────────────────────────────────────────────────────

export class TurtleLake extends EventEmitter {
  #child: ChildProcessWithoutNullStreams;
  #pending = new Map<number, (msg: JsonRpcResponse) => void>();
  #buf = "";
  #nextId = 0;
  #debug: boolean;
  #closed = false;

  private constructor(child: ChildProcessWithoutNullStreams, debug: boolean) {
    super();
    this.#child = child;
    this.#debug = debug;
    child.stdout.setEncoding("utf-8");
    child.stdout.on("data", (chunk: string) => this.#onStdout(chunk));
    child.stderr.on("data", (chunk: Buffer) => {
      if (this.#debug) process.stderr.write(`[turtlelake-mcp] ${chunk}`);
    });
    child.on("exit", (code, signal) => {
      this.#closed = true;
      this.emit("exit", { code, signal });
      // Reject any in-flight requests so callers don't hang.
      for (const resolve of this.#pending.values()) {
        resolve({
          jsonrpc: "2.0",
          id: -1,
          error: {
            code: -32000,
            message: `turtlelake-mcp exited (code=${code}, signal=${signal})`,
          },
        });
      }
      this.#pending.clear();
    });
  }

  /** Spawn `turtlelake-mcp` and complete the MCP `initialize` handshake. */
  static async spawn(opts: SpawnOptions): Promise<TurtleLake> {
    const bin = opts.binPath ?? "turtlelake-mcp";
    const env = {
      ...process.env,
      ...(opts.env ?? {}),
      TURTLELAKE_PATH: opts.storePath,
    };
    const child = spawn(bin, [], { env, stdio: ["pipe", "pipe", "pipe"] });
    const client = new TurtleLake(child, opts.debug ?? false);
    // Standard MCP handshake — the server does not respond to tool
    // calls until after `initialize` succeeds.
    await client.#rpc("initialize", {
      protocolVersion: "2024-11-05",
      capabilities: {},
      clientInfo: { name: "@turtlelake/client", version: "0.0.1" },
    });
    await client.#notify("notifications/initialized", {});
    return client;
  }

  /** Cleanly shut down the spawned MCP server. */
  async close(): Promise<void> {
    if (this.#closed) return;
    this.#child.stdin.end();
    return new Promise((resolve) => {
      const timer = setTimeout(() => {
        this.#child.kill("SIGTERM");
        resolve();
      }, 2000);
      this.#child.once("exit", () => {
        clearTimeout(timer);
        resolve();
      });
    });
  }

  // ── tool wrappers (typed) ─────────────────────────────────────

  /** Free-text guide to the canonical agent workflow. */
  guide(): Promise<string> {
    return this.callText("guide");
  }

  /** Runtime-introspected schema as a JSON-parsed object. */
  schema(args: { topClasses?: number; topPredicates?: number } = {}): Promise<unknown> {
    return this.callJson("schema", {
      top_classes: args.topClasses,
      top_predicates: args.topPredicates,
    });
  }

  /** Run a SPARQL 1.1 query. Returns the parsed bindings array. */
  sparql(query: string, opts: { timeoutMs?: number } = {}): Promise<unknown[]> {
    return this.callJson("sparql", {
      query,
      timeout_ms: opts.timeoutMs,
    }) as Promise<unknown[]>;
  }

  /**
   * Pull the structured neighborhood of `iri`. `similar > 0` appends
   * vector-nearest IRIs (requires embeddings).
   */
  entity(
    iri: string,
    opts: { hops?: number; similar?: number; modelId?: string } = {},
  ): Promise<EntityResult> {
    return this.callJson("entity", {
      iri,
      hops: opts.hops,
      similar: opts.similar,
      model_id: opts.modelId,
    }) as Promise<EntityResult>;
  }

  /** Append a TTL/N-Quads/JSON-LD/RDF-XML file to the dataset. */
  ingest(path: string, opts: { source?: string; author?: string; graph?: string } = {}): Promise<string> {
    return this.callText("ingest", {
      path,
      source: opts.source,
      author: opts.author,
      graph: opts.graph,
    });
  }

  /** Append a TTL string (agent-memory entry point). */
  insert(turtle: string, opts: { source?: string; author?: string; graph?: string } = {}): Promise<string> {
    return this.callText("insert", {
      turtle,
      source: opts.source,
      author: opts.author,
      graph: opts.graph,
    });
  }

  /** Tag both Lance datasets at their current versions; crash-safe. */
  checkpoint(name: string, opts: { author?: string } = {}): Promise<string> {
    return this.callText("checkpoint", { name, author: opts.author });
  }

  /** Restore the dataset (triples + embeddings) to a tagged version. */
  rollback(name: string): Promise<string> {
    return this.callText("rollback", { name });
  }

  /** Append per-IRI vectors. Caller supplies pre-computed floats. */
  embed(
    iris: string[],
    vectors: number[][],
    modelId: string,
    opts: { author?: string } = {},
  ): Promise<string> {
    return this.callText("embed", {
      iris,
      vectors,
      model_id: modelId,
      author: opts.author,
    });
  }

  /** ANN search over the embeddings dataset. */
  vectorSearch(
    queryVector: number[],
    opts: { k?: number; modelId?: string } = {},
  ): Promise<VectorHit[]> {
    return this.callJson("vector_search", {
      query_vector: queryVector,
      k: opts.k,
      model_id: opts.modelId,
    }) as Promise<VectorHit[]>;
  }

  /** Vector retrieval + entity expansion in one call. */
  graphRag(
    queryVector: number[],
    opts: { k?: number; hops?: number; modelId?: string } = {},
  ): Promise<GraphRagResult> {
    return this.callJson("graph_rag", {
      query_vector: queryVector,
      k: opts.k,
      hops: opts.hops,
      model_id: opts.modelId,
    }) as Promise<GraphRagResult>;
  }

  /** Build (or skip) the ANN index based on row count. */
  buildVectorIndex(
    opts: {
      indexType?: "auto" | "IVF_FLAT" | "IVF_SQ" | "IVF_PQ";
      metric?: "L2" | "cosine" | "dot";
      numPartitions?: number;
      numSubVectors?: number;
    } = {},
  ): Promise<BuildIndexResult> {
    return this.callJson("build_vector_index", {
      index_type: opts.indexType,
      metric: opts.metric,
      num_partitions: opts.numPartitions,
      num_sub_vectors: opts.numSubVectors,
    }) as Promise<BuildIndexResult>;
  }

  /** Compact small Lance fragments on both datasets. */
  compact(): Promise<unknown> {
    return this.callJson("compact");
  }

  /** Drop old Lance versions, preserving tagged ones. */
  pruneVersions(opts: { keepVersions?: number } = {}): Promise<unknown> {
    return this.callJson("prune_versions", {
      keep_versions: opts.keepVersions,
    });
  }

  /** Lance versions and tags as JSON. */
  versions(): Promise<VersionsResult> {
    return this.callJson("versions") as Promise<VersionsResult>;
  }

  /** Quad-level provenance log. */
  provenance(): Promise<unknown[]> {
    return this.callJson("provenance") as Promise<unknown[]>;
  }

  // ── escape hatch: call any tool by name ──────────────────────

  /** Invoke a tool whose return value is text (e.g. status strings). */
  async callText(tool: string, args: Record<string, unknown> = {}): Promise<string> {
    const res = (await this.#callTool(tool, args)) as { content?: { text?: string }[] };
    const text = res?.content?.[0]?.text ?? "";
    return text;
  }

  /**
   * Invoke a tool whose return value is JSON-encoded text. Parsed for
   * you. The turtlelake `@secure` decorator wraps tool exceptions as
   * `{"error": "..."}` payloads (rate-limit hits, input rejections,
   * runtime errors); we detect that envelope and convert to a
   * rejected promise so callers can `try/catch` naturally.
   */
  async callJson(tool: string, args: Record<string, unknown> = {}): Promise<unknown> {
    const text = await this.callText(tool, args);
    if (!text) return null;
    let parsed: unknown;
    try {
      parsed = JSON.parse(text);
    } catch {
      return text;
    }
    if (
      parsed &&
      typeof parsed === "object" &&
      !Array.isArray(parsed) &&
      "error" in (parsed as Record<string, unknown>)
    ) {
      const obj = parsed as { error?: unknown };
      throw new Error(`${tool} failed: ${String(obj.error)}`);
    }
    return parsed;
  }

  // ── internals ────────────────────────────────────────────────

  #callTool(name: string, args: Record<string, unknown>): Promise<unknown> {
    // FastMCP / MCP SDK convention: tool calls go through `tools/call`.
    // Strip undefined values so we don't surface them to the server.
    const filtered: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(args)) {
      if (v !== undefined) filtered[k] = v;
    }
    return this.#rpc("tools/call", { name, arguments: filtered });
  }

  #rpc(method: string, params: unknown): Promise<unknown> {
    if (this.#closed) {
      return Promise.reject(new Error("turtlelake-mcp client is closed"));
    }
    const id = ++this.#nextId;
    const payload: JsonRpcRequest = { jsonrpc: "2.0", id, method, params };
    return new Promise((resolve, reject) => {
      this.#pending.set(id, (response) => {
        if (response.error) {
          reject(new Error(`${method} failed: ${response.error.message}`));
        } else {
          resolve(response.result);
        }
      });
      this.#child.stdin.write(JSON.stringify(payload) + "\n");
    });
  }

  #notify(method: string, params: unknown): Promise<void> {
    const payload: JsonRpcNotification = { jsonrpc: "2.0", method, params };
    this.#child.stdin.write(JSON.stringify(payload) + "\n");
    return Promise.resolve();
  }

  #onStdout(chunk: string): void {
    this.#buf += chunk;
    let nl: number;
    // FastMCP delivers one JSON message per line.
    while ((nl = this.#buf.indexOf("\n")) >= 0) {
      const line = this.#buf.slice(0, nl).trim();
      this.#buf = this.#buf.slice(nl + 1);
      if (!line) continue;
      let msg: IncomingMessage;
      try {
        msg = JSON.parse(line) as IncomingMessage;
      } catch {
        if (this.#debug) process.stderr.write(`[turtlelake-mcp] non-json: ${line}\n`);
        continue;
      }
      if ("id" in msg && msg.id != null) {
        const handler = this.#pending.get(msg.id);
        if (handler) {
          this.#pending.delete(msg.id);
          handler(msg);
        }
      } else {
        // Notifications are not yet routed to listeners, but exposed
        // via `client.on('notification', ...)` for advanced callers.
        this.emit("notification", msg);
      }
    }
  }
}
