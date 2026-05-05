/**
 * Integration test: spawn the real `turtlelake-mcp` and exercise the
 * full end-to-end shape (ingest → embed → graph_rag → checkpoint).
 *
 * Requires turtlelake-mcp on PATH. The test is skipped if the binary
 * is not available — this lets `npm test` pass on a clean clone
 * without forcing a Python install.
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { execSync } from "node:child_process";

import { TurtleLake } from "../src/index.ts";

function turtlelakeMcpAvailable(): string | null {
  try {
    return execSync("which turtlelake-mcp", { stdio: ["ignore", "pipe", "ignore"] })
      .toString()
      .trim();
  } catch {
    return null;
  }
}

const bin = turtlelakeMcpAvailable();
const skip = bin === null;
const skipReason = skip ? "turtlelake-mcp not on PATH" : undefined;

test("client wraps the MCP server end-to-end", { skip: skipReason }, async () => {
  const dir = mkdtempSync(join(tmpdir(), "tlk-ts-"));
  const ttl = join(dir, "seed.ttl");
  writeFileSync(
    ttl,
    `@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
     @prefix ex: <https://ex.org/> .
     ex:A rdfs:label "Stratix 10" .
     ex:B rdfs:label "Agilex 7" .`,
  );

  const kg = await TurtleLake.spawn({ storePath: join(dir, "kg") });
  try {
    // Minimal ingest + sparql round trip.
    const ingestResult = await kg.ingest(ttl);
    assert.match(ingestResult, /ingested/);
    const rows = await kg.sparql(
      "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> SELECT ?s ?l WHERE { ?s rdfs:label ?l }",
    );
    assert.equal(rows.length, 2);

    // Vector layer + GraphRAG.
    await kg.embed(
      ["https://ex.org/A", "https://ex.org/B"],
      [[1.0, 0.0], [0.0, 1.0]],
      "test:m1",
    );
    const hits = await kg.vectorSearch([1.0, 0.0], { k: 2 });
    assert.equal(hits[0]?.iri, "https://ex.org/A");

    const graph = await kg.graphRag([1.0, 0.0], { k: 1, hops: 1 });
    assert.equal(graph.hits.length, 1);
    assert.ok(graph.entities[graph.hits[0]!.iri]);

    // Build-index policy: at this scale should skip.
    const idx = await kg.buildVectorIndex();
    assert.equal(idx.action, "skipped");

    // Crash-safe checkpoint.
    await kg.checkpoint("baseline");
    const v = await kg.versions();
    assert.ok(v.tags.includes("baseline"));
  } finally {
    await kg.close();
    rmSync(dir, { recursive: true, force: true });
  }
});

test("client surfaces tool errors as rejected promises", { skip: skipReason }, async () => {
  const dir = mkdtempSync(join(tmpdir(), "tlk-ts-err-"));
  const kg = await TurtleLake.spawn({ storePath: join(dir, "kg") });
  try {
    // Querying without ingesting should fail at the dataset layer.
    await assert.rejects(
      kg.sparql("SELECT * WHERE { ?s ?p ?o }"),
      /No triples dataset|error/i,
    );
  } finally {
    await kg.close();
    rmSync(dir, { recursive: true, force: true });
  }
});
