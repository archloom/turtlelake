# Test scenarios

One table per use case from `REQUIREMENTS.md §3`. Each scenario is phrased so a test author can implement it directly. Status reflects what's on this branch today.

Legend: ✅ passing · 🟡 stub/partial · ⬜ pending · 🚫 blocked-on-dep

---

## UC-1 -- Ingest an ontology into a local KG

| # | Scenario | Preconditions | Steps | Expected | Status | Test file |
|---|---|---|---|---|---|---|
| 1.1 | TTL round-trip | empty dataset, ≥1-quad TTL file | `ds.ingest_ttl(f)` → `ds.query(SELECT *)` | all ingested quads come back | ✅ | `tests/test_ingest.py::test_dataset_roundtrip_and_sparql` |
| 1.2 | Literal with language tag | empty dataset, TTL with `"hola"@es` | ingest → scan | `object_lang == "es"` for that row | ✅ | `tests/test_ingest.py::test_quads_to_record_batch_preserves_kinds` |
| 1.3 | Literal with datatype | TTL with `"42"^^xsd:integer` | ingest → scan | `object_datatype == "http://www.w3.org/2001/XMLSchema#integer"` | ✅ | `tests/test_ingest.py::test_1_3_typed_literal_preserves_datatype` |
| 1.4 | Blank nodes preserved | TTL with `_:b1 :p :o` | ingest → scan | `object_kind == "bnode"` where applicable | ✅ | `tests/test_ingest.py` |
| 1.5 | Named graphs | N-Quads with 2 named graphs + default | ingest → scan | `graph` column matches IRI or null | ✅ | `tests/test_ingest.py::test_1_5_named_graphs_preserved_from_nquads` |
| 1.6 | Multi-file ingest = multi-version | empty dataset, two TTL files | ingest A → ingest B | 2 Lance versions, each visible via `versions()` | ⬜ | `tests/test_versioning.py` |
| 1.7 | 100k-quad TTL in <10s (NFR-2) | 100k-quad fixture | ingest, time it | elapsed < 10s on commodity hardware | ⬜ | `tests/perf_test_ingest.py` |
| 1.8 | Unknown suffix raises cleanly | `foo.xyz` file | ingest | `ValueError` with guidance on `mime_type=...` | ✅ (implicit) | `tests/test_ingest.py` |

---

## UC-2 -- Agent asks "what do I know about X?"

| # | Scenario | Preconditions | Steps | Expected | Status | Test file |
|---|---|---|---|---|---|---|
| 2.1 | 1-hop outgoing | IRI has ≥1 outgoing predicate | `ds.entity(iri, hops=1)` | all outgoing predicates in result | ✅ | `tests/test_agent_primitives.py::test_entity_one_hop...` |
| 2.2 | 1-hop incoming | IRI has ≥1 incoming edge | `entity` | `incoming` list has the subject | ✅ | `tests/test_agent_primitives.py::test_2_2_entity_reports_incoming_edges` |
| 2.3 | 2-hop follows IRI objects only | 2-level graph | `entity(iri, hops=2)` | `neighbors` contains 2nd-hop IRIs | ✅ | `tests/test_agent_primitives.py::test_entity_two_hops...` |
| 2.4 | IRI not in KG | unknown IRI | `entity` | returns `{"iri": …, "outgoing": [], "incoming": []}` (no error) | ✅ | `tests/test_agent_primitives.py::test_2_4_unknown_iri_returns_empty_shape` |
| 2.5 | hops < 1 rejected | any | `entity(iri, hops=0)` | `ValueError` | ✅ (implicit) | `tests/test_agent_primitives.py` |
| 2.6 | JSON-serializable output | typed literal in neighbors | `json.dumps(entity(...))` | no exception | ✅ | `tests/test_agent_primitives.py::test_2_6_entity_output_is_json_serializable` |

---

## UC-3 -- Agent checkpoint → risky write → rollback

| # | Scenario | Preconditions | Steps | Expected | Status | Test file |
|---|---|---|---|---|---|---|
| 3.1 | Rollback restores count | dataset with N quads | checkpoint → write +1 → rollback | new handle's count == N | ✅ | `tests/test_agent_primitives.py::test_checkpoint_then_rollback_restores_count` |
| 3.2 | Rollback does not destroy forward history | baseline + risky + tag | rollback → `versions()` | both versions still listed | ⬜ | `tests/test_versioning.py` |
| 3.3 | Two checkpoints, rollback to older | chain of 3 writes with 2 tags | rollback to older tag | sees only up to that tag's state | ⬜ | `tests/test_versioning.py` |
| 3.4 | Rollback to missing tag | any | `rollback("nope")` | clear error (not silent no-op) | ✅ | `tests/test_agent_primitives.py::test_3_4_rollback_to_missing_tag_raises_clear_error` |

---

## UC-4 -- Ship a KG inside a project / container

| # | Scenario | Preconditions | Steps | Expected | Status | Test file |
|---|---|---|---|---|---|---|
| 4.1 | `tar` → `untar` round-trip | written dataset at path A | tar path A → untar to path B → open B | identical SPARQL result vs A | ⬜ | `tests/test_portability.py` |
| 4.2 | Read-only open works | dataset, directory chmod 0555 | open + query | succeeds without write attempts | ⬜ | `tests/test_portability.py` |
| 4.3 | Path with spaces | dataset under `"some dir/kg"` | open + query | succeeds | ⬜ | `tests/test_portability.py` |

---

## UC-5 -- Reproducible agent evaluation

| # | Scenario | Preconditions | Steps | Expected | Status | Test file |
|---|---|---|---|---|---|---|
| 5.1 | Same tag → same SPARQL result | tagged dataset, 2 runs | open(tag=T) + query × 2 | byte-identical output | ⬜ | `tests/test_reproducibility.py` |
| 5.2 | New write after tag doesn't affect tagged read | tagged, then wrote more | open(tag=T) + query | sees only pre-tag state | ⬜ | `tests/test_reproducibility.py` |

---

## UC-6 -- Columnar analytics on the same file

| # | Scenario | Preconditions | Steps | Expected | Status | Test file |
|---|---|---|---|---|---|---|
| 6.1 | Polars scan with no turtlelake import | written dataset | `polars.scan_pylance(triples_path)` | schema matches §6.1 | ⬜ | `tests/test_external_readers.py` |
| 6.2 | DuckDB reads Lance directly | written dataset | `SELECT object_kind, COUNT(*) FROM read_lance('…')` | histogram of kinds | ⬜ | `tests/test_external_readers.py` |
| 6.3 | DataFusion reads Lance directly | written dataset | `datafusion.SessionContext().read_lance(...)` | same row count as `ds.count()` | ⬜ | `tests/test_external_readers.py` |

---

## UC-7 -- MCP integration

| # | Scenario | Preconditions | Steps | Expected | Status | Test file |
|---|---|---|---|---|---|---|
| 7.1 | Every declared tool is registered | fresh MCP server | enumerate tool names | set has exactly 12: `{sparql, entity, scan, ingest, insert, checkpoint, rollback, versions, refresh, diff, provenance, validate}` | ✅ | `tests/test_mcp_tools.py` |
| 7.2 | `sparql` returns JSON-parseable string | ingested dataset | call tool | `json.loads(result)` succeeds | ⬜ | `tests/test_mcp_tools.py` |
| 7.3 | `entity` returns the same dict shape as Python API | ingested dataset | call tool with IRI | keys == `{iri, outgoing, incoming, neighbors?}` | ⬜ | `tests/test_mcp_tools.py` |
| 7.4 | `ingest` followed by `sparql` returns new data | empty dataset | ingest then sparql | new data visible | ⬜ | `tests/test_mcp_tools.py` |
| 7.5 | `rollback` actually changes subsequent reads | write → checkpoint → write → rollback | call `scan` after | reflects pre-rollback state | ⬜ | `tests/test_mcp_tools.py` |
| 7.6 | Help strings present and agent-legible | all tools | inspect docstrings | non-empty, no placeholder text | ⬜ | `tests/test_mcp_tools.py` |

---

## UC-8 -- Offline / air-gapped

| # | Scenario | Preconditions | Steps | Expected | Status | Test file |
|---|---|---|---|---|---|---|
| 8.1 | No `requests` / `urllib3` / `httpx` imports in turtlelake source | clean checkout | static grep | zero matches under `src/turtlelake/` | ⬜ | `tests/test_no_egress.py` |
| 8.2 | Socket monkey-patch: block all outbound | socket patched to raise on connect | full workflow (ingest + sparql + entity) | no raise | ⬜ | `tests/test_no_egress.py` |

---

## UC-9 -- Validate writes against SHACL shapes

| # | Scenario | Preconditions | Steps | Expected | Status | Test file |
|---|---|---|---|---|---|---|
| 9.1 | Violating data → non-empty report | dataset with labelless Device + shapes file requiring label | `ds.validate(shapes)` | `conforms == False`, violation lists the IRI | 🚫 pyshacl | `tests/test_validate.py` |
| 9.2 | Conforming data → empty report | same shapes, compliant data | `validate` | `conforms == True`, no violations | 🚫 pyshacl | `tests/test_validate.py` |
| 9.3 | Missing pyshacl → actionable error | clean env without pyshacl | `validate` | `RuntimeError` naming the extra to install | ⬜ | `tests/test_validate.py` |

---

## UC-10 -- Provenance per version

| # | Scenario | Preconditions | Steps | Expected | Status | Test file |
|---|---|---|---|---|---|---|
| 10.1 | Ingest records source + author + timestamp | empty dataset | `ingest_ttl(f, source="s", author="a")` | `provenance()` list has 1 entry with those fields | ⬜ | `tests/test_provenance.py` |
| 10.2 | Mixed sequence: ingest + manual write + checkpoint | empty dataset | 3 writes with distinct `source=` | `provenance()` has 3 ordered entries | ⬜ | `tests/test_provenance.py` |
| 10.3 | Provenance survives tar copy | written dataset + log | tar → untar → open → `provenance()` | identical list | ⬜ | `tests/test_provenance.py` |
| 10.4 | Default source/author when omitted | no kwargs | `ingest_ttl(f)` | `source == basename(f)`, `author == os.getenv('USER')` or `"unknown"` | ✅ | `tests/test_provenance_and_diff.py::test_10_4_default_source_is_filename` |

---

## UC-11 -- Multi-agent shared state (`refresh`)

| # | Scenario | Preconditions | Steps | Expected | Status | Test file |
|---|---|---|---|---|---|---|
| 11.1 | Single-process refresh picks up writes | `dsA` + `dsB` opened at same path | `dsB` writes new quad → `dsA.refresh()` + `scan()` | new quad visible | ⬜ | `tests/test_multiprocess.py` |
| 11.2 | Two-process (subprocess) write + read | worker subprocess writes | main: `refresh` + `query` | new quad visible in parent | ⬜ | `tests/test_multiprocess.py` |
| 11.3 | Reader at tag is not affected by writes | `dsA` opened at tag, `dsB` writes | `dsA` re-query (no refresh) | result unchanged | ⬜ | `tests/test_multiprocess.py` |

---

## UC-12 -- Diff two versions

| # | Scenario | Preconditions | Steps | Expected | Status | Test file |
|---|---|---|---|---|---|---|
| 12.1 | Pure append → diff returns only added | v_old + 3 new quads in v_new | `diff(v_old, v_new)` | `added` has 3 rows, `removed` empty | ⬜ | `tests/test_diff.py` |
| 12.2 | Identical versions → empty diff | v_old == v_new | `diff` | both lists empty | ⬜ | `tests/test_diff.py` |
| 12.3 | Quad counts match set-difference semantics | known overlapping versions | `diff` | cardinality matches set difference | ✅ | `tests/test_provenance_and_diff.py::test_12_3_diff_cardinality_matches_set_difference` |
| 12.4 | Out-of-range version → clear error | version N doesn't exist | `diff` | raises (Lance error surfaced) | ✅ | `tests/test_provenance_and_diff.py::test_12_4_out_of_range_version_raises_clear_error` |

---

## UC-13 -- Durable agent memory

| # | Scenario | Preconditions | Steps | Expected | Status | Test file |
|---|---|---|---|---|---|---|
| 13.1 | Facts persist across re-open | write user-pref quad | close handle → reopen → `entity(user_iri)` | fact present | ⬜ | `tests/test_agent_memory.py` |
| 13.2 | Ad-hoc writes don't require checkpoint | no tag | `insert_quads` | succeeds, count increases | ⬜ | `tests/test_agent_memory.py` |

---

## UC-14 -- Publish / distribute (remote URIs)

| # | Scenario | Preconditions | Steps | Expected | Status | Test file |
|---|---|---|---|---|---|---|
| 14.1 | `file://` URI is a drop-in for a path | written dataset | `Dataset.open("file:///…")` + query | matches local-path result | ⬜ | `tests/test_remote.py` |
| 14.2 | `s3://` / `hf://` -- smoke test with `moto` or mocked fsspec | moto S3 or mocked store | `open(s3://…)` + query | same result | ⬜ (nice-to-have) | `tests/test_remote.py` |

---

## NFR-9 -- Per-tool rate limits

Application-calibrated caps so agents don't trip legitimate workflows and
destructive ops stay rare. Not a UC but a contract worth pinning.

| # | Scenario | Preconditions | Steps | Expected | Status | Test file |
|---|---|---|---|---|---|---|
| RL.1 | Read tools have high caps | default policy | inspect `DEFAULT_RATE_LIMITS` | `entity` ≥ 200, `sparql` ≥ 100, `refresh` ≥ 200 | ✅ | `tests/test_rate_limits.py::test_read_tools_have_high_caps` |
| RL.2 | Destructive tools have low caps | default policy | inspect | `rollback` ≤ 15 | ✅ | `tests/test_rate_limits.py::test_destructive_tools_have_low_caps` |
| RL.3 | Every declared MCP tool has a policy | -- | compare `tool_names()` vs `DEFAULT_RATE_LIMITS` | subset | ✅ | `tests/test_rate_limits.py::test_every_declared_tool_has_a_policy` |
| RL.4 | Per-tool env override | `TURTLELAKE_RATE_LIMIT_ENTITY=5` | `tool_rate_limit("entity")` | 5 | ✅ | `tests/test_rate_limits.py::test_per_tool_env_override` |
| RL.5 | Global env override | `TURTLELAKE_RATE_LIMIT=17` | `tool_rate_limit("any")` | 17 | ✅ | `tests/test_rate_limits.py::test_global_env_override` |
| RL.6 | Per-tool beats global | both set | `tool_rate_limit("entity")` | per-tool value | ✅ | `tests/test_rate_limits.py::test_per_tool_beats_global` |
| RL.7 | Zero disables the limit | `TURTLELAKE_RATE_LIMIT_ENTITY=0` | 10k calls | no raise | ✅ | `tests/test_rate_limits.py::test_zero_disables_limit` |
| RL.8 | Invalid env raises at resolution | `=not-a-number` | `tool_rate_limit` | `RuntimeError` | ✅ | `tests/test_rate_limits.py::test_invalid_env_raises` |
| RL.9 | Rollback strict cap enforced | default | exceed cap | `RateLimitExceeded` | ✅ | `tests/test_rate_limits.py::test_rollback_rate_limit_trips_at_cap` |
| RL.10 | 100-entity call burst passes | default | loop 100x `entity` | no raise (would fail at flat 30/min) | ✅ | `tests/test_rate_limits.py::test_entity_rate_limit_does_not_trip_on_normal_browsing` |

---

## UC-15 -- Privacy-first on-device

Covered by UC-8 (no egress) plus manual inspection -- no DB-side encryption is attempted; OS-level (LUKS/APFS) is the boundary.

| # | Scenario | Preconditions | Steps | Expected | Status | Test file |
|---|---|---|---|---|---|---|
| 15.1 | No network calls in a full workflow | socket patched | ingest + sparql + entity + checkpoint + rollback + diff + provenance | no raise, all pass | ⬜ | `tests/test_no_egress.py` |
| 15.2 | No environment variables leaked into provenance | env var `SECRET=…` set | `ingest_ttl(f)` → `provenance()` | no field contains `SECRET`'s value | ⬜ | `tests/test_provenance.py` |

---

## Priority buckets

**Must-green before calling MVP done (acceptance items 4 + 6 + 8 + 9 + 10):**
1.1, 1.2, 1.4, 2.1, 2.3, 3.1, 7.1, 7.2, 7.3, 7.4, 10.1, 10.2, 11.1, 12.1, 12.2, 15.1

**Should-green (quality):** 1.5, 1.6, 2.4, 2.6, 3.2–3.4, 4.1, 5.1–5.2, 7.5–7.6, 12.3–12.4, 13.1–13.2

**Nice-to-green (later):** 1.3, 1.7 (perf), 6.1–6.3 (external readers), 8.1–8.2, 9.x (needs pyshacl extra), 14.x (remote storage)

---

## How to read a row

- **Status** is point-in-time. When a scenario goes green, flip the box and link to the commit SHA.
- If a scenario is in the "must-green" bucket but turns out to be blocked on a dependency upgrade or a dep we rejected, move it to out-of-scope in `REQUIREMENTS.md §4` and remove the row here -- don't leave zombie scenarios.
- New use cases added to `REQUIREMENTS.md §3` must come with at least one scenario added here.
