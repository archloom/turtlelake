"""The MCP tool set is REQUIREMENTS.md §8 acceptance #6. If a tool drops,
this test fails loud. No FastMCP mocking — we boot the server and ask it
what it registered."""

import asyncio

import pytest

from turtlelake.mcp_server import build_server, tool_names


def test_server_registers_the_declared_tool_set(tmp_path):
    pytest.importorskip("fastmcp")
    server = build_server(store_path=tmp_path / "kg")
    tools = asyncio.run(server.list_tools())
    registered = {t.name for t in tools}
    expected = set(tool_names())
    missing = expected - registered
    extra = registered - expected
    assert not missing, f"missing tools: {missing}"
    assert not extra, f"unexpected tools: {extra}"


def test_tool_names_is_stable_and_complete():
    names = tool_names()
    # Contract: 25 tools covering the REQUIREMENTS §8 acceptance list,
    # including the GraphRAG vector surface (embed / vector_search /
    # graph_rag / build_vector_index) and the maintenance ops
    # (compact / prune_versions).
    assert len(names) == len(set(names)), "duplicate names in tool list"
    for category in (
        {"guide", "schema", "sources"},                     # discovery
        {"sparql", "entity", "scan", "explain"},            # read
        {"ingest", "insert", "checkpoint", "rollback"},     # write
        {"versions", "refresh", "diff"},                    # versioning
        {"provenance"},                                     # audit
        {"validate"},                                       # quality
        {"dump"},                                           # export
        {"save_query", "run_saved"},                        # saved queries
        {"embed", "vector_search", "graph_rag",
         "build_vector_index"},                             # vectors
        {"compact", "prune_versions"},                      # maintenance
    ):
        assert category <= set(names), f"missing category members: {category}"
    assert len(names) == 25
