import json
from pathlib import Path

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from mcp_server import server
from mcp_server.results import ArtifactStore, flatten_actions, page_result


def data(result):
    return json.loads("".join(block.text for block in result.content if hasattr(block, "text")))


async def test_small_catalog_and_exact_schema():
    async with Client(server.mcp) as client:
        tools = await client.list_tools()
        assert {tool.name for tool in tools} == {
            "list_instances", "browse_tools", "get_tool_schema", "search_tools", "call_tool"}
        definition = data(await client.call_tool("get_tool_schema", {"name": "get_pipeline_state"}))
        schema = definition["inputSchema"]
        assert "instance_id" in schema["required"]
        assert "event_id" in schema["required"]


async def test_proxy_validates_arguments_and_routes_to_explicit_instance(monkeypatch):
    calls = []

    def fake_call(instance_id, method, params):
        calls.append((instance_id, method, params))
        return {"loaded": True}

    monkeypatch.setattr(server.bridge, "call", fake_call)
    async with Client(server.mcp) as client:
        result = await client.call_tool("call_tool", {
            "name": "get_capture_status", "arguments": {"instance_id": "selected"}
        })
        assert not result.is_error
        assert calls == [("selected", "get_capture_status", {})]
        with pytest.raises(ToolError):
            await client.call_tool("call_tool", {"name": "get_capture_status", "arguments": {}})
        with pytest.raises(ToolError):
            await client.call_tool("call_tool", {"name": "call_tool", "arguments": {}})


async def test_search_is_paginated_and_does_not_hide_truncation():
    async with Client(server.mcp) as client:
        first = data(await client.call_tool("search_tools", {"pattern": ".*", "limit": 3}))
        assert first["returned"] == 3 and first["total_matches"] > 3 and first["truncated"]
        assert first["next_offset"] == 3
        second = data(await client.call_tool("search_tools", {"pattern": ".*", "offset": 3, "limit": 50}))
        assert not {t["name"] for t in first["tools"]} & {t["name"] for t in second["tools"]}
        assert first["returned"] + second["returned"] == first["total_matches"]
        assert not second["truncated"] and second["next_offset"] is None


async def test_directory_covers_every_registered_operation_once():
    from mcp_server.catalog import TOOL_CATEGORY
    async with Client(server.mcp) as client:
        directory = data(await client.call_tool("browse_tools", {}))
        assert directory["complete"] and len(directory["categories"]) == 6
        names = []
        for category in directory["categories"]:
            result = data(await client.call_tool("browse_tools", {"category": category["name"], "limit": 50}))
            assert result["returned"] == category["count"]
            assert not result["truncated"]
            assert all("inputSchema" not in t for t in result["tools"])
            names.extend(t["name"] for t in result["tools"])
        assert len(names) == len(set(names)) == directory["total_tools"]
        assert set(names) == set(TOOL_CATEGORY)


@pytest.mark.parametrize("pattern", ["get_texture_data", "^get_texture_data$"])
async def test_exact_name_search_is_reachable_and_ranked_first(pattern):
    async with Client(server.mcp) as client:
        result = data(await client.call_tool("search_tools", {"pattern": pattern}))
        assert result["tools"][0]["name"] == "get_texture_data"
        schemas = data(await client.call_tool("search_tools", {"pattern": ".*", "detail": "schema"}))
        assert schemas["returned"] == 3 and schemas["truncated"]
        assert "inputSchema" in schemas["tools"][0]


@pytest.mark.parametrize("name,arguments", [
    ("search_tools", {"pattern": "["}), ("browse_tools", {"category": "unknown"}),
    ("get_tool_schema", {"name": "unknown"}), ("browse_tools", {"offset": -1}),
    ("search_tools", {"pattern": ".*", "limit": 51}),
])
async def test_discovery_errors_are_not_empty_catalogs(name, arguments):
    async with Client(server.mcp) as client:
        with pytest.raises(ToolError):
            await client.call_tool(name, arguments)


async def test_cross_capture_pairing_and_missing_fields_do_not_claim_equality(monkeypatch):
    def call(instance_id, method, params):
        if method == "get_capture_status":
            return {"loaded": True, "filename": instance_id + ".rdc"}
        if method == "get_pipeline_state":
            return {"api": "D3D11", "unavailable": {"shaders": "failed"}}
        return {"status": "unsupported", "complete": False}
    monkeypatch.setattr(server.bridge, "call", call)
    async with Client(server.mcp) as client:
        result = data(await client.call_tool("call_tool", {"name": "compare_capture_states", "arguments": {
            "instance_id": "A", "event_id": 3, "other_instance_id": "B", "other_event_id": 90,
            "fields": ["render_targets"]}}))
        assert result["equal"] is None and not result["complete"]
        assert result["before"]["event_id"] == 3 and result["after"]["event_id"] == 90
        assert result["before"]["capture"]["filename"] == "A.rdc"


async def test_constant_byte_followup_has_complete_instance_arguments(monkeypatch):
    monkeypatch.setattr(server.bridge, "call", lambda *_: {"raw_read": {
        "tool": "get_buffer_contents", "arguments": {"event_id": 5, "resource_id": "123", "offset": 256, "length": 16}}})
    async with Client(server.mcp) as client:
        result = data(await client.call_tool("call_tool", {"name": "get_constant_buffer", "arguments": {
            "instance_id": "chosen", "event_id": 5, "stage": "pixel"}}))
        assert result["raw_read"]["arguments"] == {
            "instance_id": "chosen", "event_id": 5, "resource_id": "123", "offset": 256, "length": 16}


def test_pagination_flattens_children_and_selects_fields():
    actions = [{"event_id": 1, "name": "pass", "children": [
        {"event_id": 2, "name": "draw", "flags": ["Drawcall"]},
        {"event_id": 3, "name": "draw2"},
    ]}]
    result = page_result({"actions": list(flatten_actions(actions))}, "actions", 1, 1,
                         ["event_id", "parent_event_id"])
    assert result["actions"] == [{"event_id": 2, "parent_event_id": 1}]
    assert result["total"] == 3 and result["next_offset"] == 2
    assert page_result({"actions": []}, "actions", 0, 50)["next_offset"] is None


def test_large_result_preserved_and_read_in_bounded_chunks(tmp_path):
    store = ArtifactStore(tmp_path)
    value = {"shader": "中文 shader\n" * 6000}
    result = store.bound(value)
    assert result["stored"] and "shader" not in result
    assert json.loads(Path(result["path"]).read_text("utf-8")) == value
    first = store.read(result["artifact_id"], 0, 100)
    second = store.read(result["artifact_id"], first["next_offset"], 100)
    assert len(first["text"]) == 100
    assert first["text"] + second["text"] == Path(result["path"]).read_text("utf-8")[:200]
    with pytest.raises(ValueError):
        store.read("../../config.json", 0, 100)


@pytest.mark.parametrize("arguments", [
    {"name": "get_draw_calls", "arguments": {"instance_id": "x", "limit": 201}},
    {"name": "get_draw_calls", "arguments": {"instance_id": "x", "offset": -1}},
    {"name": "get_pipeline_states", "arguments": {"instance_id": "x", "event_ids": []}},
    {"name": "get_buffer_contents", "arguments": {
        "instance_id": "x", "event_id": 1, "resource_id": "1", "output": "inline", "length": 0}},
    {"name": "get_api_events", "arguments": {"instance_id": "x", "event_id_min": 5, "event_id_max": 1}},
    {"name": "get_api_events", "arguments": {"instance_id": "x", "event_id_min": 0, "event_id_max": 5000}},
    {"name": "debug_shader", "arguments": {
        "instance_id": "x", "event_id": 1, "stage": "vertex", "invocation": {"vertex": [0], "index": 0}}},
    {"name": "debug_shader", "arguments": {
        "instance_id": "x", "event_id": 1, "stage": "compute", "invocation": {"group": [0], "thread": [0, 0, 0]}}},
])
async def test_invalid_or_unbounded_requests_fail(arguments):
    async with Client(server.mcp) as client:
        with pytest.raises(ToolError):
            await client.call_tool("call_tool", arguments)
