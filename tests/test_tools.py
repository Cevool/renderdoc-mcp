import json
from pathlib import Path

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from mcp_server import server
from mcp_server.results import ArtifactStore, flatten_actions, page_result


async def test_advertises_only_three_tools_and_search_returns_schemas():
    async with Client(server.mcp) as client:
        tools = await client.list_tools()
        assert {tool.name for tool in tools} == {"list_instances", "search_tools", "call_tool"}
        result = await client.call_tool("search_tools", {"pattern": r"get_pipeline_state\b"})
        text = "".join(block.text for block in result.content if hasattr(block, "text"))
        definitions = json.loads(text)
        assert len(definitions) == 1
        schema = definitions[0].get("inputSchema", definitions[0].get("input_schema"))
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


async def test_search_result_count_is_bounded():
    async with Client(server.mcp) as client:
        result = await client.call_tool("search_tools", {"pattern": ".*"})
        definitions = json.loads("".join(block.text for block in result.content if hasattr(block, "text")))
        assert len(definitions) == 3


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
])
async def test_invalid_or_unbounded_requests_fail(arguments):
    async with Client(server.mcp) as client:
        with pytest.raises(ToolError):
            await client.call_tool("call_tool", arguments)
