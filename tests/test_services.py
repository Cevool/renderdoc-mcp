"""Service regressions tested with deterministic controller data, without a GPU."""

import enum
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

rd = types.ModuleType("renderdoc")
rd.ActionFlags = enum.IntFlag("ActionFlags", {
    name: 1 << i for i, name in enumerate([
        "Drawcall", "Dispatch", "Clear", "PushMarker", "PopMarker", "SetMarker", "Present",
        "Copy", "Resolve", "GenMips", "PassBoundary", "Indexed", "Instanced", "Auto", "Indirect",
        "ClearColor", "ClearDepthStencil", "BeginPass", "EndPass",
    ])
})
rd.Subresource = SimpleNamespace
rd.ResourceId = SimpleNamespace(Null=lambda: 0)
rd.ShaderStage = SimpleNamespace(**{name: i for i, name in enumerate(
    ["Vertex", "Hull", "Domain", "Geometry", "Pixel", "Compute"])})
sys.modules["renderdoc"] = rd

from renderdoc_extension.renderdoc_facade import RenderDocFacade  # noqa: E402
from renderdoc_extension.services.resource_service import ResourceService  # noqa: E402
from renderdoc_extension.services.pipeline_service import PipelineService  # noqa: E402
from renderdoc_extension.utils.diff import differences  # noqa: E402
from renderdoc_extension.utils.parsers import Parsers  # noqa: E402
from renderdoc_extension.utils.serializers import Serializers  # noqa: E402


def action(event, name, flags, children=None):
    return SimpleNamespace(eventId=event, actionId=event, GetName=lambda _: name,
                           flags=flags, children=children or [], numIndices=3, numInstances=1)


def test_nested_marker_filter_preserves_ancestry_and_leaf_markers():
    draw = action(3, "Draw", rd.ActionFlags.Drawcall)
    camera = action(2, "Camera.Render", rd.ActionFlags.PushMarker, [draw])
    root = action(1, "Frame", rd.ActionFlags.PushMarker, [camera])
    value = Serializers.serialize_actions([root], None, True, marker_filter="Camera.Render")
    assert value[0]["children"][0]["children"][0]["event_id"] == 3
    leaf = action(4, "Label", rd.ActionFlags.SetMarker)
    assert Serializers.serialize_actions([leaf], None, True)[0]["event_id"] == 4
    assert Serializers.serialize_actions([root], None, True, marker_filter="missing") == []
    assert Serializers.serialize_actions([root], None, True, exclude_markers=["Frame"]) == []


def test_only_actions_range_and_flags_compose():
    draw = action(3, "Draw", rd.ActionFlags.Drawcall)
    compute = action(4, "Compute", rd.ActionFlags.Dispatch)
    root = action(1, "Pass", rd.ActionFlags.PushMarker, [draw, compute])
    value = Serializers.serialize_actions([root], None, True, only_actions=True,
                                          event_id_min=4, flags_filter=["Dispatch"])
    assert [item["event_id"] for item in value] == [4]


@pytest.mark.parametrize("fail", [False, True])
def test_replay_event_restored_even_when_callback_fails(fail):
    events = []
    controller = SimpleNamespace(SetFrameEvent=lambda event, _: events.append(event))
    ctx = SimpleNamespace(CurEvent=lambda: 7,
                          Replay=lambda: SimpleNamespace(BlockInvoke=lambda callback: callback(controller)))
    facade = RenderDocFacade(ctx)

    def callback(ctrl):
        ctrl.SetFrameEvent(99, True)
        if fail:
            raise ValueError("controller failure")

    if fail:
        with pytest.raises(ValueError, match="controller failure"):
            facade._invoke(callback)
    else:
        facade._invoke(callback)
    assert events == [99, 7]


def test_pipeline_batch_and_difference():
    facade = object.__new__(RenderDocFacade)
    facade.get_pipeline_state = lambda event: {
        "event_id": event, "shaders": {"pixel": {"resource_id": str(event)}}}
    states = facade.get_pipeline_states([1, 2])["states"]
    assert [item["event_id"] for item in states] == [1, 2]
    result = facade.compare_pipeline_states(1, 2)
    assert result["changes"] == [{"path": "/shaders/pixel/resource_id", "change": "changed",
                                 "before": "1", "after": "2"}]
    assert facade.compare_pipeline_states(1, 1)["equal"]
    assert differences({"a/b": None}, {"a/b": None, "~key": 1})[0]["path"] == "/~0key"


def test_pipeline_reports_real_output_and_vertex_bindings():
    viewport = SimpleNamespace(x=0, y=0, width=640, height=480, minDepth=0, maxDepth=1)
    scissor = SimpleNamespace(x=0, y=0, width=640, height=480)
    buffer = SimpleNamespace(resourceId=12, byteOffset=32, byteSize=512, byteStride=16)
    pipe = SimpleNamespace(
        GetShader=lambda _: 0, GetOutputTargets=lambda: [SimpleNamespace(resource=24)],
        GetDepthTarget=lambda: SimpleNamespace(resource=25), GetPrimitiveTopology=lambda: "TriangleList",
        GetVBuffers=lambda: [buffer], GetIBuffer=lambda: buffer,
        IsCaptureVK=lambda: False, IsCaptureD3D11=lambda: True,
    )
    controller = SimpleNamespace(
        SetFrameEvent=lambda *_: None, GetPipelineState=lambda: pipe,
        GetAPIProperties=lambda: SimpleNamespace(pipelineType="D3D11"),
        GetD3D11PipelineState=lambda: SimpleNamespace(rasterizer=SimpleNamespace(
            viewports=[viewport], scissors=[scissor])),
    )
    service = PipelineService(SimpleNamespace(IsCaptureLoaded=lambda: True), lambda cb: cb(controller))
    result = service.get_pipeline_state(5)
    assert result["render_targets"] == [{"index": 0, "resource_id": "24"}]
    assert result["depth_target"] == "25"
    assert result["input_assembly"]["vertex_buffers"][0]["offset"] == 32
    assert result["viewport_scissor"]["viewports"][0]["width"] == 640
    assert "unavailable" not in result


def test_incomplete_pipeline_does_not_claim_equality():
    facade = object.__new__(RenderDocFacade)
    facade.get_pipeline_state = lambda _: {"api": "unknown", "unavailable": {"outputs": "unsupported"}}
    result = facade.compare_pipeline_states(1, 2)
    assert result["equal"] is None and not result["complete"]


def test_buffer_remaining_length_at_offset_and_file_export(tmp_path, monkeypatch):
    monkeypatch.setattr(Parsers, "parse_resource_id", lambda _: 1)
    calls = []
    controller = SimpleNamespace(
        SetFrameEvent=lambda event, _: calls.append(("event", event)),
        GetBuffers=lambda: [SimpleNamespace(resourceId=1, length=8)],
        GetBufferData=lambda rid, offset, length: calls.append((offset, length)) or b"abcdefgh"[offset:offset + length],
    )
    service = ResourceService(SimpleNamespace(IsCaptureLoaded=lambda: True), lambda cb: cb(controller))
    path = tmp_path / "buffer.bin"
    result = service.get_buffer_contents("1", offset=5, length=0, event_id=3, output_path=str(path))
    assert calls == [("event", 3), (5, 3)]
    assert path.read_bytes() == b"fgh"
    assert result["length"] == 3 and "content_base64" not in result
    with pytest.raises(ValueError, match="Invalid buffer"):
        service.get_buffer_contents("1", offset=9)


def test_cube_face_bounds_and_texture_file_export(tmp_path):
    texture = SimpleNamespace(resourceId=1, width=4, height=4, depth=1, arraysize=6,
                              mips=1, cubemap=True, msSamp=1,
                              format=SimpleNamespace(Name=lambda: "RGBA8"), type="TextureCube")
    controller = SimpleNamespace(GetTextures=lambda: [texture], GetTextureData=lambda *_: b"pixels",
                                 SetFrameEvent=lambda *_: None)
    service = ResourceService(SimpleNamespace(IsCaptureLoaded=lambda: True), lambda cb: cb(controller))
    result = service.get_texture_data("1", slice=5, event_id=2, output_path=str(tmp_path / "raw.bin"))
    assert result["data_length"] == 6 and "content_base64" not in result
    assert (tmp_path / "raw.bin").read_bytes() == b"pixels"
    for kwargs in ({"slice": 6}, {"sample": 1}, {"mip": 1}):
        with pytest.raises(ValueError, match="Invalid"):
            service.get_texture_data("1", **kwargs)


def test_extension_remains_python36_syntax_compatible():
    import ast
    for path in Path("renderdoc_extension").rglob("*.py"):
        ast.parse(path.read_text("utf-8"), filename=str(path), feature_version=(3, 6))
