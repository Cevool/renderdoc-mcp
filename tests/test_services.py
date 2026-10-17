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
rd.MeshDataStage = SimpleNamespace(VSOut=1, GSOut=2)
rd.CompType = SimpleNamespace(Float=1, Typeless=0)
rd.VarType = SimpleNamespace(Float=1, Int=2, UInt=3)
rd.SDBasic = enum.IntEnum("SDBasic", "Struct Array Chunk Resource Float Boolean SignedInteger UnsignedInteger Enum GPUAddress String Character Null Buffer")
sys.modules["renderdoc"] = rd

from renderdoc_extension.renderdoc_facade import RenderDocFacade  # noqa: E402
from renderdoc_extension.services.resource_service import ResourceService  # noqa: E402
from renderdoc_extension.services.pipeline_service import PipelineService  # noqa: E402
from renderdoc_extension.services.analysis_service import AnalysisService  # noqa: E402
from renderdoc_extension.services.search_service import SearchService  # noqa: E402
from renderdoc_extension.utils.native import native_value, incomplete  # noqa: E402
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


def test_sampler_filter_compares_values_instead_of_swig_pointer_addresses():
    class TextureFilter:
        minify = "Linear"
        magnify = "Linear"
        mip = "Point"
    pipe = SimpleNamespace(GetSamplers=lambda *_: [SimpleNamespace(
        access=SimpleNamespace(index=0), sampler=SimpleNamespace(filter=TextureFilter()))])
    service = PipelineService(None, None)
    before = service._get_stage_samplers(pipe, rd.ShaderStage.Pixel, None)
    after = service._get_stage_samplers(pipe, rd.ShaderStage.Pixel, None)
    assert before == after and before[0]["filter"]["minify"] == "Linear"


def test_selected_missing_pipeline_fields_preserve_incompleteness():
    facade = object.__new__(RenderDocFacade)
    facade.get_pipeline_state = lambda _: {"api": "D3D11", "unavailable": {"shaders": "failed"}}
    result = facade.compare_pipeline_states(1, 2, fields=["render_targets"])
    assert result["equal"] is None and not result["complete"]


def test_buffer_remaining_length_at_offset_and_file_export(tmp_path, monkeypatch):
    monkeypatch.setattr(Parsers, "parse_resource_id", lambda *_: 1)
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


def analysis_service(controller, **context):
    ctx = {"IsCaptureLoaded": lambda: True,
           "GetAction": lambda _: action(5, "draw", rd.ActionFlags.Drawcall), **context}
    return AnalysisService(SimpleNamespace(**ctx), lambda cb: cb(controller))


def test_native_fixed_state_preserves_false_zero_enums_and_reports_missing():
    cull = enum.IntEnum("CullMode", {"None_": 0, "Back": 2})
    controller = SimpleNamespace(SetFrameEvent=lambda *_: None,
        GetAPIProperties=lambda: SimpleNamespace(pipelineType="D3D11"),
        GetD3D11PipelineState=lambda: SimpleNamespace(
            rasterizer=SimpleNamespace(state=SimpleNamespace(frontCCW=False, cullMode=cull.Back, depthBias=0)),
            outputMerger=SimpleNamespace(depthStencilState=SimpleNamespace(depthEnable=False))))
    result = analysis_service(controller).get_native_pipeline_state(5)
    state = result["native"]["rasterizer"]["state"]
    assert result["complete"] and state["frontCCW"] is False and state["depthBias"] == 0
    assert state["cullMode"]["value"] == 2
    controller.GetD3D11PipelineState = lambda: SimpleNamespace(rasterizer=SimpleNamespace(frontCCW=False))
    result = analysis_service(controller).get_native_pipeline_state(5)
    assert not result["complete"] and "outputMerger" in result["unavailable"]


def test_native_serialization_bounds_are_explicit():
    result = native_value(list(range(300)))
    assert len(result["items"]) == 256 and result["truncated"]
    assert incomplete(result) and native_value(float("inf")) == {"non_finite": "inf"}


def test_post_vs_resolves_indices_base_vertex_offsets_and_nonfinite():
    import struct
    mesh = SimpleNamespace(status="", vertexResourceId=10, indexResourceId=11,
        vertexByteOffset=32, vertexByteStride=16, indexByteOffset=4, indexByteStride=2,
        baseVertex=1, numIndices=4, allowRestart=True, restartIndex=0xFFFF,
        format=SimpleNamespace(Special=lambda: False, compType=rd.CompType.Float, compCount=4, compByteWidth=4))
    calls = []
    def buffer(rid, offset, size):
        calls.append((rid, offset, size))
        if rid == 11:
            return struct.pack("<HHH", 2, 0xFFFF, 0)
        return struct.pack("<ffff", float("nan") if offset == 80 else 1, 2, 3, 0)
    controller = SimpleNamespace(SetFrameEvent=lambda *_: None, GetPostVSData=lambda *_: mesh, GetBufferData=buffer)
    result = analysis_service(controller).get_post_vs_data(5, offset=1, limit=3, include_bytes=True)
    assert calls == [(11, 6, 6), (10, 80, 16), (10, 48, 16)]
    assert result["vertices"][0]["vertex_index"] == 3
    assert not result["vertices"][0]["finite"] and result["vertices"][0]["w_zero"]
    assert result["vertices"][1]["restart"]
    assert result["next_offset"] is None and result["total"] == 4
    assert "raw_base64" in result["vertices"][2]


def test_post_vs_missing_capture_invalid_event_and_empty_output_are_distinct():
    controller = SimpleNamespace(SetFrameEvent=lambda *_: None,
        GetPostVSData=lambda *_: SimpleNamespace(status="", vertexResourceId=0))
    result = analysis_service(controller).get_post_vs_data(5)
    assert result["status"] == "empty" and result["complete"]
    with pytest.raises(ValueError, match="No action"):
        analysis_service(controller, GetAction=lambda _: None).get_post_vs_data(5)
    with pytest.raises(ValueError, match="No capture"):
        analysis_service(controller, IsCaptureLoaded=lambda: False).get_post_vs_data(5)


def test_pixel_history_support_empty_failure_flags_and_pages(monkeypatch):
    monkeypatch.setattr(Parsers, "parse_resource_id", lambda *_: 1)
    controller = SimpleNamespace(SetFrameEvent=lambda *_: None,
        GetAPIProperties=lambda: SimpleNamespace(pixelHistory=False))
    service = analysis_service(controller)
    assert service.get_pixel_history(5, "1", 0, 0)["status"] == "unsupported"
    controller.GetAPIProperties = lambda: SimpleNamespace(pixelHistory=True)
    controller.GetTextures = lambda: [SimpleNamespace(resourceId=1, mips=1, arraysize=1, depth=1,
                                                     width=8, height=8, msSamp=1)]
    controller.PixelHistory = lambda *_: [SimpleNamespace(eventId=i, depthTestFailed=True, primitiveID=12) for i in (1, 4, 8)]
    result = service.get_pixel_history(5, "1", 2, 3, limit=1)
    assert result["total"] == 2 and result["next_offset"] == 1 and result["truncated"]
    assert result["history"][0]["depthTestFailed"] and result["coordinate_origin"] == "top-left"
    controller.PixelHistory = lambda *_: []
    assert service.get_pixel_history(5, "1", 0, 0)["status"] == "empty"
    with pytest.raises(ValueError, match="Invalid pixel"):
        service.get_pixel_history(5, "1", 8, 0)


def test_api_events_read_real_chunk_parameters_and_resource_relationships():
    def sd(name, kind, value=None, children=()):
        return SimpleNamespace(name=name, type=SimpleNamespace(name="test", basetype=kind, flags=0),
            NumChildren=lambda: len(children), GetChild=lambda i: children[i],
            AsResourceId=lambda: value,
            data=SimpleNamespace(basic=SimpleNamespace(i=value, u=value, d=value, b=value, c=value), string=str(value)))
    chunks = [sd("ClearRenderTargetView", rd.SDBasic.Chunk, children=[
        sd("target", rd.SDBasic.Resource, 12), sd("red", rd.SDBasic.Float, 0.25)])]
    clear = action(3, "clear", rd.ActionFlags.Clear)
    clear.events = [SimpleNamespace(eventId=2, chunkIndex=0)]
    controller = SimpleNamespace(GetRootActions=lambda: [clear])
    service = analysis_service(controller, GetStructuredFile=lambda: SimpleNamespace(chunks=chunks))
    result = service.get_api_events(2, 2, include_parameters=True)
    record = result["events"][0]
    assert result["complete"] and record["event_id"] == 2 and record["action_event_id"] == 3
    assert record["parameters"]["children"][0]["resource_id"] == "12"
    assert record["parameters"]["children"][1]["value"] == 0.25


@pytest.mark.parametrize("fail", [False, True])
def test_shader_debug_prefix_is_bounded_and_trace_freed_on_every_path(fail):
    freed = []
    trace = SimpleNamespace(debugger="debug", inputs=[], constantBlocks=[], sourceVars=[])
    def continuation(_):
        if fail:
            raise RuntimeError("debug failure")
        return [SimpleNamespace(stepIndex=i, nextInstruction=i, flags=0, callstack=[], changes=[]) for i in range(5)]
    controller = SimpleNamespace(SetFrameEvent=lambda *_: None,
        GetAPIProperties=lambda: SimpleNamespace(shaderDebugging=True),
        DebugVertex=lambda *_: trace, ContinueDebug=continuation, FreeTrace=lambda t: freed.append(t))
    service = analysis_service(controller)
    if fail:
        with pytest.raises(RuntimeError, match="debug failure"):
            service.debug_shader(5, "vertex", {"vertex": 0, "index": 0}, max_steps=2)
    else:
        result = service.debug_shader(5, "vertex", {"vertex": 0, "index": 0}, max_steps=2)
        assert result["returned"] == 2 and result["truncated"] and not result["finished"]
        assert not result["complete"] and result["trace_released"]
    assert freed == [trace]


def test_constant_buffer_compute_binding_and_unbound_are_explicit():
    block = SimpleNamespace(name="Constants", byteSize=16, bufferBacked=True, variables=[])
    reflection = SimpleNamespace(constantBlocks=[block], resourceId=5, entryPoint="CSMain")
    bound = SimpleNamespace(resource=10, byteOffset=256, byteSize=16)
    pipe = SimpleNamespace(GetShaderReflection=lambda _: reflection,
        GetConstantBlock=lambda *_: SimpleNamespace(descriptor=bound), GetComputePipelineObject=lambda: 99)
    reads = []
    controller = SimpleNamespace(SetFrameEvent=lambda *_: None, GetPipelineState=lambda: pipe,
        GetCBufferVariableContents=lambda *args: reads.append(args) or [])
    service = PipelineService(SimpleNamespace(IsCaptureLoaded=lambda: True, GetAction=lambda _: True), lambda cb: cb(controller))
    result = service.get_constant_buffer(5, "compute")
    assert reads[0][0] == 99 and reads[0][5:] == (10, 256, 16)
    assert result["raw_read"]["arguments"]["offset"] == 256 and result["bound"]
    bound.resource = 0
    result = service.get_constant_buffer(5, "compute")
    assert result["status"] == "empty" and not result["bound"] and "raw_read" not in result


def test_shader_values_keep_signed_scalars_and_nonfinite_diagnostics():
    variable = SimpleNamespace(name="control", type=rd.VarType.Int, rows=1, columns=1,
                               value=SimpleNamespace(s32v=[-7]), members=[])
    assert Serializers.serialize_variables([variable])[0]["value"] == [-7]
    variable.type = rd.VarType.Float
    variable.value = SimpleNamespace(f32v=[float("nan")])
    assert Serializers.serialize_variables([variable])[0]["value"] == ["nan"]
    variable.value = SimpleNamespace()
    assert "error" in Serializers.serialize_variables([variable])[0]


def test_opaque_resource_ids_are_resolved_from_replay_not_python_attributes():
    class OpaqueID:
        __slots__ = ()
        def __str__(self):
            return "ResourceId::123"
    rid = OpaqueID()
    controller = SimpleNamespace(GetResources=lambda: [SimpleNamespace(resourceId=rid)])
    assert Parsers.parse_resource_id("123", controller) is rid
    assert Parsers.parse_resource_id("ResourceId::123", controller) is rid
    with pytest.raises(ValueError, match="Unknown resource"):
        Parsers.parse_resource_id("124", controller)
    with pytest.raises(ValueError, match="non-null"):
        Parsers.parse_resource_id("0", controller)


def test_reverse_lookup_finds_output_attachments_with_current_pipeline_api():
    pipe = SimpleNamespace(GetShader=lambda _: 0, GetReadOnlyResources=lambda *_: [],
        GetReadWriteResources=lambda *_: [], GetOutputTargets=lambda: [SimpleNamespace(resource=123)],
        GetDepthTarget=lambda: SimpleNamespace(resource=0))
    controller = SimpleNamespace(GetResources=lambda: [SimpleNamespace(resourceId=123)])
    ctx = SimpleNamespace(GetResourceName=lambda _: "FaceColor")
    service = SearchService(ctx, None)
    service._search_draws = lambda matcher: matcher(pipe, controller, None, ctx)
    assert service.find_draws_by_resource("123") == "RenderTarget[0]"
    assert "RenderTarget[0]" in service.find_draws_by_texture("face")


def test_resource_usage_works_without_optional_view_ids(monkeypatch):
    monkeypatch.setattr(Parsers, "parse_resource_id", lambda *_: 1)
    controller = SimpleNamespace(GetUsage=lambda _: [SimpleNamespace(eventId=2, usage="Clear")])
    result = analysis_service(controller).get_resource_usage("1")
    assert result["status"] == "ok" and result["usages"][0]["view_id"] is None
    controller.GetUsage = lambda _: []
    assert analysis_service(controller).get_resource_usage("1")["status"] == "empty"


def test_vulkan_native_state_keeps_pass_attachments_and_load_store_fields():
    native = SimpleNamespace(rasterizer=SimpleNamespace(frontCCW=True), depthStencil={},
        colorBlend={}, multisample={}, viewportScissor={},
        currentPass={"attachments": [{"resource": "12", "loadOp": "Clear", "storeOp": "Store"}]})
    controller = SimpleNamespace(SetFrameEvent=lambda *_: None,
        GetAPIProperties=lambda: SimpleNamespace(pipelineType="GraphicsAPI.Vulkan"),
        GetVulkanPipelineState=lambda: native)
    result = analysis_service(controller).get_native_pipeline_state(5)
    assert result["complete"] and result["native"]["currentPass"]["attachments"][0]["loadOp"] == "Clear"


def test_pixel_debug_keeps_native_defaults_without_module_no_preference(monkeypatch):
    class PixelInputs:
        def __init__(self):
            self.sample, self.primitive, self.view = 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
    monkeypatch.setattr(rd, "DebugPixelInputs", PixelInputs, raising=False)
    assert not hasattr(rd, "NoPreference")
    trace = SimpleNamespace(debugger="debugger", inputs=[], constantBlocks=[], sourceVars=[])
    selected, freed = [], []
    controller = SimpleNamespace(SetFrameEvent=lambda *_: None,
        GetAPIProperties=lambda: SimpleNamespace(shaderDebugging=True),
        DebugPixel=lambda x, y, inputs: selected.append((x, y, inputs)) or trace,
        ContinueDebug=lambda _: [], FreeTrace=lambda value: freed.append(value))
    result = analysis_service(controller).debug_shader(5, "pixel", {"x": 10, "y": 20, "primitive": 7})
    assert result["finished"] and not result["truncated"] and freed == [trace]
    assert selected[0][:2] == (10, 20)
    assert (selected[0][2].sample, selected[0][2].primitive, selected[0][2].view) == (0xFFFFFFFF, 7, 0xFFFFFFFF)
