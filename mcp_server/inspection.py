"""Buffer elements, vertex inputs, mesh export, batch debugging, messages and state scans."""

from pathlib import Path
from typing import Annotated, Literal

from fastmcp.exceptions import ToolError
from pydantic import BaseModel, Field

Event = Annotated[int, Field(ge=0)]
Offset = Annotated[int, Field(ge=0)]
# Mirrors renderdoc_extension.utils.elements.TYPE_PATTERN (the extension cannot be imported here).
TYPE_PATTERN = (r"^((float|half|double|int|uint|bool)([1-4](x[1-4])?)?"
                r"|(int8|uint8|int16|uint16|int64|uint64|unorm8|snorm8|unorm16|snorm16)(x[1-4])?"
                r"|r10g10b10a2_unorm|r10g10b10a2_uint|r11g11b10_float)$")
SECTIONS = ("shaders", "render_targets", "depth_target", "input_assembly", "viewport_scissor", "native")
DEFAULT_SECTIONS = ["shaders", "render_targets", "depth_target", "viewport_scissor", "native"]
LONG_TIMEOUT = 300.0
SCAN_TIMEOUT = 600.0


class ElementField(BaseModel):
    name: Annotated[str, Field(min_length=1, max_length=128)]
    offset: Annotated[int, Field(ge=0, le=65535, description="Byte offset inside one element")]
    type: Annotated[str, Field(pattern=TYPE_PATTERN, description=(
        "float/half/double/int/uint/bool with optional HLSL shape (float3, float4x4); "
        "int8/uint8/int16/uint16/int64/uint64/unorm8/snorm8/unorm16/snorm16 with optional xN "
        "(unorm8x4); r10g10b10a2_unorm, r10g10b10a2_uint, r11g11b10_float"))]


Fields = Annotated[list[ElementField], Field(min_length=1, max_length=256)]
DebugInvocation = Annotated[dict, Field(description=(
    "vertex: {vertex,index,instance?,view?}; index includes draw offsets. "
    "pixel: {x,y,sample?,primitive?,view?}, top-left coordinates. "
    "compute: {group:[x,y,z],thread:[x,y,z]} or {global:[x,y,z]} (SV_DispatchThreadID). "
    "All coordinates are nonnegative integers."))]


def check_invocation(stage, invocation):
    if stage == "compute" and "global" in invocation:
        required, optional = {"global"}, set()
    else:
        required = {"vertex": {"vertex", "index"}, "pixel": {"x", "y"}, "compute": {"group", "thread"}}[stage]
        optional = {"vertex": {"instance", "view"}, "pixel": {"sample", "primitive", "view"}, "compute": set()}[stage]
    if not required <= invocation.keys() or invocation.keys() - required - optional:
        raise ToolError("Invalid %s invocation: required keys %s; optional keys %s" % (
            stage, sorted(required), sorted(optional)))
    for value in invocation.values():
        values = value if isinstance(value, list) else [value]
        if (stage == "compute" and (not isinstance(value, list) or len(value) != 3)
                or stage != "compute" and isinstance(value, list)
                or any(type(v) is not int or v < 0 or v > 0xFFFFFFFF for v in values)):
            raise ToolError("Invocation coordinates must be unsigned integers; compute coordinates need three axes")


def register_inspection_tools(mcp, invoke, artifacts):
    @mcp.tool
    def read_buffer_elements(instance_id: str, event_id: Event, resource_id: str,
                             fields: Fields | None = None,
                             element_stride: Annotated[int, Field(ge=1, le=65536)] | None = None,
                             byte_offset: Offset | None = None, start: Offset = 0,
                             count: Annotated[int, Field(ge=1, le=1000)] = 64,
                             include_hex: bool = False, binding: Offset | None = None) -> dict:
        """Decode a buffer as typed elements at an event (structured, typed, vertex or index data).

        Without fields, the layout comes from how the buffer is bound at event_id: view format,
        shader reflection type + view element size, vertex input layout or index width; the
        view's byte offset becomes element 0. The result lists every binding; select one with
        binding (its index) when a pooled buffer is bound several ways. Stripped shaders often
        reflect only the element type: pass fields (and element_stride) for real structures.
        start/count are elements.
        """
        return artifacts.bound(invoke(instance_id, "read_buffer_elements", event_id=event_id,
            resource_id=resource_id, fields=[f.model_dump() for f in fields] if fields else None,
            element_stride=element_stride, byte_offset=byte_offset, start=start, count=count,
            include_hex=include_hex, binding=binding))

    @mcp.tool
    def compare_buffer_contents(instance_id: str, event_id: Event, resource_id: str,
                                baseline_event_id: Event | None = None, offset: Offset = 0,
                                length: Offset = 0,
                                element_stride: Annotated[int, Field(ge=1, le=65536)] = 4,
                                fields: Fields | None = None,
                                limit: Annotated[int, Field(ge=1, le=1000)] = 100,
                                max_bytes: Annotated[int, Field(ge=1, le=268435456)] = 33554432) -> dict:
        """Diff a buffer's bytes after baseline_event_id vs after event_id (what a dispatch/draw wrote).

        baseline defaults to event_id - 1 (the state just before event_id). Reports changed
        element count, merged changed index ranges, hashes and the first `limit` changes decoded
        with fields (or as uint/float for 4-byte elements). length=0 compares to the end, capped
        by max_bytes.
        """
        if baseline_event_id == event_id or (baseline_event_id is None and event_id == 0):
            raise ToolError("baseline_event_id must differ from event_id")
        return artifacts.bound(invoke(instance_id, "compare_buffer_contents", _timeout=LONG_TIMEOUT,
            event_id=event_id, resource_id=resource_id, baseline_event_id=baseline_event_id,
            offset=offset, length=length, element_stride=element_stride,
            fields=[f.model_dump() for f in fields] if fields else None, limit=limit,
            max_bytes=max_bytes))

    @mcp.tool
    def get_vertex_inputs(instance_id: str, event_id: Event, instance: Offset = 0,
                          offset: Offset = 0, limit: Annotated[int, Field(ge=1, le=500)] = 50,
                          attributes: Annotated[list[str], Field(min_length=1, max_length=64)] | None = None) -> dict:
        """Read a draw's input layout and decoded vertex attributes per index row (VS input).

        Returns topology, index buffer (effective offset, width, restart), vertex buffers, every
        attribute's format/slot/offset/step rate, and rows with resolved vertex indices. attributes
        selects names (default: attributes the shader uses). offset/limit are index rows.
        """
        return artifacts.bound(invoke(instance_id, "get_vertex_inputs", event_id=event_id,
            instance=instance, offset=offset, limit=limit, attributes=attributes))

    @mcp.tool
    def export_mesh(instance_id: str, event_id: Event,
                    stage: Literal["vsin", "vsout", "gsout"] = "vsin",
                    format: Literal["obj", "ply"] = "obj", instance: Offset = 0, view: Offset = 0,
                    space: Literal["ndc", "clip"] = "ndc", position_attribute: str | None = None,
                    normal_attribute: str | None = None, uv_attribute: str | None = None,
                    output_path: str | None = None) -> dict:
        """Export one draw's geometry as OBJ or PLY, with faces/lines from its topology and indices.

        vsin = vertex shader inputs (positions as fed to the VS, plus normal/uv when found or
        named). Attributes are picked by name (POSITION/ATTRIBUTE0, NORMAL, TEXCOORD0); pass the
        names for engines with generic semantics. vsout/gsout = post-transform positions in NDC
        or clip space. Shared vertices stay shared. output_path must be a new absolute file.
        """
        if output_path is not None:
            path = Path(output_path)
            if not path.is_absolute() or path.exists() or path.suffix.lower() != "." + format:
                raise ToolError("output_path must be a new absolute .%s file" % format)
        else:
            path = artifacts.allocate("." + format)
        value = invoke(instance_id, "export_mesh", _timeout=SCAN_TIMEOUT, event_id=event_id,
                       output_path=str(path.resolve()), fmt=format, stage=stage, instance=instance,
                       view=view, space=space, position_attribute=position_attribute,
                       normal_attribute=normal_attribute, uv_attribute=uv_attribute)
        if output_path is None:
            value.update(artifacts.describe(path))
        return artifacts.bound(value)

    @mcp.tool
    def debug_shader_batch(instance_id: str, event_id: Event,
                           stage: Literal["vertex", "pixel", "compute"],
                           invocations: Annotated[list[DebugInvocation], Field(min_length=1, max_length=64)],
                           variables: Annotated[list[str], Field(min_length=1, max_length=64)] | None = None,
                           max_steps: Annotated[int, Field(ge=1, le=1000000)] = 100000) -> dict:
        """Debug up to 64 invocations of one event and return each one's final variable values.

        Each invocation runs to completion (or max_steps; 120 s for the whole batch) without
        keeping step history. variables filters inputs/variables by case-insensitive substring of
        their own or nested member names (DXBC: ["o0", "r3"]; DXIL outputs: ["SV_Target"]).
        Use debug_shader for one invocation's step-by-step trace.
        """
        for invocation in invocations:
            check_invocation(stage, invocation)
        return artifacts.bound(invoke(instance_id, "debug_shader_batch", _timeout=LONG_TIMEOUT,
            event_id=event_id, stage=stage, invocations=invocations, variables=variables,
            max_steps=max_steps))

    @mcp.tool
    def get_debug_messages(instance_id: str,
                           min_severity: Literal["high", "medium", "low", "info"] = "info",
                           event_id_min: Event | None = None, event_id_max: Event | None = None,
                           pattern: str | None = None, offset: Offset = 0,
                           limit: Annotated[int, Field(ge=1, le=200)] = 50) -> dict:
        """List API debug/validation messages with severity, category, event and text.

        These exist only if the capture was taken with API validation, or the replay runs with
        it. pattern is a case-insensitive substring of the message text.
        """
        if event_id_min is not None and event_id_max is not None and event_id_min > event_id_max:
            raise ToolError("event_id_min must not exceed event_id_max")
        return artifacts.bound(invoke(instance_id, "get_debug_messages", min_severity=min_severity,
            event_id_min=event_id_min, event_id_max=event_id_max, pattern=pattern,
            offset=offset, limit=limit))

    @mcp.tool
    def scan_state_changes(instance_id: str, event_id_min: Event, event_id_max: Event,
                           sections: Annotated[list[Literal[SECTIONS]], Field(min_length=1)] | None = None,
                           ignore_paths: Annotated[list[str], Field(max_length=32)] | None = None,
                           include_dispatches: bool = True,
                           max_events: Annotated[int, Field(ge=2, le=1000)] = 200,
                           max_changes_per_event: Annotated[int, Field(ge=1, le=500)] = 32) -> dict:
        """Walk the draws (and dispatches) in an event range and report state changes between them.

        One replay pass; each event lists JSON-pointer changes relative to the previous scanned
        event, grouped by section. Default sections: shaders, render_targets, depth_target,
        viewport_scissor and native (API blend/depth/raster state). ignore_paths are regexes
        matched against change paths (e.g. "/constant_buffers/" to hide per-draw CB offsets).
        Continue truncated scans from next_event_id_min.
        """
        import re
        if event_id_min > event_id_max:
            raise ToolError("event_id_min must not exceed event_id_max")
        for pattern in ignore_paths or []:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ToolError("Invalid ignore_paths regex %r: %s" % (pattern, exc)) from exc
        return artifacts.bound(invoke(instance_id, "scan_state_changes", _timeout=SCAN_TIMEOUT,
            event_id_min=event_id_min, event_id_max=event_id_max,
            sections=list(sections or DEFAULT_SECTIONS), ignore_paths=ignore_paths,
            include_dispatches=include_dispatches, max_events=max_events,
            max_changes=max_changes_per_event))
