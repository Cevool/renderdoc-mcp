"""Small discovery interface; RenderDoc operations are disclosed on demand."""

from typing import Annotated, Literal

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.utilities.types import Image
from pydantic import Field

from .bridge.client import RenderDocBridge, RenderDocBridgeError
from .catalog import CategorizedDiscovery
from .config import settings
from .results import ArtifactStore, flatten_actions, page_result

Offset = Annotated[int, Field(ge=0)]
Limit = Annotated[int, Field(ge=1, le=200)]
Event = Annotated[int, Field(ge=0)]
Events = Annotated[list[Event], Field(min_length=1, max_length=32)]
Stage = Literal["vertex", "hull", "domain", "geometry", "pixel", "compute"]
PipelineField = Literal["event_id", "api", "shaders", "render_targets", "depth_target",
                        "input_assembly", "viewport_scissor", "unavailable"]
ActionField = Literal["event_id", "action_id", "name", "flags", "num_indices", "num_instances",
                      "parent_event_id"]

mcp = FastMCP(
    name="RenderDoc MCP Server",
    instructions=("Start with list_instances. Pass the chosen instance_id in every RenderDoc "
                  "operation's arguments. Browse categories with browse_tools(), then "
                  "browse_tools(category), get_tool_schema(name), and call_tool(name, arguments). "
                  "search_tools is an optional paginated shortcut, not the full directory. "
                  "Results are paginated; large results are saved as local artifacts."),
    transforms=[CategorizedDiscovery()],
)
bridge = RenderDocBridge(settings.ipc_dir, settings.timeout)
artifacts = ArtifactStore(settings.artifact_dir)


def invoke(instance_id, method, **params):
    try:
        return bridge.call(instance_id, method, params)
    except RenderDocBridgeError as exc:
        import json
        raise ToolError(json.dumps({"message": str(exc), **exc.context})) from exc


@mcp.tool
def list_instances() -> dict:
    """Discover live RenderDoc instances with IDs, PIDs, versions and loaded capture filenames."""
    return bridge.list_instances()


@mcp.tool
def get_capture_status(instance_id: str) -> dict:
    """Check whether the selected RenderDoc instance has a capture loaded."""
    return invoke(instance_id, "get_capture_status")


@mcp.tool
def get_frame_summary(instance_id: str) -> dict:
    """Start frame analysis: action statistics, top-level markers and resource counts."""
    return artifacts.bound(invoke(instance_id, "get_frame_summary"))


@mcp.tool
def get_draw_calls(
    instance_id: str, marker_filter: str | None = None,
    exclude_markers: list[str] | None = None, event_id_min: Event | None = None,
    event_id_max: Event | None = None, only_actions: bool = False,
    flags_filter: list[str] | None = None, offset: Offset = 0, limit: Limit = 50,
    fields: list[ActionField] | None = None, preset: Literal["unity_game_rendering"] | None = None,
) -> dict:
    """Page through flat draw/dispatch/marker records with parent_event_id; filter before paging.

    fields selects record keys. Unity preset is optional; custom marker filters override it.
    """
    if event_id_min is not None and event_id_max is not None and event_id_min > event_id_max:
        raise ToolError("event_id_min must not exceed event_id_max")
    if preset:
        marker_filter = marker_filter or "Camera.Render"
        if exclude_markers is None:
            exclude_markers = ["GUI.Repaint", "UIR.DrawChain"]
    value = invoke(instance_id, "get_draw_calls", include_children=True,
                   marker_filter=marker_filter, exclude_markers=exclude_markers,
                   event_id_min=event_id_min, event_id_max=event_id_max,
                   only_actions=only_actions, flags_filter=flags_filter)
    value["actions"] = list(flatten_actions(value["actions"]))
    return artifacts.bound(page_result(value, "actions", offset, limit, fields))


def search_result(instance_id, method, offset, limit, **params):
    return artifacts.bound(page_result(invoke(instance_id, method, **params), "matches", offset, limit))


@mcp.tool
def find_draws_by_shader(instance_id: str, shader_name: str, stage: Stage | None = None,
                         offset: Offset = 0, limit: Limit = 50) -> dict:
    """Find draws using a shader by partial shader/entry-point name and optional stage."""
    return search_result(instance_id, "find_draws_by_shader", offset, limit,
                         shader_name=shader_name, stage=stage)


@mcp.tool
def find_draws_by_texture(instance_id: str, texture_name: str,
                          offset: Offset = 0, limit: Limit = 50) -> dict:
    """Find draws binding a texture by partial texture name, including outputs and UAVs."""
    return search_result(instance_id, "find_draws_by_texture", offset, limit, texture_name=texture_name)


@mcp.tool
def find_draws_by_resource(instance_id: str, resource_id: str,
                           offset: Offset = 0, limit: Limit = 50) -> dict:
    """Find draws using an exact ResourceId, including shaders and resource bindings."""
    return search_result(instance_id, "find_draws_by_resource", offset, limit, resource_id=resource_id)


@mcp.tool
def get_draw_call_details(instance_id: str, event_id: Event) -> dict:
    """Inspect a draw's counts, flags, offsets and output resource IDs."""
    return artifacts.bound(invoke(instance_id, "get_draw_call_details", event_id=event_id))


@mcp.tool
def get_action_timings(instance_id: str, event_ids: Events | None = None,
                        marker_filter: str | None = None, exclude_markers: list[str] | None = None,
                        offset: Offset = 0, limit: Limit = 50) -> dict:
    """Measure GPU action timings when counters are supported; page through the results."""
    value = invoke(instance_id, "get_action_timings", event_ids=event_ids,
                   marker_filter=marker_filter, exclude_markers=exclude_markers)
    return artifacts.bound(page_result(value, "timings", offset, limit))


@mcp.tool
def get_shader_info(instance_id: str, event_id: Event, stage: Stage) -> dict:
    """Read shader disassembly/source, constant buffer values and resource bindings at an event."""
    return artifacts.bound(invoke(instance_id, "get_shader_info", event_id=event_id, stage=stage))


@mcp.tool
def get_constant_buffer(instance_id: str, event_id: Event, stage: Stage,
                        block_index: Offset = 0, array_index: Offset = 0) -> dict:
    """Link reflected constant declarations to bound resource/byte range and decoded GPU values.

    block_index indexes reflection.constantBlocks, not a register number. raw_read contains a
    complete get_buffer_contents call, including this instance_id, for the exact bytes.
    """
    value = invoke(instance_id, "get_constant_buffer", event_id=event_id,
                   stage=stage, block_index=block_index, array_index=array_index)
    if value.get("raw_read"):
        value["raw_read"]["arguments"]["instance_id"] = instance_id
    return artifacts.bound(value)


@mcp.tool
def get_native_pipeline_state(instance_id: str, event_id: Event) -> dict:
    """Read API-native rasterizer, depth/stencil, blend and attachment/pass state without defaults.

    Native member names and enum values are retained. D3D11/D3D12/Vulkan/OpenGL are supported;
    absent fields are explicitly unavailable. Use get_api_events for clear/load/store parameters.
    """
    return artifacts.bound(invoke(instance_id, "get_native_pipeline_state", event_id=event_id))


@mcp.tool
def get_post_vs_data(instance_id: str, event_id: Event, stage: Literal["vs", "gs"] = "vs",
                     instance: Offset = 0, view: Offset = 0, offset: Offset = 0,
                     limit: Limit = 50, include_bytes: bool = False) -> dict:
    """Read bounded Post-VS/GS rows, resolved indices, native layout and floating point positions.

    Reports finite positions and zero W only for returned rows. include_bytes includes raw output
    records for other varyings. offset is a mesh index-row offset, not a byte offset.
    """
    return artifacts.bound(invoke(instance_id, "get_post_vs_data", event_id=event_id, stage=stage,
        instance=instance, view=view, offset=offset, limit=limit, include_bytes=include_bytes))


@mcp.tool
def get_pixel_history(instance_id: str, event_id: Event, resource_id: str, x: Offset, y: Offset,
                       mip: Offset = 0, slice: Offset = 0, sample: Offset = 0,
                       offset: Offset = 0, limit: Limit = 50) -> dict:
    """Read pixel writes through event_id, values, primitive IDs and native rejection flags.

    Coordinates use a top-left origin on every API. Unsupported API/driver is distinct from an
    empty history. Pixel history is expensive even when only one result page is returned.
    """
    return artifacts.bound(invoke(instance_id, "get_pixel_history", event_id=event_id,
        resource_id=resource_id, x=x, y=y, mip=mip, slice=slice, sample=sample,
        offset=offset, limit=limit))


@mcp.tool
def get_api_events(instance_id: str, event_id_min: Event, event_id_max: Event,
                    offset: Offset = 0, limit: Limit = 50, include_parameters: bool = False) -> dict:
    """Inspect a bounded API event range, including clear/pass calls and structured parameters.

    Function names, flags and chunk indices are returned first. Enable parameters for clear values,
    load/store enums and resource references; serialized fields depend on API/RenderDoc version.
    """
    if event_id_min > event_id_max or event_id_max - event_id_min > 4096:
        raise ToolError("Expected an ordered event range spanning at most 4096 IDs")
    return artifacts.bound(invoke(instance_id, "get_api_events", event_id_min=event_id_min,
        event_id_max=event_id_max, offset=offset, limit=limit, include_parameters=include_parameters))


@mcp.tool
def get_resource_usage(instance_id: str, resource_id: str, offset: Offset = 0, limit: Limit = 50) -> dict:
    """Read actual resource usage events to establish read/write/clear ordering for initialization."""
    return artifacts.bound(page_result(invoke(instance_id, "get_resource_usage",
        resource_id=resource_id), "usages", offset, limit))


DebugInvocation = Annotated[dict, Field(description=(
    "vertex: {vertex,index,instance?,view?}; index includes draw offsets. "
    "pixel: {x,y,sample?,primitive?,view?}, top-left coordinates. "
    "compute: {group:[x,y,z],thread:[x,y,z]}. All coordinates are nonnegative integers."))]


@mcp.tool
def debug_shader(instance_id: str, event_id: Event, stage: Literal["vertex", "pixel", "compute"],
                   invocation: DebugInvocation,
                   max_steps: Annotated[int, Field(ge=1, le=10000)] = 1000) -> dict:
    """Debug one invocation: inputs, constants and bounded instruction/variable changes.

    Stops after max_steps, 64 replay batches or 10 seconds between batches; a single native API call
    cannot be interrupted. Always frees the debugger. A truncated trace is a prefix, not completion.
    Large traces become artifacts readable with read_artifact; no live debugger is retained.
    """
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
    return artifacts.bound(invoke(instance_id, "debug_shader", event_id=event_id,
        stage=stage, invocation=invocation, max_steps=max_steps))


@mcp.tool
def get_pipeline_state(instance_id: str, event_id: Event,
                         fields: list[PipelineField] | None = None) -> dict:
    """Inspect graphics pipeline bindings and state; fields optionally selects top-level sections."""
    value = invoke(instance_id, "get_pipeline_state", event_id=event_id)
    if fields:
        selected = {key: value[key] for key in {*fields, "unavailable"} if key in value}
        for key in set(fields) - value.keys():
            selected.setdefault("unavailable", {})[key] = "Requested section missing"
        value = selected
    return artifacts.bound(value)


@mcp.tool
def get_pipeline_states(instance_id: str, event_ids: Events,
                         fields: list[PipelineField] | None = None) -> dict:
    """Inspect up to 32 events in one serialized batch; optionally select pipeline sections."""
    return artifacts.bound(invoke(instance_id, "get_pipeline_states", event_ids=event_ids, fields=fields))


@mcp.tool
def compare_pipeline_states(instance_id: str, event_a: Event, event_b: Event,
                             fields: list[PipelineField] | None = None) -> dict:
    """Compare two events' pipeline states and return only changed values with JSON paths."""
    return artifacts.bound(invoke(instance_id, "compare_pipeline_states",
                                  event_a=event_a, event_b=event_b, fields=fields))


@mcp.tool
def compare_capture_states(instance_id: str, event_id: Event, other_instance_id: str,
                             other_event_id: Event, fields: list[PipelineField] | None = None,
                             include_native: bool = True) -> dict:
    """Compare explicitly paired events in two open captures/instances; no automatic ID matching.

    Captures are read in place and filenames checked before/after. Resource IDs are capture-local;
    raw ID differences are evidence to inspect, not semantic resource or API equivalence.
    """
    from .comparison import differences, incomplete

    def identity(capture):
        return {key: value for key, value in capture.items() if key != "event_id"}

    def read(selected, event):
        capture = invoke(selected, "get_capture_status")
        if not capture.get("loaded"):
            raise ToolError("No capture loaded in " + selected)
        state = invoke(selected, "get_pipeline_state", event_id=event)
        if fields:
            # Preserve failures even if the caller selects only a successful section.
            state = {key: state[key] for key in {*fields, "unavailable"} if key in state}
            missing = set(fields) - state.keys()
            if missing:
                state.setdefault("unavailable", {}).update({key: "Requested section missing" for key in missing})
        state.pop("event_id", None)
        if include_native:
            native = invoke(selected, "get_native_pipeline_state", event_id=event)
            native.pop("event_id", None)
            state["native_state"] = native
        after = invoke(selected, "get_capture_status")
        if identity(capture) != identity(after):
            raise ToolError("Capture changed while reading " + selected + "; rediscover and retry")
        return {"instance_id": selected, "event_id": event, "capture": capture, "state": state}

    before, after = read(instance_id, event_id), read(other_instance_id, other_event_id)
    # Recheck the first capture after querying the second instance too.
    if identity(before["capture"]) != identity(invoke(instance_id, "get_capture_status")):
        raise ToolError("First capture changed during comparison; rediscover and retry")
    changes = differences(before["state"], after["state"])
    complete = not incomplete(before["state"]) and not incomplete(after["state"])
    return artifacts.bound({"before": {k: v for k, v in before.items() if k != "state"},
        "after": {k: v for k, v in after.items() if k != "state"},
        "mapping_basis": "explicit caller-supplied event pair", "complete": complete,
        "equal": not changes if complete else None, "changes": changes,
        "comparison_scope": "Structural values only; native API conventions and resource IDs require explicit interpretation"})


@mcp.tool
def get_buffer_contents(instance_id: str, event_id: Event, resource_id: str,
                         offset: Offset = 0, length: Offset = 4096,
                         output: Literal["file", "inline"] = "file") -> dict:
    """Read buffer bytes at an explicit event. Default exports a binary artifact; length=0 exports all.

    Inline Base64 is limited to 4096 bytes. Offset and length are in bytes.
    """
    if output == "inline" and (length == 0 or length > 4096):
        raise ToolError("Inline reads require 1..4096 bytes; use output='file' for larger reads")
    path = artifacts.allocate(".bin") if output == "file" else None
    value = invoke(instance_id, "get_buffer_contents", event_id=event_id,
                   resource_id=resource_id, offset=offset, length=length,
                   output_path=str(path.resolve()) if path else None)
    if path:
        value.update(artifacts.describe(path))
    return value


@mcp.tool
def get_texture_info(instance_id: str, resource_id: str) -> dict:
    """Read texture dimensions, format, mip count, array size and sample count."""
    return invoke(instance_id, "get_texture_info", resource_id=resource_id)


@mcp.tool
def get_texture_data(instance_id: str, event_id: Event, resource_id: str,
                      mip: Offset = 0, slice: Offset = 0, sample: Offset = 0,
                      depth_slice: Offset | None = None) -> dict:
    """Export raw texture bytes at an explicit event as a binary artifact, including format metadata."""
    path = artifacts.allocate(".bin")
    value = invoke(instance_id, "get_texture_data", event_id=event_id, resource_id=resource_id,
                   mip=mip, slice=slice, sample=sample, depth_slice=depth_slice,
                   output_path=str(path.resolve()))
    return {**value, **artifacts.describe(path)}


@mcp.tool
def export_texture(instance_id: str, event_id: Event, resource_id: str,
                    mip: Offset = 0, slice: Offset = 0) -> dict:
    """Export one texture mip/slice as a viewable PNG artifact using RenderDoc's format conversion."""
    path = artifacts.allocate(".png")
    value = invoke(instance_id, "export_texture", event_id=event_id, resource_id=resource_id,
                   mip=mip, slice=slice, output_path=str(path.resolve()))
    return {**value, **artifacts.describe(path)}


@mcp.tool
def view_image_artifact(artifact_id: str) -> Image:
    """Display a previously exported PNG texture artifact as an MCP image."""
    import re
    if not re.fullmatch(r"[0-9a-f]{32}\.png", artifact_id):
        raise ToolError("Expected a PNG artifact_id returned by export_texture")
    return Image(path=str(artifacts.directory / artifact_id))


@mcp.tool
def read_artifact(artifact_id: str, offset: Offset = 0,
                   length: Annotated[int, Field(ge=1, le=16000)] = 8000) -> dict:
    """Read a bounded text chunk from a saved JSON result; offsets count Unicode characters."""
    try:
        return artifacts.read(artifact_id, offset, length)
    except (OSError, ValueError) as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool
def list_captures(instance_id: str, directory: str, offset: Offset = 0, limit: Limit = 50) -> dict:
    """Find .rdc files in a directory, newest first, with pagination."""
    return artifacts.bound(page_result(invoke(instance_id, "list_captures", directory=directory),
                                       "captures", offset, limit))


@mcp.tool
def open_capture(instance_id: str, capture_path: str) -> dict:
    """Open an .rdc file in the selected instance, replacing its currently loaded capture."""
    return invoke(instance_id, "open_capture", capture_path=capture_path)


def main():
    mcp.run(show_banner=False)


if __name__ == "__main__":
    main()
