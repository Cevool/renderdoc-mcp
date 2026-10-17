"""Deep replay queries using the already-open, serialized replay session."""

import base64
import math
import struct
import time

import renderdoc as rd

from ..utils import Parsers, Serializers
from ..utils.native import native_value, result_status


class AnalysisService:
    def __init__(self, ctx, invoke_fn):
        self.ctx = ctx
        self._invoke = invoke_fn

    def _query(self, event_id, read):
        if not self.ctx.IsCaptureLoaded():
            raise ValueError("No capture loaded")
        result = {}

        def callback(controller):
            if event_id is not None:
                if event_id != 0 and self.ctx.GetAction(event_id) is None:
                    raise ValueError("No action at event_id %s" % event_id)
                controller.SetFrameEvent(event_id, True)
            result.update(read(controller))
        self._invoke(callback)
        return result

    def get_native_pipeline_state(self, event_id):
        def read(controller):
            api = str(controller.GetAPIProperties().pipelineType)
            sections = {
                "D3D11": ("GetD3D11PipelineState", ["rasterizer", "outputMerger"]),
                "D3D12": ("GetD3D12PipelineState", ["rasterizer", "outputMerger"]),
                "Vulkan": ("GetVulkanPipelineState", ["rasterizer", "depthStencil", "colorBlend",
                                                      "multisample", "viewportScissor", "currentPass"]),
                "OpenGL": ("GetGLPipelineState", ["rasterizer", "depthState", "stencilState", "framebuffer"]),
            }
            api_name = api.rsplit(".", 1)[-1]
            if api_name not in sections:
                return {"event_id": event_id, "api": api, "status": "unsupported", "complete": False}
            getter, names = sections[api_name]
            native = getattr(controller, getter)()
            value = {"event_id": event_id, "api": api, "native": {}}
            for name in names:
                try:
                    value["native"][name] = native_value(getattr(native, name))
                except Exception as exc:
                    value.setdefault("unavailable", {})[name] = str(exc)
            return result_status(value)
        return self._query(event_id, read)

    def get_resource_usage(self, resource_id):
        def read(controller):
            rid = Parsers.parse_resource_id(resource_id, controller)
            usages = []
            for usage in controller.GetUsage(rid):
                usages.append({"event_id": usage.eventId, "usage": str(usage.usage),
                               "view_id": str(usage.view) if hasattr(usage, "view") else None})
            return result_status({"resource_id": resource_id, "usages": usages,
                "view_scope": "view_id is null when this RenderDoc binding exposes only event/usage"}, usages)
        return self._query(None, read)

    def get_post_vs_data(self, event_id, stage="vs", instance=0, view=0, offset=0, limit=50,
                         include_bytes=False):
        def read(controller):
            action = self.ctx.GetAction(event_id)
            if not action.flags & rd.ActionFlags.Drawcall:
                raise ValueError("Post-VS requires a draw action")
            if not 0 <= instance < max(1, action.numInstances):
                raise ValueError("Invalid instance index")
            mesh_stage = {"vs": rd.MeshDataStage.VSOut, "gs": rd.MeshDataStage.GSOut}[stage]
            mesh = controller.GetPostVSData(instance, view, mesh_stage)
            result = {"event_id": event_id, "stage": stage, "instance": instance, "view": view,
                      "layout": native_value(mesh), "vertices": [], "offset": offset}
            if mesh.status:
                result.update(status="error", complete=False, error=str(mesh.status))
                return result
            if mesh.vertexResourceId == rd.ResourceId.Null():
                result.update(status="empty", complete=True, total=0, next_offset=None)
                return result
            if not 1 <= mesh.vertexByteStride <= 16384:
                raise ValueError("Invalid or oversized Post-VS stride")
            total = mesh.numIndices
            end = min(total, offset + limit)
            indices = list(range(offset, end))
            if mesh.indexResourceId != rd.ResourceId.Null() and mesh.indexByteStride:
                width = mesh.indexByteStride
                if width not in (1, 2, 4):
                    raise ValueError("Unsupported Post-VS index width")
                data = bytes(controller.GetBufferData(mesh.indexResourceId,
                    mesh.indexByteOffset + offset * width, max(0, end - offset) * width)) if end > offset else b""
                if len(data) != max(0, end - offset) * width:
                    raise ValueError("Short Post-VS index read")
                indices = [v[0] for v in struct.iter_unpack("<" + {1: "B", 2: "H", 4: "I"}[width], data)]
            fmt = mesh.format
            decode = (not fmt.Special() and fmt.compType == rd.CompType.Float
                      and fmt.compByteWidth in (2, 4, 8) and 1 <= fmt.compCount <= 4)
            for row, index in enumerate(indices):
                vertex = {"row": offset + row, "index": index}
                if mesh.allowRestart and mesh.indexByteStride and index == (mesh.restartIndex & ((1 << (8 * mesh.indexByteStride)) - 1)):
                    vertex["restart"] = True
                else:
                    resolved = index + mesh.baseVertex if mesh.indexByteStride else index
                    if resolved < 0:
                        raise ValueError("Negative Post-VS vertex index")
                    raw = bytes(controller.GetBufferData(mesh.vertexResourceId,
                        mesh.vertexByteOffset + resolved * mesh.vertexByteStride, mesh.vertexByteStride))
                    vertex["vertex_index"] = resolved
                    if len(raw) != mesh.vertexByteStride:
                        vertex["error"] = "Short Post-VS vertex read"
                    elif decode and fmt.compCount * fmt.compByteWidth <= len(raw):
                        values = struct.unpack_from("<" + {2: "e", 4: "f", 8: "d"}[fmt.compByteWidth] * fmt.compCount, raw)
                        vertex["position"] = [v if math.isfinite(v) else str(v) for v in values]
                        vertex["finite"] = all(math.isfinite(v) for v in values)
                        vertex["w_zero"] = len(values) == 4 and values[3] == 0
                    else:
                        vertex["unavailable"] = "Position decoding requires a regular floating point format"
                    if include_bytes:
                        vertex["raw_base64"] = base64.b64encode(raw).decode("ascii")
                result["vertices"].append(vertex)
            result.update(total=total, returned=len(result["vertices"]),
                          next_offset=end if end < total else None, truncated=end < total)
            result["validation_scope"] = "Only returned vertices; finite positions do not prove visible nondegenerate primitives"
            return result_status(result, result["vertices"])
        return self._query(event_id, read)

    def get_pixel_history(self, event_id, resource_id, x, y, mip=0, slice=0, sample=0,
                          offset=0, limit=50):
        def read(controller):
            if not controller.GetAPIProperties().pixelHistory:
                return {"status": "unsupported", "complete": False, "reason": "API/driver reports no pixel history support"}
            rid = Parsers.parse_resource_id(resource_id, controller)
            texture = next((t for t in controller.GetTextures() if t.resourceId == rid), None)
            if texture is None:
                raise ValueError("Unknown texture")
            if (not 0 <= mip < texture.mips or not 0 <= x < max(1, texture.width >> mip)
                    or not 0 <= y < max(1, texture.height >> mip)
                    or not 0 <= slice < max(texture.arraysize, max(1, texture.depth >> mip))
                    or not 0 <= sample < max(1, texture.msSamp)):
                raise ValueError("Invalid pixel/subresource coordinates")
            sub = rd.Subresource()
            sub.mip, sub.slice, sub.sample = mip, slice, sample
            history = controller.PixelHistory(rid, x, y, sub, rd.CompType.Typeless)
            # The replay API limits history to the current event; retain the explicit bound too.
            history = [h for h in history if h.eventId <= event_id]
            records = [native_value(h) for h in history[offset:offset + limit]]
            next_offset = offset + len(records) if offset + len(records) < len(history) else None
            return result_status({"event_id": event_id, "resource_id": resource_id, "x": x, "y": y,
                "coordinate_origin": "top-left", "mip": mip, "slice": slice, "sample": sample,
                "history": records, "total": len(history), "offset": offset,
                "next_offset": next_offset, "truncated": next_offset is not None}, records)
        return self._query(event_id, read)

    def get_api_events(self, event_id_min, event_id_max, offset=0, limit=50,
                       include_parameters=False):
        def read(controller):
            structured = self.ctx.GetStructuredFile()
            events = {}
            def visit(actions):
                for action in actions:
                    for event in action.events:
                        if event_id_min <= event.eventId <= event_id_max:
                            events[event.eventId] = (event, action)
                    visit(action.children)
            visit(controller.GetRootActions())
            selected = sorted(events)[offset:offset + limit]
            records = []
            for event_id in selected:
                event, action = events[event_id]
                record = {"event_id": event_id, "action_event_id": action.eventId,
                          "action_flags": Serializers.serialize_flags(action.flags), "chunk_index": event.chunkIndex}
                if event.chunkIndex >= len(structured.chunks):
                    record["unavailable"] = "No structured chunk for this event"
                else:
                    chunk = structured.chunks[event.chunkIndex]
                    record["name"] = chunk.name
                    if include_parameters:
                        budget = [2048]
                        record["parameters"] = self._structured(chunk, budget)
                records.append(record)
            next_offset = offset + len(records) if offset + len(records) < len(events) else None
            return result_status({"events": records, "total": len(events), "offset": offset,
                "event_id_min": event_id_min, "event_id_max": event_id_max,
                "next_offset": next_offset, "truncated": next_offset is not None,
                "provenance": "RenderDoc structured chunks; serialized fields are API/version dependent, not inferred load/store rules"}, records)
        return self._query(None, read)

    @staticmethod
    def _structured(obj, budget, depth=0):
        value = {"name": obj.name, "type": obj.type.name, "base_type": str(obj.type.basetype),
                 "flags": str(obj.type.flags)}
        budget[0] -= 1
        if depth >= 12 or budget[0] <= 0:
            return dict(value, truncated=True)
        kind = obj.type.basetype
        if kind in (rd.SDBasic.Struct, rd.SDBasic.Array, rd.SDBasic.Chunk):
            children = []
            total = obj.NumChildren()
            for i in range(min(total, 128)):
                if budget[0] <= 0:
                    break
                children.append(AnalysisService._structured(obj.GetChild(i), budget, depth + 1))
            value.update(children=children, total_children=total, truncated=len(children) < total)
        elif kind == rd.SDBasic.Resource:
            value["resource_id"] = str(obj.AsResourceId())
        elif kind == rd.SDBasic.Float:
            value["value"] = native_value(obj.data.basic.d)
        elif kind == rd.SDBasic.Boolean:
            value["value"] = obj.data.basic.b
        elif kind == rd.SDBasic.SignedInteger:
            value["value"] = obj.data.basic.i
        elif kind in (rd.SDBasic.UnsignedInteger, rd.SDBasic.Enum, rd.SDBasic.GPUAddress):
            value["value"] = obj.data.basic.u
            value["display"] = obj.data.string
        elif kind == rd.SDBasic.Character:
            value["value"] = obj.data.basic.c
        elif kind == rd.SDBasic.String:
            text = obj.data.string
            value.update(value=text[:4096], truncated=len(text) > 4096)
        elif kind == rd.SDBasic.Null:
            value["value"] = None
        else:
            value["unavailable"] = "Opaque buffer or unhandled structured type"
        return value

    def debug_shader(self, event_id, stage, invocation, max_steps=1000):
        def read(controller):
            if not controller.GetAPIProperties().shaderDebugging:
                return {"status": "unsupported", "complete": False, "reason": "API/driver reports no shader debugger support"}
            trace = None
            result = {"event_id": event_id, "stage": stage, "invocation": invocation, "states": []}
            try:
                if stage == "vertex":
                    trace = controller.DebugVertex(invocation["vertex"], invocation.get("instance", 0),
                        invocation["index"], invocation.get("view", 0))
                elif stage == "pixel":
                    inputs = rd.DebugPixelInputs()
                    # Keep the native constructor defaults for omitted fields. NoPreference is
                    # not a module-level constant in RenderDoc's Python binding.
                    for name in ("sample", "primitive", "view"):
                        if name in invocation:
                            setattr(inputs, name, invocation[name])
                    trace = controller.DebugPixel(invocation["x"], invocation["y"], inputs)
                else:
                    trace = controller.DebugThread(invocation["group"], invocation["thread"])
                if trace is None or trace.debugger is None:
                    result.update(status="error", complete=False, reason="No debuggable invocation; inspect shader reflection and coordinates")
                    return result
                result["inputs"] = Serializers.serialize_variables(trace.inputs)
                result["constant_blocks"] = Serializers.serialize_variables(trace.constantBlocks)
                result["source_mapping"] = native_value(trace.sourceVars)
                deadline = time.monotonic() + 10
                finished = False
                # ContinueDebug batches have implementation-defined sizes. Bound both calls and saved steps.
                for _ in range(64):
                    states = controller.ContinueDebug(trace.debugger)
                    if not states:
                        finished = True
                        break
                    remaining = max_steps - len(result["states"])
                    for state in states[:remaining]:
                        result["states"].append({"step": state.stepIndex, "next_instruction": state.nextInstruction,
                            "flags": str(state.flags), "callstack": list(state.callstack),
                            "changes": [{"before": Serializers.serialize_variables([c.before])[0],
                                         "after": Serializers.serialize_variables([c.after])[0]} for c in state.changes]})
                    if len(states) > remaining or len(result["states"]) >= max_steps or time.monotonic() >= deadline:
                        break
                result.update(finished=finished, truncated=not finished, returned=len(result["states"]),
                              trace_released=True,
                              hint="Read saved trace with read_artifact. A truncated trace is a prefix; no live debugger is retained.")
                return result_status(result)
            finally:
                if trace is not None:
                    controller.FreeTrace(trace)
        return self._query(event_id, read)
