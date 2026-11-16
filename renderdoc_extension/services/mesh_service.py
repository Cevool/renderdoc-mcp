"""Vertex input decoding and OBJ/PLY mesh export at a draw (Python 3.6)."""

import os
import struct

import renderdoc as rd

from ..utils.elements import decode, format_spec
from ..utils.geometry import primitives, write_mesh
from .replay_common import enum_name, event_query

MAX_EXPORT_INDICES = 16 * 1024 * 1024
MAX_RANGE_BYTES = 1024 * 1024 * 1024


def _spec(fmt):
    return format_spec(enum_name(fmt.type), enum_name(fmt.compType), fmt.compCount,
                       fmt.compByteWidth, fmt.BGRAOrder())


class _Ranges(object):
    """Read each buffer range once; vertex lookups slice cached bytes."""

    def __init__(self, controller):
        self.controller = controller
        self.cache = {}

    def load(self, key, rid, start, size):
        if size > MAX_RANGE_BYTES:
            raise ValueError("Vertex data range exceeds 1 GiB; draw indices are too scattered")
        self.cache[key] = (start, bytes(self.controller.GetBufferData(rid, start, size)) if size else b"")

    def get(self, key, address, size):
        start, data = self.cache[key]
        offset = address - start
        return data[offset:offset + size] if 0 <= offset and offset + size <= len(data) else b""


class _Inputs(object):
    """Input assembler view of a draw: index stream, vertex buffers and attributes."""

    def __init__(self, controller, action, instance):
        pipe = controller.GetPipelineState()
        self.controller, self.action, self.instance = controller, action, instance
        self.indexed = bool(action.flags & rd.ActionFlags.Indexed)
        self.topology = enum_name(pipe.GetPrimitiveTopology())
        ib = pipe.GetIBuffer()
        self.index_width = ib.byteStride if self.indexed else 0
        if self.indexed and self.index_width not in (1, 2, 4):
            raise ValueError("Unsupported index width %s" % ib.byteStride)
        self.index_buffer = ib.resourceId
        self.index_start = ib.byteOffset + action.indexOffset * self.index_width
        self.restart = None
        if self.indexed and pipe.IsRestartEnabled():
            self.restart = pipe.GetRestartIndex() & ((1 << (8 * self.index_width)) - 1)
        self.vbs = list(pipe.GetVBuffers())
        self.attributes = []
        for attribute in pipe.GetVertexInputs():
            fmt = attribute.format
            spec = _spec(fmt)
            self.attributes.append({
                "name": attribute.name, "format": fmt.Name(), "slot": attribute.vertexBuffer,
                "byte_offset": attribute.byteOffset, "per_instance": attribute.perInstance,
                "instance_rate": attribute.instanceRate, "used": attribute.used,
                "generic": attribute.genericEnabled,
                "generic_value": list(attribute.genericValue.floatValue) if attribute.genericEnabled else None,
                "decodable": spec is not None, "spec": spec})
        self.ranges = _Ranges(controller)

    def layout(self):
        return {"topology": self.topology, "indexed": self.indexed,
                "num_indices": self.action.numIndices, "num_instances": self.action.numInstances,
                "base_vertex": self.action.baseVertex, "vertex_offset": self.action.vertexOffset,
                "instance_offset": self.action.instanceOffset, "index_offset": self.action.indexOffset,
                "index_buffer": {"resource_id": str(self.index_buffer), "byte_offset": self.index_start,
                                 "width": self.index_width, "restart_index": self.restart}
                if self.indexed else None,
                "vertex_buffers": [{"slot": i, "resource_id": str(vb.resourceId),
                                    "byte_offset": vb.byteOffset, "stride": vb.byteStride}
                                   for i, vb in enumerate(self.vbs) if vb.resourceId != rd.ResourceId.Null()],
                "attributes": [{k: v for k, v in a.items() if k != "spec"} for a in self.attributes]}

    def stream(self, first, end):
        """[(raw index or None for restart, vertex index or None)] for index rows first..end-1."""
        if end <= first:
            return []
        if not self.indexed:
            return [(None, self.action.vertexOffset + row) for row in range(first, end)]
        width = self.index_width
        data = bytes(self.controller.GetBufferData(self.index_buffer, self.index_start + first * width,
                                                   (end - first) * width))
        if len(data) != (end - first) * width:
            raise ValueError("Short index buffer read")
        code = {1: "B", 2: "H", 4: "I"}[width]
        values = struct.unpack("<%d%s" % (end - first, code), data)
        return [(v, None) if v == self.restart else (v, v + self.action.baseVertex) for v in values]

    def element(self, attribute, vertex):
        if attribute["per_instance"]:
            rate = attribute["instance_rate"]
            return self.action.instanceOffset + (self.instance // rate if rate else 0)
        return vertex

    def preload(self, attributes, vertices):
        """Read the byte range each attribute's slot needs for these vertices, once per slot."""
        spans = {}
        for attribute in attributes:
            if attribute["generic"] or attribute["spec"] is None:
                continue
            elements = [self.element(attribute, v) for v in vertices] or [0]
            lo, hi = min(elements), max(elements)
            tail = attribute["byte_offset"] + attribute["spec"]["bytes"]
            span = spans.setdefault(attribute["slot"], [lo, hi, tail])
            span[0], span[1], span[2] = min(span[0], lo), max(span[1], hi), max(span[2], tail)
        for slot, (lo, hi, tail) in spans.items():
            vb = self.vbs[slot] if slot < len(self.vbs) else None
            if vb is None or vb.resourceId == rd.ResourceId.Null():
                continue
            start = vb.byteOffset + lo * vb.byteStride
            self.ranges.load(slot, vb.resourceId, start, (hi - lo) * vb.byteStride + tail)

    def value(self, attribute, vertex):
        if attribute["generic"]:
            return attribute["generic_value"]
        if attribute["spec"] is None:
            return None
        slot = attribute["slot"]
        if slot not in self.ranges.cache:
            return None
        vb = self.vbs[slot]
        address = vb.byteOffset + self.element(attribute, vertex) * vb.byteStride + attribute["byte_offset"]
        return decode(attribute["spec"], self.ranges.get(slot, address, attribute["spec"]["bytes"]))


def _pick(attributes, explicit, needles, fallback=None):
    usable = [a for a in attributes if a["used"] and (a["decodable"] or a["generic"])]
    if explicit:
        for attribute in attributes:
            if attribute["name"].lower() == explicit.lower():
                if not attribute["decodable"] and not attribute["generic"]:
                    raise ValueError("Attribute %s has an undecodable format %s" % (explicit, attribute["format"]))
                return attribute
        raise ValueError("No vertex attribute named %s" % explicit)
    for needle in needles:
        for attribute in usable:
            if needle in attribute["name"].lower():
                return attribute
    return fallback(usable) if fallback else None


class MeshService(object):
    def __init__(self, ctx, invoke_fn, state):
        self.ctx = ctx
        self._invoke = invoke_fn
        self.state = state

    def _draw(self, event_id, instance):
        action = self.ctx.GetAction(event_id)
        if action is None or not action.flags & rd.ActionFlags.Drawcall:
            raise ValueError("Event %s is not a draw" % event_id)
        if not 0 <= instance < max(1, action.numInstances):
            raise ValueError("Invalid instance index %s (draw has %s)" % (instance, action.numInstances))
        return action

    def get_vertex_inputs(self, event_id, instance=0, offset=0, limit=50, attributes=None):
        action = self._draw(event_id, instance)
        names = set(a.lower() for a in attributes or [])

        def read(controller):
            inputs = _Inputs(controller, action, instance)
            chosen = [a for a in inputs.attributes if (a["name"].lower() in names if names else a["used"])]
            if names and len(chosen) != len(names):
                known = [a["name"] for a in inputs.attributes]
                raise ValueError("Unknown attribute in %s; available: %s" % (sorted(names), known))
            end = min(action.numIndices, offset + limit)
            rows = inputs.stream(offset, end)
            inputs.preload(chosen, [v for _, v in rows if v is not None])
            records = []
            for row, (raw, vertex) in enumerate(rows):
                record = {"row": offset + row, "index": raw, "vertex": vertex}
                if vertex is None:
                    record["restart"] = True
                else:
                    record["values"] = {a["name"]: inputs.value(a, vertex) for a in chosen}
                records.append(record)
            return {"instance": instance, "layout": inputs.layout(), "rows": records,
                    "offset": offset, "total": action.numIndices, "returned": len(records),
                    "next_offset": end if end < action.numIndices else None,
                    "scope": ("Input assembler data as fetched by this draw (before the vertex shader). "
                              "values is null where the format is not decodable or the read is short.")}
        return event_query(self.ctx, self._invoke, self.state, event_id, read)

    def export_mesh(self, event_id, output_path, fmt="obj", stage="vsin", instance=0, view=0,
                    space="ndc", position_attribute=None, normal_attribute=None, uv_attribute=None):
        action = self._draw(event_id, instance)
        if os.path.exists(output_path):
            raise ValueError("output_path already exists")

        def read(controller):
            if stage == "vsin":
                source = self._vsin(controller, action, instance, position_attribute,
                                    normal_attribute, uv_attribute)
            else:
                source = self._post(controller, instance, view, stage, space)
            stream = source["stream"]
            kind, prims = primitives(source["topology"], [v for _, v in stream])
            order, remap = [], {}
            for _, vertex in stream:
                if vertex is not None and vertex not in remap:
                    remap[vertex] = len(order)
                    order.append(vertex)
            vertices = [source["vertex"](v) for v in order]
            header = ["RenderDoc capture %s" % os.path.basename(self.ctx.GetCaptureFilename() or ""),
                      "event %d, %s, instance %d, topology %s" % (event_id, stage, instance, source["topology"])]
            written = write_mesh(output_path, fmt, vertices, kind,
                                 [tuple(remap[i] for i in prim) for prim in prims], header)
            degenerate = sum(1 for prim in prims if len(set(prim)) < len(prim))
            result = dict(source["summary"], **written)
            result.update(source.get("extra", dict)())
            result.update(path=os.path.abspath(output_path), format=fmt, stage=stage,
                          instance=instance, topology=source["topology"], index_rows=len(stream),
                          degenerate_primitives=degenerate)
            return result
        return event_query(self.ctx, self._invoke, self.state, event_id, read)

    def _vsin(self, controller, action, instance, position, normal, uv):
        if action.numIndices > MAX_EXPORT_INDICES:
            raise ValueError("Draw has more than %d indices" % MAX_EXPORT_INDICES)
        inputs = _Inputs(controller, action, instance)

        def first_vector(usable):
            return next((a for a in usable if a["spec"] and a["spec"]["count"] >= 3
                         and a["spec"].get("kind") == "float"), None)
        pos = _pick(inputs.attributes, position, ["position", "attribute0"], first_vector)
        if pos is None:
            raise ValueError("No position attribute found; pass position_attribute")
        nrm = _pick(inputs.attributes, normal, ["normal"])
        tex = _pick(inputs.attributes, uv, ["texcoord0", "texcoord", "uv0", "uv"])
        stream = inputs.stream(0, action.numIndices)
        chosen = [a for a in (pos, nrm, tex) if a is not None]
        inputs.preload(chosen, [v for _, v in stream if v is not None])

        def vertex(index):
            value = {"position": inputs.value(pos, index) or []}
            if nrm is not None:
                value["normal"] = inputs.value(nrm, index) or []
            if tex is not None:
                value["uv"] = inputs.value(tex, index) or []
            return value
        return {"topology": inputs.topology, "stream": stream, "vertex": vertex,
                "summary": {"space": "vertex shader input (object space as fed to the shader)",
                            "attributes": {"position": pos["name"], "normal": nrm and nrm["name"],
                                           "uv": tex and tex["name"]},
                            "available_attributes": [{"name": a["name"], "format": a["format"], "used": a["used"]}
                                                     for a in inputs.attributes]}}

    @staticmethod
    def _post(controller, instance, view, stage, space):
        mesh = controller.GetPostVSData(instance, view, {"vsout": rd.MeshDataStage.VSOut,
                                                          "gsout": rd.MeshDataStage.GSOut}[stage])
        if mesh.status:
            raise ValueError("Post-transform data unavailable: %s" % mesh.status)
        if mesh.vertexResourceId == rd.ResourceId.Null() or not mesh.numIndices:
            raise ValueError("No %s output for this draw" % stage)
        if mesh.numIndices > MAX_EXPORT_INDICES:
            raise ValueError("Output has more than %d indices" % MAX_EXPORT_INDICES)
        spec = _spec(mesh.format)
        if spec is None or spec.get("kind") != "float":
            raise ValueError("Position format %s is not decodable" % mesh.format.Name())
        width = mesh.indexByteStride if mesh.indexResourceId != rd.ResourceId.Null() else 0
        if width:
            if width not in (1, 2, 4):
                raise ValueError("Unsupported index width %s" % width)
            data = bytes(controller.GetBufferData(mesh.indexResourceId, mesh.indexByteOffset,
                                                  mesh.numIndices * width))
            if len(data) != mesh.numIndices * width:
                raise ValueError("Short post-transform index read")
            restart = (mesh.restartIndex & ((1 << (8 * width)) - 1)) if mesh.allowRestart else None
            values = struct.unpack("<%d%s" % (mesh.numIndices, {1: "B", 2: "H", 4: "I"}[width]), data)
            stream = [(v, None) if v == restart else (v, v + mesh.baseVertex) for v in values]
        else:
            stream = [(None, i) for i in range(mesh.numIndices)]
        used = [v for _, v in stream if v is not None]
        lo, hi = min(used), max(used)
        ranges = _Ranges(controller)
        start = mesh.vertexByteOffset + lo * mesh.vertexByteStride
        ranges.load(0, mesh.vertexResourceId, start, (hi - lo) * mesh.vertexByteStride + spec["bytes"])
        clipped = [0]

        def vertex(index):
            values = decode(spec, ranges.get(0, mesh.vertexByteOffset + index * mesh.vertexByteStride,
                                             spec["bytes"])) or []
            if space == "ndc" and len(values) == 4:
                w = values[3]
                if not isinstance(w, float) or w <= 0:
                    clipped[0] += 1
                    return {"position": ["nan"] * 3}
                values = [values[0] / w, values[1] / w, values[2] / w] \
                    if all(isinstance(v, float) for v in values[:3]) else ["nan"] * 3
            return {"position": values[:3]}
        summary = {"space": "NDC (x/w, y/w, z/w)" if space == "ndc" else "clip space xyz (w dropped)",
                   "position_format": mesh.format.Name()}
        return {"topology": enum_name(mesh.topology), "stream": stream, "vertex": vertex,
                "summary": summary, "extra": lambda: {"vertices_behind_camera": clipped[0]}
                if space == "ndc" else {}}
