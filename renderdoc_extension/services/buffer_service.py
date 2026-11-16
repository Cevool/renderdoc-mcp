"""Event-bound buffer element decoding and before/after comparison (Python 3.6)."""

import struct

import renderdoc as rd

from ..utils import Helpers, Parsers
from ..utils.elements import decode, decode_fields, format_spec, parse_type
from .replay_common import digest, enum_name, event_query

MAX_FIELDS = 256
SCALARS_BYTES = {"double": 8, "int64": 8, "uint64": 8, "half": 2, "int16": 2, "uint16": 2,
                 "int8": 1, "uint8": 1}
SCALAR_TYPES ={"Float": "float", "Half": "half", "Double": "double", "SInt": "int", "UInt": "uint",
                "Bool": "bool", "SShort": "int16", "UShort": "uint16", "SByte": "int8",
                "UByte": "uint8", "SLong": "int64", "ULong": "uint64", "Enum": "uint"}


def _type_name(base, count):
    if base in ("int8", "uint8", "int16", "uint16", "int64", "uint64"):
        return base + ("x%d" % count if count > 1 else "")
    return base + ("%d" % count if count > 1 else "")


def compile_fields(fields):
    """[{"name","offset","type"}] -> fields with parsed specs; stride hint = furthest byte."""
    compiled = []
    for field in fields:
        spec = parse_type(field["type"])
        compiled.append({"name": field["name"], "offset": int(field["offset"]), "type": spec["type"],
                         "spec": spec})
    return compiled


def _flatten(ctype, name, base, out):
    """Shader reflection type -> flat fields. Matrices become one vector field per row/column."""
    if len(out) >= MAX_FIELDS:
        return False
    elements = max(1, ctype.elements)
    stride = ctype.arrayByteStride
    if ctype.members:
        for element in range(elements if stride else 1):
            prefix = name + ("[%d]" % element if elements > 1 and stride else "")
            for member in ctype.members:
                child = (prefix + "." if prefix else "") + member.name
                if not _flatten(member.type, child, base + element * stride + member.byteOffset, out):
                    return False
        return True
    scalar = SCALAR_TYPES.get(enum_name(ctype.baseType))
    if scalar is None:
        return True
    rows, cols = max(1, ctype.rows), max(1, ctype.columns)
    vectors = [(name, base, cols)]
    if rows > 1:
        row_major = bool(ctype.flags & rd.ShaderVariableFlags.RowMajor)
        count, length = (rows, cols) if row_major else (cols, rows)
        step = ctype.matrixByteStride or SCALARS_BYTES.get(scalar, 4) * length
        label = "row" if row_major else "col"
        vectors = [("%s.%s%d" % (name, label, i), base + i * step, length) for i in range(count)]
    for element in range(elements):
        for label, offset, count in vectors:
            if len(out) >= MAX_FIELDS:
                return False
            spec = parse_type(_type_name(scalar, count))
            step = stride or spec["bytes"]
            out.append({"name": label + ("[%d]" % element if elements > 1 else ""),
                        "offset": offset + element * step, "type": spec["type"], "spec": spec})
    return True


def _public_layout(fields):
    return [{"name": f["name"], "offset": f["offset"], "type": f["type"]} for f in fields]


class BufferService(object):
    def __init__(self, ctx, invoke_fn, state):
        self.ctx = ctx
        self._invoke = invoke_fn
        self.state = state

    @staticmethod
    def _buffer(controller, resource_id):
        rid = Parsers.parse_resource_id(resource_id, controller)
        for buf in controller.GetBuffers():
            if buf.resourceId == rid:
                return rid, buf
        raise ValueError("Not a buffer: %s" % resource_id)

    @staticmethod
    def bound_layouts(controller, rid):
        """Ways the buffer is bound at the current event, each with a decoding layout."""
        pipe = controller.GetPipelineState()
        layouts = []
        stages = [rd.ShaderStage.Compute] if pipe.GetShader(rd.ShaderStage.Compute) != rd.ResourceId.Null() \
            else Helpers.get_all_shader_stages()
        for stage in stages:
            if pipe.GetShader(stage) == rd.ResourceId.Null():
                continue
            reflection = pipe.GetShaderReflection(stage)
            for kind, used, declared in (
                    ("read_write", pipe.GetReadWriteResources(stage, False),
                     reflection.readWriteResources if reflection else []),
                    ("read_only", pipe.GetReadOnlyResources(stage, False),
                     reflection.readOnlyResources if reflection else [])):
                for item in used:
                    descriptor = item.descriptor
                    if descriptor.resource != rid:
                        continue
                    index = item.access.index
                    resource = declared[index] if 0 <= index < len(declared) else None
                    layout = {"source": "%s %s binding" % (enum_name(stage).lower(), kind),
                              "binding_name": resource.name if resource is not None else None,
                              "byte_offset": descriptor.byteOffset, "view_bytes": descriptor.byteSize,
                              "element_stride": descriptor.elementByteSize or None,
                              "view_format": descriptor.format.Name()}
                    fields, spec = [], None
                    fmt = descriptor.format
                    if enum_name(fmt.type) != "Undefined" and fmt.compCount:
                        # Typed and raw (typeless) views decode through the view format.
                        spec = format_spec(enum_name(fmt.type), enum_name(fmt.compType), fmt.compCount,
                                           fmt.compByteWidth, fmt.BGRAOrder())
                    if spec is not None:
                        fields = [{"name": layout["binding_name"] or "value", "offset": 0,
                                   "type": spec["type"], "spec": spec}]
                        layout["element_stride"] = layout["element_stride"] or spec["bytes"]
                    elif resource is not None:
                        complete = _flatten(resource.variableType, "", 0, fields)
                        if fields and not fields[0]["name"]:
                            fields[0]["name"] = resource.name or "value"
                        for field in fields:
                            if field["name"].startswith("["):
                                field["name"] = (resource.name or "value") + field["name"]
                        layout["fields_truncated"] = not complete
                    layout["fields"] = fields
                    layouts.append(layout)
        attributes = [a for a in pipe.GetVertexInputs() if not a.genericEnabled]
        for slot, vb in enumerate(pipe.GetVBuffers()):
            if vb.resourceId != rid:
                continue
            fields = []
            for attribute in attributes:
                fmt = attribute.format
                spec = format_spec(enum_name(fmt.type), enum_name(fmt.compType), fmt.compCount,
                                   fmt.compByteWidth, fmt.BGRAOrder())
                if attribute.vertexBuffer == slot and spec is not None:
                    fields.append({"name": attribute.name, "offset": attribute.byteOffset,
                                   "type": spec["type"], "spec": spec})
            layouts.append({"source": "vertex buffer slot %d" % slot, "byte_offset": vb.byteOffset,
                            "element_stride": vb.byteStride or None, "fields": fields})
        ib = pipe.GetIBuffer()
        if ib.resourceId == rid and ib.byteStride in (1, 2, 4):
            spec = parse_type({1: "uint8", 2: "uint16", 4: "uint"}[ib.byteStride])
            layouts.append({"source": "index buffer", "byte_offset": ib.byteOffset,
                            "element_stride": ib.byteStride,
                            "fields": [{"name": "index", "offset": 0, "type": spec["type"], "spec": spec}]})
        return layouts

    @staticmethod
    def _covers(layout, offset):
        start, size = layout["byte_offset"], layout.get("view_bytes") or 0
        return start <= offset and (not size or offset < start + size)

    def read_buffer_elements(self, event_id, resource_id, fields=None, element_stride=None,
                             byte_offset=None, start=0, count=64, include_hex=False, binding=None):
        def read(controller):
            rid, buf = self._buffer(controller, resource_id)
            layouts = self.bound_layouts(controller, rid)
            chosen = None
            if binding is not None:
                if not 0 <= binding < len(layouts):
                    raise ValueError("binding %s is out of range; this event has %d bindings of %s" % (
                        binding, len(layouts), resource_id))
                chosen = layouts[binding]
            if fields:
                layout_fields = compile_fields(fields)
                stride = element_stride or max(f["offset"] + f["spec"]["bytes"] for f in layout_fields)
                source = "caller fields"
            else:
                if chosen is None:
                    # One buffer can be bound several ways (e.g. a pooled index buffer that is also
                    # an SRV): prefer a binding whose range holds an explicit byte_offset.
                    usable = [item for item in layouts if item["fields"] and item.get("element_stride")]
                    if byte_offset is not None:
                        usable = [item for item in usable if self._covers(item, byte_offset)] or usable
                    chosen = usable[0] if usable else None
                if chosen is None or not chosen["fields"] or not chosen.get("element_stride") and not element_stride:
                    if not element_stride:
                        raise ValueError("No decodable binding of %s at event %s; pass fields and "
                                         "element_stride (bindings: %s)" % (
                                             resource_id, event_id, [item["source"] for item in layouts] or "none"))
                    layout_fields, stride, source = [], element_stride, "raw bytes"
                else:
                    layout_fields, source = chosen["fields"], chosen["source"]
                    stride = element_stride or chosen["element_stride"]
            base = byte_offset if byte_offset is not None else (chosen["byte_offset"] if chosen else 0)
            if base > buf.length:
                raise ValueError("byte_offset %d is beyond the buffer (%d bytes)" % (base, buf.length))
            end = buf.length
            if chosen is not None and chosen.get("view_bytes") and self._covers(chosen, base):
                end = min(end, chosen["byte_offset"] + chosen["view_bytes"])
            total = (end - base) // stride
            first = min(start, total)
            last = min(total, first + count)
            raw = bytes(controller.GetBufferData(rid, base + first * stride, (last - first) * stride)) \
                if last > first else b""
            elements = []
            for row in range(last - first):
                chunk = raw[row * stride:(row + 1) * stride]
                element = {"index": first + row, "byte_offset": base + (first + row) * stride}
                if layout_fields:
                    element["values"] = decode_fields(layout_fields, chunk)
                if include_hex or not layout_fields:
                    element["hex"] = chunk.hex()
                elements.append(element)
            result = {"resource_id": resource_id, "buffer_bytes": buf.length, "byte_offset": base,
                      "element_stride": stride, "layout_source": source,
                      "layout": _public_layout(layout_fields), "total_elements": total,
                      "binding_index": layouts.index(chosen) if chosen in layouts else None,
                      "start": first, "returned": len(elements),
                      "next_start": last if last < total else None, "elements": elements,
                      "bindings": [{k: v for k, v in item.items() if k != "fields"} for item in layouts],
                      "scope": ("Contents after event_id executes. Reflected layouts depend on the "
                                "shader's reflection; stripped shaders often expose only the element "
                                "type, so pass fields for real structures.")}
            if chosen is not None and chosen.get("fields_truncated"):
                result["layout_truncated"] = True
            return result
        return event_query(self.ctx, self._invoke, self.state, event_id, read, require_action=False)

    def compare_buffer_contents(self, event_id, resource_id, baseline_event_id=None, offset=0,
                                length=0, element_stride=4, fields=None, limit=100,
                                max_bytes=64 * 1024 * 1024):
        baseline = event_id - 1 if baseline_event_id is None else baseline_event_id
        if baseline < 0 or baseline == event_id:
            raise ValueError("baseline_event_id must be a different event >= 0")
        layout_fields = compile_fields(fields) if fields else []
        if layout_fields:
            element_stride = max(element_stride, max(f["offset"] + f["spec"]["bytes"] for f in layout_fields))

        def read(controller):
            rid, buf = self._buffer(controller, resource_id)
            if offset > buf.length:
                raise ValueError("offset is beyond the buffer")
            size = buf.length - offset if length == 0 else min(length, buf.length - offset)
            truncated = size > max_bytes
            size = min(size, max_bytes)
            after = bytes(controller.GetBufferData(rid, offset, size))
            controller.SetFrameEvent(baseline, True)
            before = bytes(controller.GetBufferData(rid, offset, size))
            if len(before) != size or len(after) != size:
                raise ValueError("Short buffer read (%d/%d of %d bytes)" % (len(before), len(after), size))
            changed, ranges, details = 0, [], []
            block = max(element_stride, (65536 // element_stride) * element_stride)
            for start in range(0, size, block):
                if before[start:start + block] == after[start:start + block]:
                    continue
                for pos in range(start, min(size, start + block), element_stride):
                    a, b = before[pos:pos + element_stride], after[pos:pos + element_stride]
                    if a == b:
                        continue
                    changed += 1
                    index = pos // element_stride
                    if ranges and ranges[-1][1] == index - 1:
                        ranges[-1][1] = index
                    elif len(ranges) < 256:
                        ranges.append([index, index])
                    if len(details) < limit:
                        entry = {"element": index, "byte_offset": offset + pos}
                        if layout_fields:
                            entry["before"] = decode_fields(layout_fields, a)
                            entry["after"] = decode_fields(layout_fields, b)
                        elif element_stride == 4 and len(a) == 4 and len(b) == 4:
                            entry["before"] = {"uint": struct.unpack("<I", a)[0],
                                               "float": decode(parse_type("float"), a)[0]}
                            entry["after"] = {"uint": struct.unpack("<I", b)[0],
                                              "float": decode(parse_type("float"), b)[0]}
                        else:
                            entry.update(before_hex=a.hex(), after_hex=b.hex())
                        details.append(entry)
            return {"resource_id": resource_id, "baseline_event_id": baseline,
                    "buffer_bytes": buf.length, "offset": offset, "compared_bytes": size,
                    "range_truncated": truncated, "element_stride": element_stride,
                    "layout": _public_layout(layout_fields), "equal": changed == 0,
                    "total_elements": (size + element_stride - 1) // element_stride,
                    "changed_elements": changed,
                    "changed_ranges": [{"first": a, "last": b} for a, b in ranges],
                    "ranges_truncated": len(ranges) >= 256, "changes": details,
                    "changes_truncated": changed > len(details),
                    "before_sha256": digest(before), "after_sha256": digest(after),
                    "scope": ("before = contents after baseline_event_id (default: event_id - 1, i.e. "
                              "just before event_id); after = contents after event_id.")}
        return event_query(self.ctx, self._invoke, self.state, event_id, read, require_action=False)
