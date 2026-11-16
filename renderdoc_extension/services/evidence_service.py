"""Event-bound numeric texture evidence, output fingerprints and GPU resource bundles.

Python 3.6 / stdlib + RenderDoc only. Every query replays through an explicit event first and
records the capture session and MCP edit generation, so before/after comparisons are explicit.
Texture/buffer contents are the replayed state after the event has executed.
"""

import math
import os
import re

import renderdoc as rd

from ..utils import Helpers, Parsers, Serializers
from ..utils.native import native_value, result_status
from .replay_common import digest, enum_name, event_query, json_bytes, write_file

# Bytes per texel for packed formats common in render targets (HDR R11G11B10, R10G10B10A2).
PACKED_BYTES = {"R10G10B10A2": 4, "R11G11B10": 4, "R9G9B9E5": 4, "R5G6B5": 2, "R5G5B5A1": 2,
                "R4G4B4A4": 2, "R4G4": 1}
CASTS = ("Auto", "Typeless", "Float", "UNorm", "UNormSRGB", "SNorm", "UInt", "SInt", "Depth")


class EvidenceService(object):
    def __init__(self, ctx, invoke_fn, state):
        self.ctx = ctx
        self._invoke = invoke_fn
        self.state = state

    def _query(self, event_id, read):
        return event_query(self.ctx, self._invoke, self.state, event_id, read)

    # ------------------------------------------------------------------ textures

    @staticmethod
    def _texture_readback(tex, mip, raw):
        """Validate byte counts where the packed layout is known (one mip/layer/sample)."""
        if not raw:
            raise ValueError("Empty texture readback")
        width, height, depth = (int(getattr(tex, key, 0)) for key in ("width", "height", "depth"))
        fmt = getattr(tex, "format", None)
        kind = enum_name(getattr(fmt, "type", "Undefined"))
        expected, layout = None, "unverified_format_layout"
        if min(width, height, depth) > 0:
            width, height, depth = (max(1, value >> mip) for value in (width, height, depth))
            components, byte_width = int(getattr(fmt, "compCount", 0)), int(getattr(fmt, "compByteWidth", 0))
            if kind == "Regular" and 1 <= components <= 4 and byte_width in (1, 2, 4, 8):
                expected = width * height * depth * components * byte_width
                layout = "tightly_packed_regular"
            elif kind in PACKED_BYTES:
                expected, layout = width * height * depth * PACKED_BYTES[kind], "tightly_packed_" + kind
            elif kind in ("BC1", "BC2", "BC3", "BC4", "BC5", "BC6", "BC7"):
                block = 8 if kind in ("BC1", "BC4") else 16
                expected, layout = ((width + 3) // 4) * ((height + 3) // 4) * depth * block, "bc_4x4_blocks"
        if expected is not None and len(raw) != expected:
            raise ValueError("Texture readback size mismatch: expected %d bytes, got %d" % (expected, len(raw)))
        return {"actual_bytes": len(raw), "expected_bytes": expected, "layout": layout,
                "size_verified": expected is not None,
                "scope": "One mip, one array layer and sample; whole mip depth for 3D textures"}

    @staticmethod
    def _texture(controller, resource_id, mip, slice, sample):
        rid = Parsers.parse_resource_id(resource_id, controller)
        tex = next((t for t in controller.GetTextures() if t.resourceId == rid), None)
        if tex is None:
            raise ValueError("Resource %s is not a texture" % resource_id)
        if (not 0 <= mip < tex.mips or not 0 <= slice < max(tex.arraysize, max(1, tex.depth >> mip))
                or not 0 <= sample < max(1, tex.msSamp)):
            raise ValueError("Invalid texture subresource")
        sub = rd.Subresource()
        sub.mip, sub.slice, sub.sample = mip, slice, sample
        return tex, sub

    @staticmethod
    def _pixels(value, kind, channels):
        field = "uintValue" if kind == "UInt" else "intValue" if kind == "SInt" else "floatValue"
        return [v if not isinstance(v, float) or math.isfinite(v) else {"non_finite": str(v)}
                for v in list(getattr(value, field))[:channels]]

    @staticmethod
    def _view_cast(controller, tex):
        """Interpretation of a typeless texture from a view bound at the current event."""
        pipe = controller.GetPipelineState()
        views = [("output", d) for d in pipe.GetOutputTargets()] + [("depth", pipe.GetDepthTarget())]
        for stage in Helpers.get_all_shader_stages():
            for getter in ("GetReadOnlyResources", "GetReadWriteResources"):
                views.extend((getter, used.descriptor) for used in getattr(pipe, getter)(stage, False))
        for role, view in views:
            if view.resource == tex.resourceId:
                kind = enum_name(view.format.compType)
                if kind != "Typeless" and getattr(rd.CompType, kind, None) is not None:
                    return kind, "%s view format %s" % (role, view.format.Name())
        return None, None

    def _evidence(self, controller, resource_id, mip, slice, sample, grid, type_cast,
                  hash_byte_limit, histogram_range=None, points=None):
        if type_cast not in CASTS:
            raise ValueError("type_cast must be one of " + ", ".join(CASTS))
        tex, sub = self._texture(controller, resource_id, mip, slice, sample)
        if not 1 <= grid <= 32 or not 0 <= hash_byte_limit <= 256 * 1024 * 1024:
            raise ValueError("Invalid evidence bounds")
        source = "explicit"
        if type_cast == "Auto":
            type_cast, source = "Typeless", "texture format"
            if enum_name(tex.format.compType) == "Typeless":
                type_cast, source = self._view_cast(controller, tex)
                if type_cast is None:
                    raise ValueError("Typeless texture is not bound with a typed view at this event; "
                                     "pass an explicit type_cast")
        cast = getattr(rd.CompType, type_cast)
        kind = enum_name(tex.format.compType if type_cast == "Typeless" else cast)
        if kind == "Typeless":
            raise ValueError("Typeless texture format requires an explicit type_cast")
        count = max(1, min(4, tex.format.compCount))
        width, height = max(1, tex.width >> mip), max(1, tex.height >> mip)
        result = {"resource_id": str(tex.resourceId), "mip": mip, "slice": slice, "sample": sample,
                  "width": width, "height": height, "format": tex.format.Name(), "type_cast": type_cast,
                  "type_cast_source": source, "numeric_type": kind, "components": count, "samples": [],
                  "coordinate_convention": "texel coordinates, top-left origin",
                  "value_space": "RenderDoc decoded values; no display range mapping"}
        locations = points if points is not None else [
            [x, y] for y in sorted({int(round(i * (height - 1) / float(max(1, grid - 1)))) for i in range(grid)})
            for x in sorted({int(round(i * (width - 1) / float(max(1, grid - 1)))) for i in range(grid)})]
        if len(locations) > 1024:
            raise ValueError("At most 1024 pixel samples")
        for xy in locations:
            if (len(xy) != 2 or any(type(v) is not int for v in xy)
                    or not 0 <= xy[0] < width or not 0 <= xy[1] < height):
                raise ValueError("Invalid pixel coordinates %s for %dx%d" % (xy, width, height))
            result["samples"].append({"x": xy[0], "y": xy[1], "value": self._pixels(
                controller.PickPixel(tex.resourceId, xy[0], xy[1], sub, cast), kind, count)})
        try:
            low, high = controller.GetMinMax(tex.resourceId, sub, cast)
            result["min"], result["max"] = self._pixels(low, kind, count), self._pixels(high, kind, count)
        except Exception as exc:
            result.setdefault("unavailable", {})["minmax"] = str(exc)
        if histogram_range is not None:
            low, high = histogram_range
            if not math.isfinite(low) or not math.isfinite(high) or low >= high:
                raise ValueError("Histogram range must be finite and increasing")
            try:
                buckets = controller.GetHistogram(tex.resourceId, sub, cast, low, high,
                                                  [i < count for i in range(4)])
                result["histogram"] = {"range": [low, high], "channels": list(range(count)),
                                       "buckets": list(buckets),
                                       "scope": "Pixel counts per bucket over the selected channels"}
            except Exception as exc:
                result.setdefault("unavailable", {})["histogram"] = str(exc)
        if hash_byte_limit and tex.byteSize <= hash_byte_limit:
            raw_sub = rd.Subresource()
            raw_sub.mip, raw_sub.slice, raw_sub.sample = mip, (0 if tex.depth > 1 else slice), sample
            try:
                raw = bytes(controller.GetTextureData(tex.resourceId, raw_sub))
                validation = self._texture_readback(tex, mip, raw)
                result["raw_readback"] = validation
                if not validation["size_verified"]:
                    result.setdefault("unavailable", {})["raw"] = "Byte size unverified for this format layout"
                result.update(raw_sha256=digest(raw), raw_bytes=len(raw))
            except Exception as exc:
                result.setdefault("unavailable", {})["raw"] = str(exc)
        else:
            result["hash_skipped"] = "Disabled, or texture allocation exceeds hash_byte_limit"
        result["sample_scope"] = ("Explicit pixels" if points is not None
                                  else "Fixed grid; unsampled pixels are not covered")
        return result_status(result)

    def texture_evidence(self, event_id, resource_id, mip=0, slice=0, sample=0, grid=8,
                         type_cast="Auto", hash_byte_limit=67108864,
                         histogram_range=None, points=None):
        return self._query(event_id, lambda c: self._evidence(
            c, resource_id, mip, slice, sample, grid, type_cast, hash_byte_limit, histogram_range, points))

    def fingerprint_outputs(self, event_id, grid=8, hash_byte_limit=67108864):
        def read(controller):
            pipe = controller.GetPipelineState()
            targets = [("color:%d" % i, t) for i, t in enumerate(pipe.GetOutputTargets())]
            targets.append(("depth", pipe.GetDepthTarget()))
            outputs = []
            for role, target in targets:
                if target.resource == rd.ResourceId.Null():
                    continue
                value = self._evidence(controller, str(target.resource), target.firstMip,
                                       target.firstSlice, 0, grid, "Auto", hash_byte_limit)
                outputs.append({"role": role, "format": native_value(target.format), "evidence": value})
            return result_status({"outputs": outputs, "scope": (
                "Bound color/depth attachments, first view mip/layer and sample 0; excludes UAVs. "
                "Equal hashes establish only these bytes in this replay.")}, outputs)
        return self._query(event_id, read)

    # ------------------------------------------------------------------ bundle

    def _snapshot(self, controller, event_id):
        pipe, action = controller.GetPipelineState(), self.ctx.GetAction(event_id)
        result = {"event_id": event_id, "action": {
                      "name": action.GetName(controller.GetStructuredFile()) if action else None,
                      "flags": str(action.flags) if action else None,
                      "num_indices": getattr(action, "numIndices", None),
                      "num_instances": getattr(action, "numInstances", None),
                      "dispatch_dimension": native_value(getattr(action, "dispatchDimension", None))},
                  "shaders": {}, "bindings": []}
        resources = {}

        def binding(role, rid, details):
            if rid != rd.ResourceId.Null():
                resources[str(rid)] = rid
                result["bindings"].append({"role": role, "resource_id": str(rid), "details": native_value(details)})

        for index, target in enumerate(pipe.GetOutputTargets()):
            binding("color_output:%d" % index, target.resource, target)
        depth = pipe.GetDepthTarget()
        binding("depth_output", depth.resource, depth)
        for index, buf in enumerate(pipe.GetVBuffers()):
            binding("vertex_buffer:%d" % index, buf.resourceId, buf)
        ib = pipe.GetIBuffer()
        binding("index_buffer", ib.resourceId, ib)
        result["vertex_inputs"] = native_value(pipe.GetVertexInputs())
        result["topology"] = str(pipe.GetPrimitiveTopology())
        for stage in Helpers.get_all_shader_stages():
            rid = pipe.GetShader(stage)
            if rid == rd.ResourceId.Null():
                continue
            key = enum_name(stage)
            reflection = pipe.GetShaderReflection(stage)
            if reflection is None:
                result.setdefault("unavailable", {})[key] = "Shader reflection unavailable"
                continue
            result["shaders"][key] = {"resource_id": str(rid), "entry_point": reflection.entryPoint,
                                      "encoding": enum_name(reflection.encoding),
                                      "mcp_replacement": _replacement(self.state, rid),
                                      "samplers": native_value(pipe.GetSamplers(stage, False))}
            for getter, role in (("GetReadOnlyResources", "read"), ("GetReadWriteResources", "read_write"),
                                 ("GetConstantBlocks", "constant")):
                descriptors = getattr(pipe, getter)(stage, False)
                if len(descriptors) > 4096:
                    result.setdefault("unavailable", {})[key + ":" + role] = "Descriptor limit exceeded"
                for used in descriptors[:4096]:
                    binding(key + ":" + role, used.descriptor.resource, used)
        api = enum_name(controller.GetAPIProperties().pipelineType)
        getters = {"D3D11": "GetD3D11PipelineState", "D3D12": "GetD3D12PipelineState",
                   "Vulkan": "GetVulkanPipelineState", "OpenGL": "GetGLPipelineState"}
        if api in getters:
            result["native"] = native_value(getattr(controller, getters[api])())
        else:
            result.setdefault("unavailable", {})["native"] = "Unsupported pipeline API"
        result["resource_ids"] = sorted(resources)
        result["binding_scope"] = "Bound views at this event, not proof of execution or per-pixel use"
        return result, resources

    @staticmethod
    def _reflection(reflection):
        keys = ("stage", "entryPoint", "encoding", "inputSignature", "outputSignature",
                "constantBlocks", "readOnlyResources", "readWriteResources", "samplers",
                "interfaces", "dispatchThreadsDimension")
        value = {key: native_value(getattr(reflection, key)) for key in keys if hasattr(reflection, key)}
        debug = reflection.debugInfo
        value["debugInfo"] = {"compileFlags": native_value(debug.compileFlags),
                              "files": [str(f.filename) for f in debug.files]}
        return value

    def export_resource_bundle(self, event_id, output_dir, byte_limit=268435456, resources="all"):
        if not 1024 <= byte_limit <= 2147483648:
            raise ValueError("byte_limit must be 1 KiB..2 GiB")
        if not os.path.isabs(output_dir) or os.path.exists(output_dir):
            raise ValueError("output_dir must be a new absolute directory")
        if not (resources in ("all", "none") or isinstance(resources, list)):
            raise ValueError("resources must be 'all', 'none' or a list of resource IDs")

        def read(controller):
            os.makedirs(output_dir)
            files, omissions, total = [], [], [0]

            def store(name, data, kind, metadata):
                if total[0] + len(data) > byte_limit:
                    omissions.append({"kind": kind, "file": name, "metadata": metadata,
                                      "reason": "Bundle byte limit"})
                    return False
                info = write_file(os.path.join(output_dir, name), data)
                total[0] += len(data)
                files.append({"file": name, "bytes": info["bytes"], "sha256": info["sha256"],
                              "kind": kind, "metadata": metadata})
                return True

            snapshot, bound = self._snapshot(controller, event_id)
            store("snapshot.json", json_bytes(snapshot), "event_snapshot", {})
            pipe = controller.GetPipelineState()
            targets = list(controller.GetDisassemblyTargets(True))
            for stage in Helpers.get_all_shader_stages():
                if pipe.GetShader(stage) == rd.ResourceId.Null():
                    continue
                label = enum_name(stage)
                reflection = pipe.GetShaderReflection(stage)
                if reflection is None:
                    omissions.append({"kind": "shader", "stage": label, "reason": "Reflection unavailable"})
                    continue
                meta = {"stage": label, "resource_id": str(pipe.GetShader(stage)),
                        "encoding": enum_name(reflection.encoding), "entry_point": reflection.entryPoint}
                raw = bytes(reflection.rawBytes)
                if raw:
                    store("shader_%s.bin" % label, raw, "shader_binary", meta)
                else:
                    omissions.append({"kind": "shader_binary", "stage": label, "reason": "Empty bytecode"})
                store("shader_%s.reflection.json" % label, json_bytes(self._reflection(reflection)),
                      "shader_reflection", meta)
                pipeline = (pipe.GetComputePipelineObject() if stage == rd.ShaderStage.Compute
                            else pipe.GetGraphicsPipelineObject())
                if targets:
                    try:
                        text = controller.DisassembleShader(pipeline, reflection, targets[0])
                        store("shader_%s.disasm.txt" % label, text.encode("utf-8"), "shader_disassembly",
                              dict(meta, target=targets[0]))
                    except Exception as exc:
                        omissions.append({"kind": "shader_disassembly", "stage": label, "reason": str(exc)})
                for index, source in enumerate(reflection.debugInfo.files):
                    if source.contents:
                        base = re.sub(r"[^A-Za-z0-9_.-]+", "_", os.path.basename(str(source.filename)))[-80:]
                        store("shader_%s.source_%d_%s" % (label, index, base or "unnamed"),
                              str(source.contents).encode("utf-8"), "shader_embedded_source",
                              dict(meta, filename=str(source.filename)))
                self._constants(controller, pipe, stage, label, reflection, pipeline, store, omissions)
            if resources != "none":
                self._resources(controller, bound, resources, store, omissions, total, byte_limit)
            manifest = {"schema": "renderdoc-mcp-resource-bundle/1", "event_id": event_id,
                        "capture": self.state.identity(controller), "files": files,
                        "omissions": omissions, "data_bytes": total[0], "byte_limit": byte_limit,
                        "resources": resources,
                        "complete": not snapshot.get("unavailable") and not omissions,
                        "scope": ("Shader binaries/disassembly/reflection, decoded and raw constants, "
                                  "and the selected bound buffers/textures as replayed through this "
                                  "event (after it executes). Raw textures are tightly packed per "
                                  "mip/layer/sample in the listed format. No engine assets or earlier "
                                  "resource versions.")}
            write_file(os.path.join(output_dir, "manifest.json"), json_bytes(manifest))
            return manifest

        return self._query(event_id, read)

    def _constants(self, controller, pipe, stage, label, reflection, pipeline, store, omissions):
        buffers = None
        for index, block in enumerate(reflection.constantBlocks):
            meta = {"stage": label, "block_index": index, "name": block.name,
                    "declared_size": block.byteSize, "buffer_backed": block.bufferBacked,
                    "fixed_bind_number": getattr(block, "fixedBindNumber", None)}
            try:
                bound = pipe.GetConstantBlock(stage, index, 0).descriptor
                meta.update(resource_id=str(bound.resource), byte_offset=bound.byteOffset,
                            byte_size=bound.byteSize)
                if block.bufferBacked and bound.resource == rd.ResourceId.Null():
                    omissions.append(dict(meta, kind="constants", reason="Block is unbound at this event"))
                    continue
                variables = controller.GetCBufferVariableContents(
                    pipeline, reflection.resourceId, stage, reflection.entryPoint, index,
                    bound.resource, bound.byteOffset, bound.byteSize)
                store("constants_%s_%d.json" % (label, index),
                      json_bytes({"metadata": meta, "variables": Serializers.serialize_variables(variables)}),
                      "constants_decoded", meta)
                if block.bufferBacked:
                    if buffers is None:
                        buffers = {str(b.resourceId): b for b in controller.GetBuffers()}
                    desc = buffers.get(str(bound.resource))
                    if desc is None:
                        omissions.append(dict(meta, kind="constants_raw", reason="Bound resource is not a buffer"))
                        continue
                    size = bound.byteSize or block.byteSize
                    size = max(0, min(size, desc.length - bound.byteOffset))
                    raw = bytes(controller.GetBufferData(bound.resource, bound.byteOffset, size)) if size else b""
                    if raw:
                        store("constants_%s_%d.bin" % (label, index), raw, "constants_raw",
                              dict(meta, raw_bytes=len(raw)))
                    else:
                        omissions.append(dict(meta, kind="constants_raw", reason="Empty constant window"))
            except Exception as exc:
                omissions.append(dict(meta, kind="constants", reason=str(exc)))

    def _resources(self, controller, bound, selection, store, omissions, total, byte_limit):
        textures = {str(t.resourceId): t for t in controller.GetTextures()}
        buffers = {str(b.resourceId): b for b in controller.GetBuffers()}
        names = {str(r.resourceId): str(r.name) for r in controller.GetResources()}
        if selection == "all":
            keys = list(bound)
        else:
            wanted = {Parsers.extract_numeric_id(str(item)) for item in selection}
            keys = [key for key in bound if Parsers.extract_numeric_id(key) in wanted]
            found = {Parsers.extract_numeric_id(key) for key in keys}
            for missing in sorted(wanted - found):
                omissions.append({"kind": "resource", "resource_id": str(missing),
                                  "reason": "Not bound at this event"})
        for key in keys:
            rid, number = bound[key], Parsers.extract_numeric_id(key)
            if key in buffers:
                size = buffers[key].length
                if size <= 0 or total[0] + size > byte_limit:
                    omissions.append({"kind": "buffer", "resource_id": key, "bytes": size,
                                      "reason": "Invalid length or exceeds remaining byte budget"})
                    continue
                try:
                    raw = bytes(controller.GetBufferData(rid, 0, size))
                    if len(raw) != size:
                        omissions.append({"kind": "buffer", "resource_id": key,
                                          "reason": "Readback size %d != %d" % (len(raw), size)})
                        continue
                    store("buffer_%d.bin" % number, raw, "buffer", {"resource_id": key, "bytes": size})
                except Exception as exc:
                    omissions.append({"kind": "buffer", "resource_id": key, "reason": str(exc)})
            elif key in textures:
                tex = textures[key]
                layers = 1 if tex.depth > 1 else max(1, tex.arraysize)
                count = tex.mips * layers * max(1, tex.msSamp)
                if total[0] + tex.byteSize > byte_limit or count > 4096:
                    omissions.append({"kind": "texture", "resource_id": key, "bytes": tex.byteSize,
                                      "reason": "Exceeds remaining byte or subresource budget"})
                    continue
                description = {"resource_id": key, "name": names.get(key),
                               "format": tex.format.Name(), "width": tex.width, "height": tex.height,
                               "depth": tex.depth, "array_size": tex.arraysize, "mips": tex.mips,
                               "samples": tex.msSamp, "dimension": str(tex.type)}
                try:
                    for mip in range(tex.mips):
                        for layer in range(layers):
                            for sample in range(max(1, tex.msSamp)):
                                sub = rd.Subresource()
                                sub.mip, sub.slice, sub.sample = mip, layer, sample
                                raw = bytes(controller.GetTextureData(rid, sub))
                                try:
                                    check = self._texture_readback(tex, mip, raw)
                                except ValueError as exc:
                                    omissions.append({"kind": "texture", "resource_id": key, "mip": mip,
                                                      "slice": layer, "sample": sample, "reason": str(exc)})
                                    continue
                                store("texture_%d_m%d_s%d_x%d.bin" % (number, mip, layer, sample), raw,
                                      "texture", dict(description, mip=mip, slice=layer, sample=sample,
                                                      mip_width=max(1, tex.width >> mip),
                                                      mip_height=max(1, tex.height >> mip),
                                                      readback=check))
                except Exception as exc:
                    omissions.append({"kind": "texture", "resource_id": key, "reason": str(exc)})
            else:
                omissions.append({"kind": "resource", "resource_id": key,
                                  "reason": "Neither a captured buffer nor texture"})


def _replacement(state, rid):
    record = state.replacements.get(str(Parsers.extract_numeric_id(str(rid))))
    return record["replacement_shader"] if record else None

