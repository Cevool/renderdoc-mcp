"""
Pipeline state service for RenderDoc.
"""

import renderdoc as rd

from ..utils import Parsers, Serializers, Helpers
from ..utils.native import native_value, result_status


class PipelineService:
    """Pipeline state service"""

    def __init__(self, ctx, invoke_fn):
        self.ctx = ctx
        self._invoke = invoke_fn

    def get_shader_info(self, event_id, stage):
        """Get shader information for a specific stage"""
        if not self.ctx.IsCaptureLoaded():
            raise ValueError("No capture loaded")

        result = {"shader": None, "error": None}

        def callback(controller):
            controller.SetFrameEvent(event_id, True)

            pipe = controller.GetPipelineState()
            stage_enum = Parsers.parse_stage(stage)

            shader = pipe.GetShader(stage_enum)
            if shader == rd.ResourceId.Null():
                result["error"] = "No %s shader bound" % stage
                return

            entry = pipe.GetShaderEntryPoint(stage_enum)
            reflection = pipe.GetShaderReflection(stage_enum)

            shader_info = {
                "resource_id": str(shader),
                "entry_point": entry,
                "stage": stage,
            }

            # Get disassembly
            try:
                targets = controller.GetDisassemblyTargets(True)
                if targets:
                    pipeline_object = (pipe.GetComputePipelineObject() if stage == "compute"
                                       else pipe.GetGraphicsPipelineObject())
                    disasm = controller.DisassembleShader(pipeline_object, reflection, targets[0])
                    shader_info["disassembly"] = disasm
            except Exception as e:
                shader_info["disassembly_error"] = str(e)

            # Get constant buffer info
            if reflection:
                shader_info["constant_buffers"] = self._get_cbuffer_info(
                    controller, pipe, reflection, stage_enum
                )
                shader_info["resources"] = self._get_resource_bindings(reflection)

            result["shader"] = shader_info

        self._invoke(callback)

        if result["error"]:
            raise ValueError(result["error"])
        return result["shader"]

    def get_pipeline_state(self, event_id):
        """Get full pipeline state at an event"""
        if not self.ctx.IsCaptureLoaded():
            raise ValueError("No capture loaded")

        result = {"pipeline": None, "error": None}

        def callback(controller):
            controller.SetFrameEvent(event_id, True)

            pipe = controller.GetPipelineState()
            api = controller.GetAPIProperties().pipelineType

            pipeline_info = {
                "event_id": event_id,
                "api": str(api),
            }

            # Shader stages with detailed bindings
            stages = {}
            stage_list = Helpers.get_all_shader_stages()
            for stage in stage_list:
                shader = pipe.GetShader(stage)
                if shader != rd.ResourceId.Null():
                    stage_info = {
                        "resource_id": str(shader),
                        "entry_point": pipe.GetShaderEntryPoint(stage),
                    }

                    reflection = pipe.GetShaderReflection(stage)

                    stage_info["resources"] = self._get_stage_resources(
                        controller, pipe, stage, reflection
                    )
                    stage_info["uavs"] = self._get_stage_uavs(
                        controller, pipe, stage, reflection
                    )
                    stage_info["samplers"] = self._get_stage_samplers(
                        pipe, stage, reflection
                    )
                    stage_info["constant_buffers"] = self._get_stage_cbuffers(
                        controller, pipe, stage, reflection
                    )

                    stages[str(stage)] = stage_info

            pipeline_info["shaders"] = stages

            # Use the common pipeline API. The former GetOutputMerger/GetIAState/
            # GetViewportScissor calls do not exist on current RenderDoc's PipeState.
            sections = {
                "render_targets": lambda: [
                    {"index": i, "resource_id": str(target.resource)}
                    for i, target in enumerate(pipe.GetOutputTargets())
                    if target.resource != rd.ResourceId.Null()
                ],
                "depth_target": lambda: str(pipe.GetDepthTarget().resource),
                "input_assembly": lambda: {
                    "topology": str(pipe.GetPrimitiveTopology()),
                    "vertex_buffers": [self._bound_buffer(buffer, i)
                                       for i, buffer in enumerate(pipe.GetVBuffers())
                                       if buffer.resourceId != rd.ResourceId.Null()],
                    "index_buffer": self._bound_buffer(pipe.GetIBuffer()),
                },
                "viewport_scissor": lambda: self._viewport_scissor(controller, pipe),
            }
            for section, read in sections.items():
                try:
                    pipeline_info[section] = read()
                except Exception as exc:
                    pipeline_info.setdefault("unavailable", {})[section] = str(exc)

            result["pipeline"] = pipeline_info

        self._invoke(callback)

        if result["error"]:
            raise ValueError(result["error"])
        return result["pipeline"]

    def get_constant_buffer(self, event_id, stage, block_index=0, array_index=0):
        """Associate declarations with the actual binding and decoded GPU values."""
        if not self.ctx.IsCaptureLoaded():
            raise ValueError("No capture loaded")
        result = {}

        def callback(controller):
            if self.ctx.GetAction(event_id) is None:
                raise ValueError("No action at event_id %s" % event_id)
            controller.SetFrameEvent(event_id, True)
            pipe = controller.GetPipelineState()
            stage_enum = Parsers.parse_stage(stage)
            reflection = pipe.GetShaderReflection(stage_enum)
            if reflection is None:
                raise ValueError("No shader reflection for " + stage)
            if not 0 <= block_index < len(reflection.constantBlocks):
                raise ValueError("Invalid constant block index")
            block = reflection.constantBlocks[block_index]
            binding = pipe.GetConstantBlock(stage_enum, block_index, array_index).descriptor
            rid = binding.resource
            result.update(event_id=event_id, stage=stage, block_index=block_index,
                          array_index=array_index, name=block.name, declared_size=block.byteSize,
                          buffer_backed=block.bufferBacked, resource_id=str(rid),
                          byte_offset=binding.byteOffset, byte_size=binding.byteSize,
                          declarations=native_value(block.variables))
            if block.bufferBacked and rid == rd.ResourceId.Null():
                result.update(status="empty", complete=True, bound=False, variables=[])
                return
            pipeline = (pipe.GetComputePipelineObject() if stage == "compute" else
                        pipe.GetGraphicsPipelineObject())
            variables = controller.GetCBufferVariableContents(pipeline, reflection.resourceId, stage_enum,
                reflection.entryPoint, block_index, rid, binding.byteOffset, binding.byteSize)
            result.update(bound=True, variables=Serializers.serialize_variables(variables))
            if block.bufferBacked:
                # Reuse the existing byte export instead of introducing a second exporter.
                result["raw_read"] = {"tool": "get_buffer_contents", "arguments": {
                    "event_id": event_id, "resource_id": str(rid), "offset": binding.byteOffset,
                    "length": binding.byteSize or block.byteSize, "output": "file"}}
            result_status(result, result["variables"])

        self._invoke(callback)
        return result

    @staticmethod
    def _bound_buffer(buffer, slot=None):
        value = {"resource_id": str(buffer.resourceId), "offset": buffer.byteOffset,
                 "size": buffer.byteSize, "stride": buffer.byteStride}
        if slot is not None:
            value["slot"] = slot
        return value

    @staticmethod
    def _viewport_scissor(controller, pipe):
        if pipe.IsCaptureVK():
            pairs = controller.GetVulkanPipelineState().viewportScissor.viewportScissors
            viewports = [pair.vp for pair in pairs]
            scissors = [pair.scissor for pair in pairs]
        else:
            if pipe.IsCaptureD3D11():
                native = controller.GetD3D11PipelineState()
            elif pipe.IsCaptureD3D12():
                native = controller.GetD3D12PipelineState()
            else:
                native = controller.GetGLPipelineState()
            viewports = native.rasterizer.viewports
            scissors = native.rasterizer.scissors
        return {
            "viewports": [{"x": v.x, "y": v.y, "width": v.width, "height": v.height,
                           "min_depth": v.minDepth, "max_depth": v.maxDepth} for v in viewports],
            "scissors": [{"x": s.x, "y": s.y, "width": s.width, "height": s.height}
                         for s in scissors],
        }

    def _get_stage_resources(self, controller, pipe, stage, reflection):
        """Get shader resource views (SRVs) for a stage"""
        resources = []
        try:
            srvs = pipe.GetReadOnlyResources(stage, False)

            name_map = {}
            if reflection:
                for res in reflection.readOnlyResources:
                    name_map[res.fixedBindNumber] = res.name

            for srv in srvs:
                if srv.descriptor.resource == rd.ResourceId.Null():
                    continue

                slot = srv.access.index
                res_info = {
                    "slot": slot,
                    "name": name_map.get(slot, ""),
                    "resource_id": str(srv.descriptor.resource),
                }

                res_info.update(
                    self._get_resource_details(controller, srv.descriptor.resource)
                )

                res_info["first_mip"] = srv.descriptor.firstMip
                res_info["num_mips"] = srv.descriptor.numMips
                res_info["first_slice"] = srv.descriptor.firstSlice
                res_info["num_slices"] = srv.descriptor.numSlices

                resources.append(res_info)
        except Exception as e:
            resources.append({"error": str(e)})

        return resources

    def _get_stage_uavs(self, controller, pipe, stage, reflection):
        """Get unordered access views (UAVs) for a stage"""
        uavs = []
        try:
            uav_list = pipe.GetReadWriteResources(stage, False)

            name_map = {}
            if reflection:
                for res in reflection.readWriteResources:
                    name_map[res.fixedBindNumber] = res.name

            for uav in uav_list:
                if uav.descriptor.resource == rd.ResourceId.Null():
                    continue

                slot = uav.access.index
                uav_info = {
                    "slot": slot,
                    "name": name_map.get(slot, ""),
                    "resource_id": str(uav.descriptor.resource),
                }

                uav_info.update(
                    self._get_resource_details(controller, uav.descriptor.resource)
                )

                uav_info["byte_offset"] = uav.descriptor.byteOffset
                uav_info["byte_size"] = uav.descriptor.byteSize
                uav_info["first_mip"] = uav.descriptor.firstMip
                uav_info["num_mips"] = uav.descriptor.numMips

                uavs.append(uav_info)
        except Exception as e:
            uavs.append({"error": str(e)})

        return uavs

    def _get_stage_samplers(self, pipe, stage, reflection):
        """Get samplers for a stage"""
        samplers = []
        try:
            sampler_list = pipe.GetSamplers(stage, False)

            name_map = {}
            if reflection:
                for samp in reflection.samplers:
                    name_map[samp.fixedBindNumber] = samp.name

            for samp in sampler_list:
                slot = samp.access.index
                samp_info = {
                    "slot": slot,
                    "name": name_map.get(slot, ""),
                }

                desc = samp.sampler
                try:
                    samp_info["address_u"] = str(desc.addressU)
                    samp_info["address_v"] = str(desc.addressV)
                    samp_info["address_w"] = str(desc.addressW)
                except AttributeError:
                    pass

                try:
                    samp_info["filter"] = native_value(desc.filter)
                except AttributeError:
                    pass

                try:
                    samp_info["max_anisotropy"] = desc.maxAnisotropy
                except AttributeError:
                    pass

                try:
                    samp_info["min_lod"] = desc.minLOD
                    samp_info["max_lod"] = desc.maxLOD
                    samp_info["mip_lod_bias"] = desc.mipLODBias
                except AttributeError:
                    pass

                try:
                    samp_info["border_color"] = [
                        desc.borderColor[0],
                        desc.borderColor[1],
                        desc.borderColor[2],
                        desc.borderColor[3],
                    ]
                except (AttributeError, TypeError):
                    pass

                try:
                    samp_info["compare_function"] = str(desc.compareFunction)
                except AttributeError:
                    pass

                samplers.append(samp_info)
        except Exception as e:
            samplers.append({"error": str(e)})

        return samplers

    def _get_stage_cbuffers(self, controller, pipe, stage, reflection):
        """Get constant buffers for a stage from shader reflection"""
        cbuffers = []
        try:
            if not reflection:
                return cbuffers

            for index, cb in enumerate(reflection.constantBlocks):
                slot = cb.bindPoint if hasattr(cb, 'bindPoint') else cb.fixedBindNumber
                cb_info = {
                    "slot": slot,
                    "name": cb.name,
                    "byte_size": cb.byteSize,
                    "variable_count": len(cb.variables) if cb.variables else 0,
                    "variables": [],
                }
                if hasattr(pipe, "GetConstantBlock"):
                    bound = pipe.GetConstantBlock(stage, index, 0).descriptor
                    cb_info.update({"resource_id": str(bound.resource),
                                    "offset": bound.byteOffset, "bound_size": bound.byteSize})
                if cb.variables:
                    for var in cb.variables:
                        cb_info["variables"].append({
                            "name": var.name,
                            "byte_offset": var.byteOffset,
                            "type": str(var.type.name) if var.type else "",
                        })
                cbuffers.append(cb_info)

        except Exception as e:
            cbuffers.append({"error": str(e)})

        return cbuffers

    def _get_resource_details(self, controller, resource_id):
        """Get details about a resource (texture or buffer)"""
        details = {}

        try:
            resource_name = self.ctx.GetResourceName(resource_id)
            if resource_name:
                details["resource_name"] = resource_name
        except Exception:
            pass

        for tex in controller.GetTextures():
            if tex.resourceId == resource_id:
                details["type"] = "texture"
                details["width"] = tex.width
                details["height"] = tex.height
                details["depth"] = tex.depth
                details["array_size"] = tex.arraysize
                details["mip_levels"] = tex.mips
                details["format"] = str(tex.format.Name())
                details["dimension"] = str(tex.type)
                details["msaa_samples"] = tex.msSamp
                return details

        for buf in controller.GetBuffers():
            if buf.resourceId == resource_id:
                details["type"] = "buffer"
                details["length"] = buf.length
                return details

        return details

    def _get_cbuffer_info(self, controller, pipe, reflection, stage):
        """Get constant buffer information and values"""
        cbuffers = []

        for i, cb in enumerate(reflection.constantBlocks):
            cb_info = {
                "name": cb.name,
                "slot": i,
                "size": cb.byteSize,
                "variables": [],
            }

            try:
                if hasattr(pipe, "GetConstantBlock"):
                    bind = pipe.GetConstantBlock(stage, i, 0).descriptor
                    resource_id = bind.resource
                else:
                    bind = pipe.GetConstantBuffer(stage, i, 0)
                    resource_id = bind.resourceId
                if resource_id != rd.ResourceId.Null():
                    variables = controller.GetCBufferVariableContents(
                        (pipe.GetComputePipelineObject() if stage == rd.ShaderStage.Compute
                         else pipe.GetGraphicsPipelineObject()),
                        reflection.resourceId,
                        stage,
                        reflection.entryPoint,
                        i,
                        resource_id,
                        bind.byteOffset,
                        bind.byteSize,
                    )
                    cb_info["variables"] = Serializers.serialize_variables(variables)
            except Exception as e:
                cb_info["error"] = str(e)

            cbuffers.append(cb_info)

        return cbuffers

    def _get_resource_bindings(self, reflection):
        """Get shader resource bindings"""
        resources = []

        try:
            for res in reflection.readOnlyResources:
                resources.append(
                    {
                        "name": res.name,
                        "type": str(res.resType),
                        "binding": res.fixedBindNumber,
                        "access": "ReadOnly",
                    }
                )
        except Exception:
            pass

        try:
            for res in reflection.readWriteResources:
                resources.append(
                    {
                        "name": res.name,
                        "type": str(res.resType),
                        "binding": res.fixedBindNumber,
                        "access": "ReadWrite",
                    }
                )
        except Exception:
            pass

        return resources
