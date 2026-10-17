"""
Resource information service for RenderDoc.
"""

import base64
import os

import renderdoc as rd

from ..utils import Parsers


class ResourceService:
    """Resource information service"""

    def __init__(self, ctx, invoke_fn):
        self.ctx = ctx
        self._invoke = invoke_fn

    def _find_texture_by_id(self, controller, resource_id):
        """Find texture by resource ID"""
        target_id = Parsers.extract_numeric_id(resource_id)
        for tex in controller.GetTextures():
            tex_id_str = str(tex.resourceId)
            tex_id = Parsers.extract_numeric_id(tex_id_str)
            if tex_id == target_id:
                return tex
        return None

    def get_buffer_contents(self, resource_id, offset=0, length=0, event_id=None, output_path=None):
        """Get buffer data"""
        if not self.ctx.IsCaptureLoaded():
            raise ValueError("No capture loaded")

        result = {"data": None, "error": None}

        def callback(controller):
            if event_id is not None:
                controller.SetFrameEvent(event_id, True)
            # Parse resource ID
            try:
                rid = Parsers.parse_resource_id(resource_id, controller)
            except Exception:
                result["error"] = "Invalid resource ID: %s" % resource_id
                return

            # Find buffer
            buf_desc = None
            for buf in controller.GetBuffers():
                if buf.resourceId == rid:
                    buf_desc = buf
                    break

            if not buf_desc:
                result["error"] = "Buffer not found: %s" % resource_id
                return

            # Get data
            if offset < 0 or offset > buf_desc.length or length < 0:
                result["error"] = "Invalid buffer offset/length"
                return
            actual_length = min(length, buf_desc.length - offset) if length > 0 else buf_desc.length - offset
            data = controller.GetBufferData(rid, offset, actual_length)

            result["data"] = {
                "resource_id": resource_id,
                "length": len(data),
                "total_size": buf_desc.length,
                "offset": offset,
            }
            self._store_bytes(result["data"], data, output_path)

        self._invoke(callback)

        if result["error"]:
            raise ValueError(result["error"])
        return result["data"]

    def get_texture_info(self, resource_id):
        """Get texture metadata"""
        if not self.ctx.IsCaptureLoaded():
            raise ValueError("No capture loaded")

        result = {"texture": None, "error": None}

        def callback(controller):
            try:
                tex_desc = self._find_texture_by_id(controller, resource_id)

                if not tex_desc:
                    result["error"] = "Texture not found: %s" % resource_id
                    return

                result["texture"] = {
                    "resource_id": resource_id,
                    "width": tex_desc.width,
                    "height": tex_desc.height,
                    "depth": tex_desc.depth,
                    "array_size": tex_desc.arraysize,
                    "mip_levels": tex_desc.mips,
                    "format": str(tex_desc.format.Name()),
                    "dimension": str(tex_desc.type),
                    "msaa_samples": tex_desc.msSamp,
                    "byte_size": tex_desc.byteSize,
                }
            except Exception as e:
                import traceback
                result["error"] = "Error: %s\n%s" % (str(e), traceback.format_exc())

        self._invoke(callback)

        if result["error"]:
            raise ValueError(result["error"])
        return result["texture"]

    def get_texture_data(self, resource_id, mip=0, slice=0, sample=0, depth_slice=None,
                         event_id=None, output_path=None):
        """Get texture pixel data."""
        if not self.ctx.IsCaptureLoaded():
            raise ValueError("No capture loaded")

        result = {"data": None, "error": None}

        def callback(controller):
            if event_id is not None:
                controller.SetFrameEvent(event_id, True)
            tex_desc = self._find_texture_by_id(controller, resource_id)

            if not tex_desc:
                result["error"] = "Texture not found: %s" % resource_id
                return

            # Validate mip level
            if mip < 0 or mip >= tex_desc.mips:
                result["error"] = "Invalid mip level %d (texture has %d mips)" % (
                    mip,
                    tex_desc.mips,
                )
                return

            # Validate slice for array/cube textures
            # RenderDoc arraysize already counts cube faces.
            max_slices = max(1, tex_desc.arraysize)
            if slice < 0 or slice >= max_slices:
                result["error"] = "Invalid slice %d (texture has %d slices)" % (
                    slice,
                    max_slices,
                )
                return

            # Validate sample for MSAA
            if sample < 0 or sample >= max(1, tex_desc.msSamp):
                result["error"] = "Invalid sample %d (texture has %d samples)" % (
                    sample,
                    tex_desc.msSamp,
                )
                return

            # Calculate dimensions at this mip level
            mip_width = max(1, tex_desc.width >> mip)
            mip_height = max(1, tex_desc.height >> mip)
            mip_depth = max(1, tex_desc.depth >> mip)

            # Validate depth_slice for 3D textures
            is_3d = tex_desc.depth > 1
            if depth_slice is not None:
                if not is_3d:
                    result["error"] = "depth_slice can only be used with 3D textures"
                    return
                if depth_slice < 0 or depth_slice >= mip_depth:
                    result["error"] = "Invalid depth_slice %d (texture has %d depth at mip %d)" % (
                        depth_slice,
                        mip_depth,
                        mip,
                    )
                    return

            # Create subresource specification
            sub = rd.Subresource()
            sub.mip = mip
            sub.slice = slice
            sub.sample = sample

            # Get texture data
            try:
                data = controller.GetTextureData(tex_desc.resourceId, sub)
            except Exception as e:
                result["error"] = "Failed to get texture data: %s" % str(e)
                return

            # Extract depth slice for 3D textures if requested
            output_depth = mip_depth
            if is_3d and depth_slice is not None:
                total_size = len(data)
                bytes_per_slice = total_size // mip_depth
                slice_start = depth_slice * bytes_per_slice
                slice_end = slice_start + bytes_per_slice
                data = data[slice_start:slice_end]
                output_depth = 1

            result["data"] = {
                "resource_id": resource_id,
                "width": mip_width,
                "height": mip_height,
                "depth": output_depth,
                "mip": mip,
                "slice": slice,
                "sample": sample,
                "depth_slice": depth_slice,
                "format": str(tex_desc.format.Name()),
                "dimension": str(tex_desc.type),
                "is_3d": is_3d,
                "total_depth": mip_depth if is_3d else 1,
                "data_length": len(data),
            }
            self._store_bytes(result["data"], data, output_path)

        self._invoke(callback)

        if result["error"]:
            raise ValueError(result["error"])
        return result["data"]

    @staticmethod
    def _store_bytes(metadata, data, output_path):
        if output_path:
            temporary = output_path + ".tmp"
            try:
                with open(temporary, "wb") as stream:
                    stream.write(data)
                os.replace(temporary, output_path)
            finally:
                if os.path.exists(temporary):
                    os.remove(temporary)
            metadata["path"] = output_path
        else:
            if len(data) > 4096:
                raise ValueError("Inline data exceeds 4096 bytes; export to a file")
            metadata["content_base64"] = base64.b64encode(data).decode("ascii")

    def export_texture(self, resource_id, event_id, output_path, mip=0, slice=0):
        if not self.ctx.IsCaptureLoaded():
            raise ValueError("No capture loaded")
        result = {}

        def callback(controller):
            controller.SetFrameEvent(event_id, True)
            texture = self._find_texture_by_id(controller, resource_id)
            if texture is None:
                raise ValueError("Texture not found: %s" % resource_id)
            max_slices = max(1, texture.arraysize, texture.depth >> mip) if mip >= 0 else 0
            if mip < 0 or mip >= texture.mips or slice < 0 or slice >= max_slices:
                raise ValueError("Invalid texture mip/slice")
            save = rd.TextureSave()
            save.resourceId = texture.resourceId
            save.mip = mip
            save.slice.sliceIndex = slice
            save.destType = rd.FileType.PNG
            saved = controller.SaveTexture(save, output_path)
            if hasattr(saved, "OK") and not saved.OK():
                raise ValueError("Texture export failed: %s" % saved)
            if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
                raise ValueError("Texture export did not produce a PNG")
            result.update({"resource_id": resource_id, "event_id": event_id,
                           "mip": mip, "slice": slice,
                           "source_format": str(texture.format.Name())})

        self._invoke(callback)
        return result
