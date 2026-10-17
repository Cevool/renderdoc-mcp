"""
Parse utility functions for RenderDoc data types.
"""

import renderdoc as rd


class Parsers:
    """Parse utility functions (static methods)"""

    @staticmethod
    def parse_stage(stage_str):
        """Convert stage string to ShaderStage enum"""
        stage_map = {
            "vertex": rd.ShaderStage.Vertex,
            "hull": rd.ShaderStage.Hull,
            "domain": rd.ShaderStage.Domain,
            "geometry": rd.ShaderStage.Geometry,
            "pixel": rd.ShaderStage.Pixel,
            "compute": rd.ShaderStage.Compute,
        }
        stage_lower = stage_str.lower()
        if stage_lower not in stage_map:
            raise ValueError("Unknown shader stage: %s" % stage_str)
        return stage_map[stage_lower]

    @staticmethod
    def parse_resource_id(resource_id_str, controller):
        """Resolve an opaque ResourceId from captured resources on the replay thread.

        ResourceId.id is private: assigning a Python attribute does not set its native value.
        """
        target = Parsers.extract_numeric_id(resource_id_str)
        if not 0 < target <= 0xFFFFFFFFFFFFFFFF:
            raise ValueError("Expected a non-null 64-bit resource ID")
        for resource in controller.GetResources():
            if Parsers.extract_numeric_id(str(resource.resourceId)) == target:
                return resource.resourceId
        raise ValueError("Unknown resource_id: " + resource_id_str)

    @staticmethod
    def extract_numeric_id(resource_id_str):
        """Extract numeric ID from resource ID string"""
        if "::" in resource_id_str:
            return int(resource_id_str.split("::")[-1])
        return int(resource_id_str)
