"""Category-first discovery over FastMCP's authenticated operation catalog."""

from typing import Annotated, Literal

from fastmcp.exceptions import ToolError
from fastmcp.server.context import Context
from fastmcp.server.transforms.search.base import BaseSearchTransform
from fastmcp.tools import Tool
from pydantic import Field
from pydantic_core import SchemaError, SchemaValidator, ValidationError, core_schema

CATEGORIES = {
    "capture": "Instances and capture files",
    "actions": "Frame overview, draws, reverse lookup and GPU timings",
    "pipeline": "Bindings, native fixed state, constants and comparisons",
    "resources": "Textures, buffers, usage and exports",
    "analysis": "Post-VS geometry, pixel history, shader debugging and API events",
    "artifacts": "View images and read saved results",
}
OPERATIONS = {
    "capture": "get_capture_status list_captures open_capture",
    "actions": "get_frame_summary get_draw_calls get_draw_call_details get_action_timings "
               "find_draws_by_shader find_draws_by_texture find_draws_by_resource",
    "pipeline": "get_shader_info get_pipeline_state get_pipeline_states compare_pipeline_states "
                "get_native_pipeline_state get_constant_buffer compare_capture_states",
    "resources": "get_buffer_contents get_texture_info get_texture_data export_texture get_resource_usage",
    "analysis": "get_post_vs_data get_pixel_history debug_shader get_api_events",
    "artifacts": "view_image_artifact read_artifact",
}
TOOL_CATEGORY = {name: category for category, names in OPERATIONS.items() for name in names.split()}
Offset = Annotated[int, Field(ge=0)]
PageSize = Annotated[int, Field(ge=1, le=50)]


def category_of(tool):
    # Unclassified future operations remain discoverable and fail the coverage test.
    return TOOL_CATEGORY.get(tool.name, "other")


def summary(tool):
    return {"name": tool.name, "category": category_of(tool),
            "description": (tool.description or "").strip().split("\n")[0]}


def page(tools, offset, limit, detail="summary"):
    # Full schemas are deliberately limited to three per page.
    size = min(limit, 3) if detail == "schema" else limit
    selected = tools[offset:offset + size]
    next_offset = offset + len(selected) if offset + len(selected) < len(tools) else None
    return {"tools": [summary(t) if detail == "summary" else t.to_mcp_tool().model_dump(
                mode="json", by_alias=True, exclude_none=True) for t in selected],
            "returned": len(selected), "total_matches": len(tools), "offset": offset,
            "truncated": next_offset is not None, "next_offset": next_offset,
            "hint": "Continue with next_offset; get_tool_schema(name) returns one exact schema."}


class CategorizedDiscovery(BaseSearchTransform):
    """Keep execution/auth handling upstream; replace only the discovery interface."""

    def __init__(self):
        super().__init__(always_visible=["list_instances"])

    async def transform_tools(self, tools):
        return [*await super().transform_tools(tools), self._browse_tool(), self._schema_tool()]

    async def get_tool(self, name, call_next, *, version=None):
        if name == "browse_tools":
            return self._browse_tool()
        if name == "get_tool_schema":
            return self._schema_tool()
        return await super().get_tool(name, call_next, version=version)

    async def _catalog(self, ctx, category=None):
        tools = await self._get_visible_tools(ctx)
        if category is not None:
            known = set(CATEGORIES) | {category_of(t) for t in tools}
            if category not in known:
                raise ToolError("Unknown category; call browse_tools() for the directory")
            tools = [t for t in tools if category_of(t) == category]
        return sorted(tools, key=lambda t: t.name)

    def _browse_tool(self):
        async def browse_tools(category: str | None = None, offset: Offset = 0,
                               limit: PageSize = 20, ctx: Context = None) -> dict:
            """Browse categories first, then a category's names and purposes without schemas.

            Omit category for the complete directory. Use get_tool_schema(name) before calling.
            """
            tools = await self._catalog(ctx, category)
            if category is not None:
                return {"category": category, **page(tools, offset, limit)}
            counts = {}
            for tool in tools:
                key = category_of(tool)
                counts[key] = counts.get(key, 0) + 1
            return {"categories": [{"name": key, "description": CATEGORIES.get(key, "Other operations"),
                                    "count": counts[key]} for key in sorted(counts)],
                    "total_tools": len(tools), "complete": True,
                    "hint": "browse_tools(category) -> get_tool_schema(name) -> call_tool(name, arguments)."}
        return Tool.from_function(browse_tools)

    def _schema_tool(self):
        async def get_tool_schema(name: str, ctx: Context = None) -> dict:
            """Get the full input/output schema of one exact operation name from the directory."""
            for tool in await self._catalog(ctx):
                if tool.name == name:
                    return tool.to_mcp_tool().model_dump(mode="json", by_alias=True, exclude_none=True)
            raise ToolError("Unknown operation: %s. Browse the directory or search_tools first." % name)
        return Tool.from_function(get_tool_schema)

    def _make_search_tool(self):
        async def search_tools(pattern: str, category: str | None = None, offset: Offset = 0,
                               limit: PageSize = 10, detail: Literal["summary", "schema"] = "summary",
                               ctx: Context = None) -> dict:
            """Regex search with explicit totals and pagination; summaries by default.

            Matches names, descriptions and parameters. Exact names rank first. Invalid regex is an
            error, not an empty catalog. schema detail returns at most three definitions per page.
            """
            tools = await self._catalog(ctx, category)
            return page(list(await self._search(tools, pattern)), offset, limit, detail)
        return Tool.from_function(search_tools)

    async def _search(self, tools, query):
        try:
            matcher = SchemaValidator(core_schema.str_schema(
                pattern="(?i)" + query, regex_engine="rust-regex"))
        except SchemaError as exc:
            raise ToolError("Invalid or unsupported regex: %s" % query) from exc
        matches = []
        for tool in tools:
            text = " ".join([tool.name, tool.description or "", *tool.parameters.get("properties", {})])
            # Match names separately so ^get_texture_data$ works as advertised.
            for candidate in (tool.name, text):
                try:
                    matcher.validate_python(candidate)
                except ValidationError:
                    continue
                matches.append(tool)
                break
        return sorted(matches, key=lambda t: (t.name.casefold() != query.casefold(), t.name))
