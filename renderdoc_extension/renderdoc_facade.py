"""
RenderDoc API Facade
Provides thread-safe access to RenderDoc's ReplayController and CaptureContext.
Uses BlockInvoke to marshal calls to the replay thread.
"""

from .services import (
    CaptureManager,
    ActionService,
    SearchService,
    ResourceService,
    PipelineService,
)


class RenderDocFacade:
    """
    Facade for RenderDoc API access.

    This class delegates all operations to specialized service classes:
    - CaptureManager: Capture management (status, list, open)
    - ActionService: Draw call / action operations
    - SearchService: Reverse lookup searches
    - ResourceService: Texture and buffer data
    - PipelineService: Pipeline state and shader info
    """

    def __init__(self, ctx):
        """
        Initialize facade with CaptureContext.

        Args:
            ctx: The pyrenderdoc CaptureContext from register()
        """
        self.ctx = ctx

        # Initialize service classes
        self._capture = CaptureManager(ctx, self._invoke)
        self._action = ActionService(ctx, self._invoke)
        self._search = SearchService(ctx, self._invoke)
        self._resource = ResourceService(ctx, self._invoke)
        self._pipeline = PipelineService(ctx, self._invoke)

    def _invoke(self, callback):
        """Inspect on the replay thread, restoring the UI-selected event even after errors."""
        selected_event = self.ctx.CurEvent()
        failure = [None]

        def preserving_event(controller):
            try:
                callback(controller)
            except Exception as exc:
                failure[0] = exc
            finally:
                try:
                    controller.SetFrameEvent(selected_event, True)
                except Exception as exc:
                    if failure[0] is None:
                        failure[0] = exc

        self.ctx.Replay().BlockInvoke(preserving_event)
        if failure[0] is not None:
            raise failure[0]

    # ==================== Capture Management ====================

    def get_capture_status(self):
        """Check if a capture is loaded and get API info"""
        return self._capture.get_capture_status()

    def list_captures(self, directory):
        """List all .rdc files in the specified directory"""
        return self._capture.list_captures(directory)

    def open_capture(self, capture_path):
        """Open a capture file in RenderDoc"""
        return self._capture.open_capture(capture_path)

    # ==================== Draw Call / Action Operations ====================

    def get_draw_calls(
        self,
        include_children=True,
        marker_filter=None,
        exclude_markers=None,
        event_id_min=None,
        event_id_max=None,
        only_actions=False,
        flags_filter=None,
    ):
        """Get all draw calls/actions in the capture with optional filtering"""
        return self._action.get_draw_calls(
            include_children=include_children,
            marker_filter=marker_filter,
            exclude_markers=exclude_markers,
            event_id_min=event_id_min,
            event_id_max=event_id_max,
            only_actions=only_actions,
            flags_filter=flags_filter,
        )

    def get_frame_summary(self):
        """Get a summary of the current capture frame"""
        return self._action.get_frame_summary()

    def get_draw_call_details(self, event_id):
        """Get detailed information about a specific draw call"""
        return self._action.get_draw_call_details(event_id)

    def get_action_timings(self, event_ids=None, marker_filter=None, exclude_markers=None):
        """Get GPU timing information for actions"""
        return self._action.get_action_timings(
            event_ids=event_ids,
            marker_filter=marker_filter,
            exclude_markers=exclude_markers,
        )

    # ==================== Search Operations ====================

    def find_draws_by_shader(self, shader_name, stage=None):
        """Find all draw calls using a shader with the given name (partial match)"""
        return self._search.find_draws_by_shader(shader_name, stage)

    def find_draws_by_texture(self, texture_name):
        """Find all draw calls using a texture with the given name (partial match)"""
        return self._search.find_draws_by_texture(texture_name)

    def find_draws_by_resource(self, resource_id):
        """Find all draw calls using a specific resource ID (exact match)"""
        return self._search.find_draws_by_resource(resource_id)

    # ==================== Resource Operations ====================

    def get_buffer_contents(self, resource_id, offset=0, length=0, event_id=None, output_path=None):
        """Get buffer data"""
        return self._resource.get_buffer_contents(resource_id, offset, length, event_id, output_path)

    def get_texture_info(self, resource_id):
        """Get texture metadata"""
        return self._resource.get_texture_info(resource_id)

    def get_texture_data(self, resource_id, mip=0, slice=0, sample=0, depth_slice=None,
                         event_id=None, output_path=None):
        """Get texture pixel data"""
        return self._resource.get_texture_data(
            resource_id, mip, slice, sample, depth_slice, event_id, output_path
        )

    def export_texture(self, resource_id, event_id, output_path, mip=0, slice=0):
        return self._resource.export_texture(resource_id, event_id, output_path, mip, slice)

    # ==================== Pipeline Operations ====================

    def get_shader_info(self, event_id, stage):
        """Get shader information for a specific stage"""
        return self._pipeline.get_shader_info(event_id, stage)

    def get_pipeline_state(self, event_id):
        """Get full pipeline state at an event"""
        return self._pipeline.get_pipeline_state(event_id)

    def get_pipeline_states(self, event_ids, fields=None):
        if not isinstance(event_ids, list) or not 1 <= len(event_ids) <= 32:
            raise ValueError("Expected 1..32 event IDs")
        states = []
        for event_id in event_ids:
            state = self.get_pipeline_state(event_id)
            if fields:
                state = {key: state[key] for key in fields if key in state}
            states.append({"event_id": event_id, "state": state})
        return {"states": states}

    def compare_pipeline_states(self, event_a, event_b, fields=None):
        from .utils.diff import differences
        states = self.get_pipeline_states([event_a, event_b], fields)["states"]
        before = dict(states[0]["state"])
        after = dict(states[1]["state"])
        before.pop("event_id", None)
        after.pop("event_id", None)
        changes = differences(before, after)

        def incomplete(value):
            if isinstance(value, dict):
                return bool(value.get("unavailable") or value.get("error")) or any(
                    incomplete(child) for child in value.values()
                )
            if isinstance(value, list):
                return any(incomplete(child) for child in value)
            return False

        complete = not incomplete(before) and not incomplete(after)
        return {"event_a": event_a, "event_b": event_b,
                "equal": not changes if complete else None, "complete": complete,
                "changes": changes}
