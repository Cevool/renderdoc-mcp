"""Capture-session identity, MCP-owned replay edits and event-bound queries (Python 3.6)."""

import hashlib
import json
import os
import uuid


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      indent=2).encode("utf-8")


def write_file(path, data):
    """Create a new file; never overwrite existing evidence."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "xb") as stream:
        stream.write(data)
    return {"path": os.path.abspath(path), "bytes": len(data), "sha256": digest(data)}


def enum_name(value):
    return str(value).rsplit(".", 1)[-1]


class ReplayState(object):
    """State shared by edits and evidence for the currently loaded capture.

    A capture load/close starts a new session: replacement handles from the previous replay
    are gone, so the registry is discarded rather than reused.
    """

    def __init__(self, ctx):
        self.ctx = ctx
        self.reset()

    def reset(self):
        self.session = uuid.uuid4().hex
        self.generation = 0
        # str(original ResourceId) -> record holding the native original/target IDs.
        self.replacements = {}

    def identity(self, controller):
        return {"session_id": self.session, "filename": self.ctx.GetCaptureFilename(),
                "api": enum_name(controller.GetAPIProperties().pipelineType),
                "edit_generation": self.generation,
                "shader_replacements": sorted(self.replacements)}


def last_event_id(ctx):
    best, actions = 0, ctx.CurRootActions()
    while actions:
        best = max(best, actions[-1].eventId)
        actions = actions[-1].children
    return best


def event_query(ctx, invoke, state, event_id, read, require_action=True):
    """Replay through event_id, run read(controller) and attach capture/edit identity.

    require_action=False also accepts API events between actions (e.g. a Map/Unmap or copy),
    which buffer reads need to observe CPU writes.
    """
    if not ctx.IsCaptureLoaded():
        raise ValueError("No capture loaded")
    if event_id is not None and event_id != 0:
        if not require_action:
            if event_id > last_event_id(ctx):
                raise ValueError("event_id %s is beyond the last event %s" % (event_id, last_event_id(ctx)))
        elif ctx.GetAction(event_id) is None:
            raise ValueError("No action at event_id %s" % event_id)
    result = {}

    def callback(controller):
        if event_id is not None:
            controller.SetFrameEvent(event_id, True)
        result.update(read(controller))
        result["capture"] = state.identity(controller)
        if event_id is not None:
            result["event_id"] = event_id

    invoke(callback)
    return result
