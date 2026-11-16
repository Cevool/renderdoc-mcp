"""Start a RenderDoc GUI process and wait for its bridge instance (server side)."""

import asyncio
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Annotated

from fastmcp.exceptions import ToolError
from pydantic import Field

from .bridge.client import RenderDocBridgeError


def resolve_qrenderdoc(explicit=None):
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise ToolError("renderdoc_path does not exist: " + explicit)
        return path.resolve()
    candidates = [os.environ.get("RENDERDOC_MCP_QRENDERDOC")]
    if os.name == "nt":
        candidates.append(str(Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
                              / "RenderDoc" / "qrenderdoc.exe"))
    candidates.append(shutil.which("qrenderdoc"))
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate).resolve()
    raise ToolError("qrenderdoc not found; pass renderdoc_path or set RENDERDOC_MCP_QRENDERDOC")


def start_detached(command, cwd, env):
    """The GUI must outlive the MCP server process (and any job object it runs in)."""
    if os.name != "nt":
        return subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                start_new_session=True)
    base = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    for flags in (base | subprocess.CREATE_BREAKAWAY_FROM_JOB, base):
        try:
            return subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                    close_fds=True, creationflags=flags)
        except OSError:
            if flags == base:
                raise  # Breakaway is refused inside some job objects; retried without it.


def same_file(a, b):
    return bool(a and b) and os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def register_launcher_tools(mcp, bridge):
    @mcp.tool
    async def launch_renderdoc(capture_path: str | None = None, renderdoc_path: str | None = None,
                               wait_seconds: Annotated[float, Field(ge=5, le=1800)] = 180) -> dict:
        """Start a new RenderDoc window, optionally loading an .rdc, and return its instance_id.

        Waits until the MCP bridge in that process registers (and the capture finishes loading).
        The bridge must already be installed and enabled in RenderDoc's Manage Extensions; that
        setting persists. The window stays open after this server exits; close it in RenderDoc.
        """
        capture = None
        if capture_path is not None:
            capture = Path(capture_path)
            if not capture.is_absolute() or not capture.is_file() or capture.suffix.lower() != ".rdc":
                raise ToolError("capture_path must be an existing absolute .rdc file")
            capture = str(capture.resolve())
        executable = resolve_qrenderdoc(renderdoc_path)
        env = {**os.environ, "RENDERDOC_MCP_IPC_DIR": str(bridge.root)}
        process = start_detached([str(executable)] + ([capture] if capture else []),
                                 str(executable.parent), env)
        started = time.monotonic()
        deadline = started + wait_seconds
        instance, status = None, None
        while time.monotonic() < deadline:
            code = process.poll()
            if code is not None:
                raise ToolError("RenderDoc exited during startup with code %s" % code)
            if instance is None:
                listed = await asyncio.to_thread(bridge.list_instances)
                instance = next((item for item in listed["instances"] if item.get("pid") == process.pid), None)
            if instance is not None:
                if capture is None:
                    break
                try:
                    status = await asyncio.to_thread(bridge.call, instance["instance_id"],
                                                    "get_capture_status", None, 5.0)
                except RenderDocBridgeError:
                    status = None  # Busy loading; keep polling.
                if status and status.get("loaded") and same_file(status.get("filename"), capture):
                    break
            await asyncio.sleep(0.5)
        result = {"pid": process.pid, "executable": str(executable), "capture_path": capture,
                  "elapsed_seconds": round(time.monotonic() - started, 1)}
        if instance is None:
            return {**result, "status": "bridge_not_detected", "hint": (
                "RenderDoc started but no MCP bridge registered. Install it (scripts/install_extension.py) "
                "and enable 'RenderDoc MCP Bridge' under Tools > Manage Extensions, then call "
                "list_instances. The window was left open.")}
        result["instance_id"] = instance["instance_id"]
        if capture is not None and not (status and status.get("loaded")
                                        and same_file(status.get("filename"), capture)):
            return {**result, "status": "capture_loading", "capture": status,
                    "hint": "Poll get_capture_status with this instance_id until loaded"}
        return {**result, "status": "ready", "capture": status}
