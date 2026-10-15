"""Instance discovery and correlated, atomic file IPC. No work runs while idle."""

import ctypes
import json
import os
import re
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = 2
ID_PATTERN = re.compile(r"[0-9a-f]{32}\Z")


class RenderDocBridgeError(Exception):
    """An instance disappeared, rejected a request, or failed to answer."""


def process_alive(pid: int, created_before: float | None = None) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        kernel.GetProcessTimes.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_uint64)] * 4
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() == 5
        try:
            code = ctypes.c_uint32()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value != 259:
                return False
            if created_before is not None:
                times = [ctypes.c_uint64() for _ in range(4)]
                if kernel.GetProcessTimes(handle, *(ctypes.byref(value) for value in times)):
                    created_at = times[0].value / 10_000_000 - 11_644_473_600
                    if created_at > created_before:
                        return False  # The OS reused an exited instance's PID.
            return True
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except ProcessLookupError:
        return False


def atomic_json(path: Path, value: Any) -> None:
    staging = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with staging.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False)
        for attempt in range(8):
            try:
                os.replace(staging, path)
                break
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(0.005 * (attempt + 1))
    finally:
        staging.unlink(missing_ok=True)


class RenderDocBridge:
    def __init__(self, root: str | Path | None = None, timeout: float = 30.0):
        self.root = Path(root or os.environ.get(
            "RENDERDOC_MCP_IPC_DIR", str(Path(tempfile.gettempdir()) / "renderdoc_mcp")
        ))
        self.timeout = timeout

    def _manifest(self, instance_id: str) -> dict:
        if not ID_PATTERN.fullmatch(instance_id):
            raise RenderDocBridgeError("Invalid instance_id; use list_instances first")
        try:
            manifest = json.loads(
                (self.root / "instances" / instance_id / "instance.json").read_text("utf-8")
            )
            if manifest["instance_id"] != instance_id:
                raise ValueError("Instance identity mismatch")
            if manifest["protocol_version"] != PROTOCOL_VERSION:
                raise ValueError("Incompatible bridge protocol; update the RenderDoc extension")
            if not process_alive(int(manifest["pid"]), float(manifest["started_at"])):
                raise ValueError("RenderDoc process exited")
            if time.time() - float(manifest["heartbeat_at"]) > 30:
                raise ValueError("RenderDoc bridge is unresponsive (stale heartbeat)")
            return manifest
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise RenderDocBridgeError(
                f"Instance {instance_id} is unavailable: {exc}. Refresh list_instances."
            ) from exc

    def list_instances(self) -> dict:
        instances = []
        for path in sorted((self.root / "instances").glob("*/instance.json")):
            try:
                instances.append(self._manifest(path.parent.name))
            except RenderDocBridgeError:
                continue
        legacy = (self.root / "request.json").exists() or (self.root / "response.json").exists()
        return {
            "instances": instances, "count": len(instances),
            "hint": "Pass the chosen instance_id in every operation's arguments.",
            "legacy_ipc_detected": legacy,
        }

    def call(self, instance_id: str, method: str, params: dict | None = None) -> Any:
        manifest = self._manifest(instance_id)
        directory = self.root / "instances" / instance_id
        request_id = uuid.uuid4().hex
        request_path = directory / "requests" / (request_id + ".json")
        response_path = directory / "responses" / (request_id + ".json")
        deadline = time.monotonic() + self.timeout
        request = {
            "protocol_version": PROTOCOL_VERSION, "instance_id": instance_id,
            "id": request_id, "method": method, "params": params or {},
            "expires_at": time.time() + self.timeout,
        }
        try:
            atomic_json(request_path, request)
            delay = 0.01
            while time.monotonic() < deadline:
                if response_path.exists():
                    response = json.loads(response_path.read_text("utf-8"))
                    if (response.get("id") != request_id
                            or response.get("instance_id") != instance_id
                            or response.get("protocol_version") != PROTOCOL_VERSION):
                        raise RenderDocBridgeError("Response identity/protocol mismatch")
                    if "error" in response:
                        error = response["error"]
                        raise RenderDocBridgeError(f"[{error['code']}] {error['message']}")
                    return response.get("result")
                if not (directory / "instance.json").exists() or not process_alive(
                    manifest["pid"], manifest["started_at"]
                ):
                    raise RenderDocBridgeError("Selected RenderDoc instance exited or unloaded")
                time.sleep(min(delay, max(0, deadline - time.monotonic())))
                delay = min(delay * 1.5, 0.2)
            raise RenderDocBridgeError(
                "Request timed out; a started operation may still finish. Refresh status before retrying."
            )
        except RenderDocBridgeError:
            raise
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise RenderDocBridgeError(f"Communication error: {exc}") from exc
        finally:
            request_path.unlink(missing_ok=True)
            response_path.unlink(missing_ok=True)
