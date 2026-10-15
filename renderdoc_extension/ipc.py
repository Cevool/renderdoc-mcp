"""Qt-independent IPC engine, compatible with RenderDoc's Python 3.6+."""

import json
import os
import re
import tempfile
import time
import uuid

PROTOCOL_VERSION = 2
ID_PATTERN = re.compile(r"[0-9a-f]{32}\Z")


def atomic_json(path, value):
    staging = path + "." + uuid.uuid4().hex + ".tmp"
    try:
        with open(staging, "w", encoding="utf-8") as stream:
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
        if os.path.exists(staging):
            os.remove(staging)


class FileBridge:
    """Each instance owns its queue; the Qt thread processes requests serially."""

    def __init__(self, handler, root=None, version="", status_provider=None):
        self.handler = handler
        self.root = str(root or os.environ.get(
            "RENDERDOC_MCP_IPC_DIR", os.path.join(tempfile.gettempdir(), "renderdoc_mcp")
        ))
        self.instance_id = uuid.uuid4().hex
        self.directory = os.path.join(self.root, "instances", self.instance_id)
        self.requests = os.path.join(self.directory, "requests")
        self.responses = os.path.join(self.directory, "responses")
        self.manifest_path = os.path.join(self.directory, "instance.json")
        self.version = version
        self.status_provider = status_provider
        self.started_at = time.time()
        self.running = False
        self._processing = False
        self._last_status = None
        self._last_refresh = 0

    def start(self):
        for directory in (self.requests, self.responses):
            os.makedirs(directory, exist_ok=True)
        self.running = True
        self.refresh()

    def refresh(self):
        if not self.running:
            return
        status = self.status_provider() if self.status_provider else {"loaded": False}
        if status == self._last_status and time.monotonic() - self._last_refresh < 5:
            return
        atomic_json(self.manifest_path, {
            "protocol_version": PROTOCOL_VERSION, "instance_id": self.instance_id,
            "pid": os.getpid(), "renderdoc_version": self.version, "bridge_version": "2.0.0",
            "started_at": self.started_at, "heartbeat_at": time.time(), "capture": status,
        })
        self._last_status = status
        self._last_refresh = time.monotonic()
        cutoff = time.time() - 120
        for directory in (self.responses, self.requests):
            for name in os.listdir(directory):
                path = os.path.join(directory, name)
                try:
                    if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                        os.remove(path)
                except OSError:
                    pass

    def stop(self):
        self.running = False
        for attempt in range(8):
            try:
                os.remove(self.manifest_path)
                break
            except FileNotFoundError:
                break
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(0.005 * (attempt + 1))

    def drain(self, max_requests=8):
        """Bound work per UI turn; return whether more requests may be pending."""
        if not self.running or self._processing:
            return False
        self._processing = True
        try:
            names = sorted(name for name in os.listdir(self.requests) if name.endswith(".json"))
            progressed = False
            for name in names[:max_requests]:
                if not self.running:
                    break
                progressed = self._process(name) or progressed
            # Notifications can arrive reentrantly while BlockInvoke is running. Rescan
            # rather than relying on the initial snapshot to avoid a five-second delay.
            return progressed and self.running and any(
                name.endswith(".json") for name in os.listdir(self.requests)
            )
        finally:
            self._processing = False

    def _process(self, name):
        request_id = name[:-5]
        path = os.path.join(self.requests, name)
        try:
            with open(path, "r", encoding="utf-8") as stream:
                request = json.load(stream)
            os.remove(path)
        except ValueError:
            # Malformed files must not produce an endless zero-delay Qt drain loop.
            try:
                os.remove(path)
                return True
            except OSError:
                return False
        except OSError:
            return False  # Wait for the fallback timer if every file is temporarily locked.
        if not ID_PATTERN.fullmatch(request_id) or not isinstance(request, dict):
            return True
        expires_at = request.get("expires_at", 0)
        if not isinstance(expires_at, (int, float)) or expires_at <= time.time():
            return True
        if (request.get("id") != request_id
                or request.get("instance_id") != self.instance_id
                or request.get("protocol_version") != PROTOCOL_VERSION):
            response = {"error": {"code": -32600, "message": "Request identity/protocol mismatch"}}
        else:
            try:
                response = self.handler.handle(request)
            except Exception as exc:
                response = {"error": {"code": -32603, "message": str(exc)}}
        response.update({
            "id": request_id, "instance_id": self.instance_id,
            "protocol_version": PROTOCOL_VERSION,
        })
        if self.running and time.time() < expires_at:
            atomic_json(os.path.join(self.responses, name), response)
        self.refresh()
        return True
