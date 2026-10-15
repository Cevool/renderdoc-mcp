import json
import os
import threading
import time
from pathlib import Path

import pytest

from mcp_server.bridge.client import RenderDocBridge, RenderDocBridgeError
from renderdoc_extension.ipc import FileBridge


def test_waiting_request_fails_promptly_when_extension_unloads(tmp_path):
    server = FileBridge(None, root=tmp_path)
    server.start()

    def stop():
        while not list(Path(server.requests).glob("*.json")):
            time.sleep(0.005)
        server.stop()

    thread = threading.Thread(target=stop)
    thread.start()
    started = time.monotonic()
    with pytest.raises(RenderDocBridgeError, match="exited or unloaded"):
        RenderDocBridge(tmp_path, timeout=5).call(server.instance_id, "never")
    thread.join(2)
    assert time.monotonic() - started < 1


def test_expired_queued_request_is_discarded(tmp_path):
    from mcp_server.bridge.client import atomic_json
    server = FileBridge(None, root=tmp_path)
    server.start()
    request_id = "a" * 32
    atomic_json(Path(server.requests) / (request_id + ".json"), {
        "id": request_id, "instance_id": server.instance_id, "protocol_version": 2,
        "expires_at": time.time() - 1, "method": "never", "params": {},
    })
    assert server.drain() is False
    assert not list(Path(server.responses).glob("*.json"))
    server.stop()


def test_requests_arriving_during_drain_are_not_left_for_fallback_timer(tmp_path):
    from mcp_server.bridge.client import atomic_json
    calls = []
    server = None

    class Handler:
        def handle(self, request):
            calls.append(request["id"])
            if len(calls) == 1:
                write_request("b" * 32)
            return {"result": True}

    def write_request(request_id):
        atomic_json(Path(server.requests) / (request_id + ".json"), {
            "id": request_id, "instance_id": server.instance_id, "protocol_version": 2,
            "expires_at": time.time() + 10, "method": "echo", "params": {},
        })

    server = FileBridge(Handler(), root=tmp_path)
    server.start()
    write_request("a" * 32)
    assert server.drain() is True
    server.drain()
    assert calls == ["a" * 32, "b" * 32]
    server.stop()


def test_atomic_manifest_refresh_retries_windows_sharing_violation(tmp_path, monkeypatch):
    import renderdoc_extension.ipc as ipc
    server = FileBridge(None, root=tmp_path)
    real_replace = ipc.os.replace
    failures = []

    def contested_replace(source, target):
        if len(failures) < 2:
            failures.append(True)
            raise PermissionError("file temporarily open by another client")
        real_replace(source, target)

    monkeypatch.setattr(ipc.os, "replace", contested_replace)
    server.start()
    assert json.loads(Path(server.manifest_path).read_text())["instance_id"] == server.instance_id
    assert len(failures) == 2
    server.stop()


def test_malformed_request_does_not_trigger_an_idle_busy_loop(tmp_path):
    server = FileBridge(None, root=tmp_path)
    server.start()
    malformed = Path(server.requests) / ("a" * 32 + ".json")
    malformed.write_text("{broken")
    assert server.drain() is False
    assert not malformed.exists()
    server.stop()


def test_unreadable_queue_does_not_reschedule_immediately(tmp_path, monkeypatch):
    server = FileBridge(None, root=tmp_path)
    server.start()
    (Path(server.requests) / ("a" * 32 + ".json")).write_text("{}")
    monkeypatch.setattr(server, "_process", lambda name: False)
    assert server.drain() is False
    server.stop()


@pytest.mark.skipif(os.name != "nt", reason="Windows process creation time guard")
def test_reused_pid_guard_rejects_process_newer_than_manifest():
    from mcp_server.bridge.client import process_alive
    assert process_alive(os.getpid())
    assert not process_alive(os.getpid(), created_before=0)
