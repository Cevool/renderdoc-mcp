import concurrent.futures
import json
import os
import threading
import time
import uuid
from pathlib import Path

import pytest

from mcp_server.bridge.client import RenderDocBridge, RenderDocBridgeError, atomic_json
from renderdoc_extension.ipc import FileBridge


class EchoHandler:
    def __init__(self, label):
        self.label = label
        self.calls = []

    def handle(self, request):
        self.calls.append(request)
        time.sleep(request["params"].get("delay", 0))
        return {"result": {"instance": self.label, "value": request["params"].get("value")}}


@pytest.fixture
def instances(tmp_path):
    servers = [FileBridge(EchoHandler(label), root=tmp_path) for label in ("A", "B")]
    for server in servers:
        server.start()
    stop = threading.Event()

    def poll(server):
        while not stop.is_set():
            server.drain()
            stop.wait(0.002)

    threads = [threading.Thread(target=poll, args=(server,)) for server in servers]
    for thread in threads:
        thread.start()
    yield servers
    stop.set()
    for thread in threads:
        thread.join(3)
    for server in servers:
        server.stop()


def test_two_instances_four_clients_no_cross_talk(tmp_path, instances):
    clients = [RenderDocBridge(tmp_path, timeout=5) for _ in range(4)]
    assert clients[0].list_instances()["count"] == 2

    def call(index):
        target = instances[index % 2]
        return clients[index % 4].call(target.instance_id, "echo", {"value": index})

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(call, range(80)))
    assert results == [{"instance": "A" if i % 2 == 0 else "B", "value": i} for i in range(80)]
    for server in instances:
        assert not list(Path(server.requests).glob("*.json"))
        assert not list(Path(server.responses).glob("*.json"))


def test_stop_does_not_delete_other_instances_requests(tmp_path):
    first, second = [FileBridge(EchoHandler(label), root=tmp_path) for label in ("A", "B")]
    first.start()
    second.start()
    pending = Path(second.requests) / (uuid.uuid4().hex + ".json")
    pending.write_text("{}")
    first.stop()
    assert pending.exists()
    assert RenderDocBridge(tmp_path).list_instances()["count"] == 1
    second.stop()


def test_restart_invalidates_previous_id(tmp_path):
    old = FileBridge(EchoHandler("old"), root=tmp_path)
    old.start()
    old.stop()
    new = FileBridge(EchoHandler("new"), root=tmp_path)
    new.start()
    assert old.instance_id != new.instance_id
    with pytest.raises(RenderDocBridgeError, match="unavailable"):
        RenderDocBridge(tmp_path).call(old.instance_id, "echo")
    new.stop()


@pytest.mark.parametrize("identity", ["id", "instance_id", "protocol_version"])
def test_response_identity_validation(tmp_path, identity):
    server = FileBridge(EchoHandler("A"), root=tmp_path)
    server.start()

    def corrupt():
        while not list(Path(server.requests).glob("*.json")):
            time.sleep(0.002)
        request_path = next(Path(server.requests).glob("*.json"))
        value = json.loads(request_path.read_text())
        response = {key: value[key] for key in ("id", "instance_id", "protocol_version")}
        response[identity] = "wrong"
        response["result"] = "must never be returned"
        atomic_json(Path(server.responses) / request_path.name, response)

    thread = threading.Thread(target=corrupt)
    thread.start()
    with pytest.raises(RenderDocBridgeError, match="mismatch"):
        RenderDocBridge(tmp_path, timeout=1).call(server.instance_id, "echo")
    thread.join(2)
    server.stop()


def test_expired_request_never_executes(tmp_path):
    server = FileBridge(EchoHandler("A"), root=tmp_path)
    server.start()
    with pytest.raises(RenderDocBridgeError, match="timed out"):
        RenderDocBridge(tmp_path, timeout=0.04).call(server.instance_id, "echo")
    server.drain()
    assert server.handler.calls == []
    assert not list(Path(server.requests).glob("*.json"))
    server.stop()


def test_late_result_cannot_satisfy_next_request(tmp_path, instances):
    target = instances[0]
    with pytest.raises(RenderDocBridgeError, match="timed out"):
        RenderDocBridge(tmp_path, timeout=0.05).call(target.instance_id, "echo", {"delay": 0.15})
    assert RenderDocBridge(tmp_path, timeout=2).call(
        target.instance_id, "echo", {"value": "next"}
    ) == {"instance": "A", "value": "next"}
    assert not list(Path(target.responses).glob("*.json"))


def test_discovery_skips_dead_stale_invalid_and_legacy(tmp_path):
    server = FileBridge(EchoHandler("A"), root=tmp_path)
    server.start()
    manifest = json.loads(Path(server.manifest_path).read_text())
    for pid, age in ((os.getpid(), 100), (2147483647, 0)):
        atomic_json(Path(server.manifest_path), {**manifest, "pid": pid, "heartbeat_at": time.time() - age})
        assert RenderDocBridge(tmp_path).list_instances()["count"] == 0
    Path(server.manifest_path).write_text("invalid JSON")
    (tmp_path / "request.json").write_text("legacy")
    result = RenderDocBridge(tmp_path).list_instances()
    assert result["count"] == 0 and result["legacy_ipc_detected"]
    assert (tmp_path / "request.json").read_text() == "legacy"
    server.stop()


@pytest.mark.parametrize("instance_id", ["../escape", "", "A" * 32])
def test_invalid_instance_id_rejected(tmp_path, instance_id):
    with pytest.raises(RenderDocBridgeError, match="Invalid instance_id"):
        RenderDocBridge(tmp_path).call(instance_id, "echo")


def test_wrong_target_request_rejected_without_execution(tmp_path):
    server = FileBridge(EchoHandler("A"), root=tmp_path)
    server.start()
    request_id = uuid.uuid4().hex
    atomic_json(Path(server.requests) / (request_id + ".json"), {
        "id": request_id, "instance_id": uuid.uuid4().hex, "protocol_version": 2,
        "method": "echo", "params": {}, "expires_at": time.time() + 5,
    })
    server.drain()
    response = json.loads((Path(server.responses) / (request_id + ".json")).read_text())
    assert response["error"]["code"] == -32600
    assert server.handler.calls == []
    server.stop()
