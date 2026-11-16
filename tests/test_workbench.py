"""Replay edit ownership and RenderDoc launch flow, without a GPU."""

import json
import os
import subprocess
import sys
import textwrap
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastmcp import Client

rd = sys.modules.get("renderdoc") or types.ModuleType("renderdoc")
sys.modules["renderdoc"] = rd
for name, value in {
    "ResourceId": SimpleNamespace(Null=lambda: 0),
    "ShaderStage": SimpleNamespace(**{n: i for i, n in enumerate(
        ["Vertex", "Hull", "Domain", "Geometry", "Pixel", "Compute"])}),
    "ShaderEncoding": SimpleNamespace(DXBC=1, GLSL=2, SPIRV=3, SPIRVAsm=4, HLSL=5, DXIL=6,
                                      OpenGLSPIRV=7, OpenGLSPIRVAsm=8, Slang=9),
    "ShaderCompileFlags": type("ShaderCompileFlags", (), {"__init__": lambda self: setattr(self, "flags", [])}),
    "ShaderCompileFlag": type("ShaderCompileFlag", (), {}),
}.items():
    if not hasattr(rd, name):
        setattr(rd, name, value)

from renderdoc_extension.services.edit_service import EditService  # noqa: E402
from renderdoc_extension.services.replay_common import ReplayState  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SHADER = 7


class Replay:
    """Fake controller; replacement queries on ctx must never run inside BlockInvoke."""

    def __init__(self):
        self.calls, self.inside, self.next_target = [], False, 100
        self.flags = SimpleNamespace(flags=[SimpleNamespace(name="@cmdline", value="/O1 /T ps_5_0")])
        reflection = SimpleNamespace(entryPoint="main", debugInfo=SimpleNamespace(compileFlags=self.flags))
        self.pipe = SimpleNamespace(GetShader=lambda stage: SHADER, GetShaderReflection=lambda stage: reflection)
        self.ui = {}
        self.ctx = SimpleNamespace(
            IsCaptureLoaded=lambda: True, GetAction=lambda event: object(),
            GetCaptureFilename=lambda: "capture.rdc",
            IsResourceReplaced=lambda rid: self._ui(lambda: rid in self.ui),
            GetResourceReplacement=lambda rid: self._ui(lambda: self.ui.get(rid, 0)),
            RegisterReplacement=lambda a, b: self._ui(lambda: self.ui.__setitem__(a, b)),
            UnregisterReplacement=lambda a: self._ui(lambda: self.ui.pop(a, None)))

    def _ui(self, action):
        assert not self.inside, "CaptureContext UI call inside BlockInvoke deadlocks RenderDoc"
        return action()

    def invoke(self, callback):
        self.inside = True
        try:
            callback(self)
        finally:
            self.inside = False

    def SetFrameEvent(self, event, force):
        pass

    def GetPipelineState(self):
        return self.pipe

    def GetAPIProperties(self):
        return SimpleNamespace(pipelineType="D3D11")

    def GetTargetShaderEncodings(self):
        return [1, 5]

    def BuildTargetShader(self, entry, encoding, source, flags, stage):
        self.calls.append(("build", entry, [(f.name, f.value) for f in flags.flags]))
        if b"bad" in source:
            return 0, "error X3004: undeclared identifier"
        self.next_target += 1
        return self.next_target, ""

    def ReplaceResource(self, original, target):
        self.calls.append(("replace", original, target))

    def RemoveReplacement(self, original):
        self.calls.append(("remove", original))

    def FreeTargetResource(self, target):
        self.calls.append(("free", target))


def source(text):
    import base64
    return base64.b64encode(text.encode()).decode()


def test_replacements_swap_free_revert_and_keep_ui_calls_outside_replay():
    replay = Replay()
    edits = EditService(replay.ctx, replay.invoke, ReplayState(replay.ctx))
    first = edits.replace_shader(9, "pixel", "HLSL", source("float4 main() : SV_Target0 { return 1; }"))
    assert first["applied"] and first["generation"] == 1 and replay.ui == {SHADER: 101}
    assert first["compile_flags"] == [{"name": "@cmdline", "value": "/O1 /T ps_5_0"}]

    failed = edits.replace_shader(9, "pixel", "HLSL", source("bad"))
    assert failed["status"] == "compile_failed" and "X3004" in failed["compiler_output"]
    assert ("replace", SHADER, 101) == replay.calls[1] and len(replay.calls) == 3  # Nothing applied.

    second = edits.replace_shader(9, "pixel", "HLSL", source("float4 main() : SV_Target0 { return 0; }"),
                                  compile_flags={"@cmdline": "/Od /T ps_5_0"})
    assert second["replaced_previous_mcp_edit"] and replay.ui == {SHADER: 102}
    assert replay.calls[-4:] == [("build", "main", [("@cmdline", "/Od /T ps_5_0")]),
                                 ("remove", SHADER), ("free", 101), ("replace", SHADER, 102)]

    reverted = edits.revert_shader()
    assert [r["replacement_shader"] for r in reverted["reverted"]] == ["102"]
    assert replay.calls[-2:] == [("remove", SHADER), ("free", 102)]
    assert not reverted["remaining"] and replay.ui == {} and edits.state.generation == 3
    with pytest.raises(ValueError, match="No MCP replacement"):
        edits.revert_shader(str(SHADER))


def test_shader_edited_in_renderdoc_ui_is_not_taken_over():
    replay = Replay()
    replay.ui[SHADER] = 55
    edits = EditService(replay.ctx, replay.invoke, ReplayState(replay.ctx))
    with pytest.raises(ValueError, match="already replaced in the RenderDoc UI"):
        edits.replace_shader(9, "pixel", "HLSL", source("float4 main() : SV_Target0 { return 1; }"))
    assert not replay.calls and replay.ui == {SHADER: 55}


FAKE_RENDERDOC = textwrap.dedent("""
    import os, sys, time
    sys.path.insert(0, sys.argv[1])
    from renderdoc_extension.ipc import FileBridge
    capture, started = sys.argv[2], time.time()
    class Handler:
        def handle(self, request):
            # Simulates a capture that is still loading for the first second.
            return {"result": {"loaded": time.time() - started > 1.0, "filename": capture}}
    time.sleep(0.5)
    bridge = FileBridge(Handler())  # IPC root comes only from the inherited environment.
    bridge.start()
    while time.time() - started < 30:
        bridge.drain()
        time.sleep(0.02)
    bridge.stop()
""")


async def test_launch_matches_its_own_process_and_waits_for_capture(tmp_path, monkeypatch):
    from mcp_server import launcher, server
    from renderdoc_extension.ipc import FileBridge

    script, capture = tmp_path / "fake_renderdoc.py", tmp_path / "frame.rdc"
    script.write_text(FAKE_RENDERDOC, encoding="utf-8")
    capture.write_bytes(b"RDOC")
    decoy = FileBridge(None, root=tmp_path / "ipc")  # Another live instance in this process.
    decoy.start()
    started = []

    def start(command, cwd, env):
        assert command[1:] == [str(capture.resolve())]
        # A venv python.exe on Windows can be a trampoline with a different child PID.
        python = getattr(sys, "_base_executable", sys.executable)
        process = subprocess.Popen([python, str(script), str(ROOT), command[1]], env=env)
        started.append(process)
        return process

    monkeypatch.setattr(launcher, "start_detached", start)
    monkeypatch.setattr(server.bridge, "root", tmp_path / "ipc")
    try:
        async with Client(server.mcp) as client:
            result = await client.call_tool("call_tool", {"name": "launch_renderdoc", "arguments": {
                "capture_path": str(capture), "renderdoc_path": sys.executable, "wait_seconds": 20}})
        value = json.loads("".join(block.text for block in result.content if hasattr(block, "text")))
        assert value["status"] == "ready" and value["pid"] == started[0].pid
        assert value["instance_id"] != decoy.instance_id and value["capture"]["loaded"]
        assert os.path.samefile(value["capture"]["filename"], capture)
        assert value["elapsed_seconds"] >= 1.0
    finally:
        decoy.stop()
        for process in started:
            process.terminate()
            process.wait(10)
