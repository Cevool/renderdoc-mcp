"""Real stdio clients + two dedicated RenderDoc/Qt processes; save measurable evidence.

Run: uv run python scripts/smoke_test.py [--capture path/to/file.rdc].
Without a capture, deterministic PNG captures exercise real texture replay/export.
"""

import argparse
import asyncio
import json
import os
import struct
import subprocess
import sys
import time
import uuid
import zlib
from contextlib import AsyncExitStack
from pathlib import Path

import psutil
from fastmcp import Client
from fastmcp.client.transports import StdioTransport

ROOT = Path(__file__).resolve().parents[1]


def png(path, color):
    def chunk(name, data):
        return struct.pack(">I", len(data)) + name + data + struct.pack(">I", zlib.crc32(name + data))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 4, 4, 8, 6, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress((b"\0" + bytes(color) * 4) * 4)) + chunk(b"IEND", b""))


def data(result):
    return json.loads("".join(block.text for block in result.content if hasattr(block, "text")))


def hidden_process(command, env, log):
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = 0
    return subprocess.Popen(command, env=env, cwd=ROOT, stdout=log, stderr=log, startupinfo=startup)


def cpu_seconds(process):
    try:
        cpu = process.cpu_times()
        return cpu.user + cpu.system
    except psutil.NoSuchProcess:
        return 0


async def idle_sample(label, processes, seconds=10):
    before = {proc.pid: cpu_seconds(proc) for proc in processes}
    started = time.monotonic()
    await asyncio.sleep(seconds)
    elapsed = time.monotonic() - started
    per_process = [{"pid": proc.pid, "cpu_seconds": round(cpu_seconds(proc) - before[proc.pid], 6)}
                   for proc in processes]
    result = {"label": label, "wall_seconds": round(elapsed, 3), "processes": per_process,
              "percent_of_one_core": round(sum(p["cpu_seconds"] for p in per_process) / elapsed * 100, 3)}
    print(json.dumps(result), flush=True)
    return result


def mcp_processes():
    return [proc for proc in psutil.Process().children(recursive=True)
            if "mcp_server.server" in " ".join(proc.cmdline())]


async def run(args):
    output = ROOT / ".test-artifacts" / ("smoke-" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True)
    report = {"directory": str(output), "checks": [], "idle_cpu": []}
    processes, logs, stops = [], [], []
    ready = []
    try:
        for index in range(2):
            capture = Path(args.capture).resolve() if args.capture else output / (str(index) + ".png")
            if not args.capture:
                png(capture, (255, 0, 0, 255) if index == 0 else (0, 0, 255, 255))
            control = output / (str(index) + "-control.json")
            ready_path = output / (str(index) + "-ready.json")
            stop = output / (str(index) + "-stop")
            stops.append(stop)
            control.write_text(json.dumps({
                "repository": str(ROOT), "ipc_dir": str(output / "ipc"),
                "ready": str(ready_path), "stop": str(stop), "capture": str(capture),
            }), encoding="utf-8")
            log = (output / (str(index) + "-renderdoc.log")).open("w", encoding="utf-8")
            logs.append(log)
            processes.append(hidden_process(
                [args.renderdoc, "--python", str(ROOT / "scripts" / "renderdoc_test_host.py")],
                {**os.environ, "RENDERDOC_MCP_TEST_CONTROL": str(control)}, log))
            deadline = time.monotonic() + 30
            while not ready_path.exists():
                for error_path in (Path(str(ready_path) + ".error"), Path(str(control) + ".error")):
                    if error_path.exists():
                        raise RuntimeError(error_path.read_text("utf-8"))
                if processes[-1].poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("RenderDoc test host did not start; inspect " + str(output))
                await asyncio.sleep(0.1)
            ready.append(json.loads(ready_path.read_text("utf-8")))
        report["renderdoc_instances"] = ready
        environment = {"RENDERDOC_MCP_IPC_DIR": str(output / "ipc"),
                       "RENDERDOC_MCP_ARTIFACT_DIR": str(output / "artifacts")}

        def client():
            return Client(StdioTransport(sys.executable, ["-m", "mcp_server.server"], env=environment,
                                         cwd=str(ROOT), keep_alive=False))

        async with AsyncExitStack() as stack:
            first = await stack.enter_async_context(client())
            tools = await first.list_tools()
            assert {tool.name for tool in tools} == {"list_instances", "search_tools", "call_tool"}
            report["advertised_schema_bytes"] = len(json.dumps(
                [tool.model_dump(by_alias=True) for tool in tools]).encode("utf-8"))
            instances = data(await first.call_tool("list_instances", {}))["instances"]
            assert {item["instance_id"] for item in instances} == {item["instance_id"] for item in ready}
            report["checks"].append("real stdio handshake, exactly 3 tools, discover 2 RenderDoc instances")
            await asyncio.sleep(2)
            report["idle_cpu"].append(await idle_sample("one MCP client", mcp_processes(), args.idle_seconds))
            second = await stack.enter_async_context(client())
            await second.list_tools()
            await asyncio.sleep(2)
            report["idle_cpu"].append(await idle_sample("two MCP clients", mcp_processes(), args.idle_seconds))

            async def call(which, operation, **params):
                selected = ready[which]
                return data(await (first if which == 0 else second).call_tool("call_tool", {
                    "name": operation, "arguments": {"instance_id": selected["instance_id"], **params}
                }))

            results = await asyncio.gather(*[call(index % 2, "get_capture_status") for index in range(40)])
            for index, result in enumerate(results):
                assert result["loaded"]
                if not args.capture:
                    assert Path(result["filename"]).name == str(index % 2) + ".png"
            report["checks"].append("40 overlapping requests from 2 stdio clients routed to correct captures")
            for which, instance in enumerate(ready):
                summary = await call(which, "get_frame_summary")
                report.setdefault("summaries", []).append(summary)
                draw_page = await call(which, "get_draw_calls", limit=1)
                assert len(draw_page["actions"]) <= 1
                states = await call(which, "get_pipeline_states", event_ids=[instance["event_id"]])
                assert len(states["states"]) == 1
                difference = await call(which, "compare_pipeline_states",
                                        event_a=instance["event_id"], event_b=instance["event_id"])
                assert difference["equal"] and difference["changes"] == []
                if instance["textures"]:
                    rid = instance["textures"][0]
                    event = instance["event_id"]
                    info = await call(which, "get_texture_info", resource_id=rid)
                    raw = await call(which, "get_texture_data", event_id=event, resource_id=rid)
                    image = await call(which, "export_texture", event_id=event, resource_id=rid)
                    assert Path(raw["path"]).stat().st_size > 0 and "content_base64" not in raw
                    assert raw["format"] == info["format"] and raw["artifact_format"] == "bin"
                    assert Path(image["path"]).read_bytes().startswith(b"\x89PNG")
                    viewed = await first.call_tool("call_tool", {
                        "name": "view_image_artifact", "arguments": {"artifact_id": image["artifact_id"]}
                    })
                    assert any(block.type == "image" for block in viewed.content)
                    report.setdefault("texture_exports", []).append({"info": info, "raw": raw, "png": image})
            report["checks"].append("real RenderDoc texture metadata, raw export, PNG conversion and MCP image")
            report["checks"].append("real action pagination, pipeline batch and identical-event comparison")
            stops[0].touch()
            await asyncio.to_thread(processes[0].wait, 10)
            remaining = data(await second.call_tool("list_instances", {}))["instances"]
            assert len(remaining) == 1 and remaining[0]["instance_id"] == ready[1]["instance_id"]
            assert (await call(1, "get_capture_status"))["loaded"]
            report["checks"].append("closing one RenderDoc removes only its own instance; other client still works")
        report["success"] = True
    except Exception as exc:
        report["success"] = False
        report["error"] = repr(exc)
        raise
    finally:
        for stop in stops:
            stop.touch()
        for process in processes:
            try:
                await asyncio.to_thread(process.wait, 10)
            except subprocess.TimeoutExpired:
                process.terminate()
                await asyncio.to_thread(process.wait, 5)
        for log in logs:
            log.close()
        (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print("Report: " + str(output / "report.json"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--renderdoc", default=r"C:\Program Files\RenderDoc\qrenderdoc.exe")
    parser.add_argument("--capture")
    parser.add_argument("--idle-seconds", type=float, default=10)
    args = parser.parse_args()
    if args.idle_seconds <= 0:
        parser.error("--idle-seconds must be positive")
    asyncio.run(run(args))
