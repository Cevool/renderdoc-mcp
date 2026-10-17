"""Real stdio clients + two dedicated RenderDoc/Qt processes; save measurable evidence.

Run: uv run python scripts/smoke_test.py [--capture path/to/file.rdc].
Without a capture, deterministic PNG captures exercise real texture replay/export.
"""

import argparse
import asyncio
import base64
import hashlib
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
    ui_config = Path(os.environ["APPDATA"]) / "qrenderdoc" / "UI.config"
    config_before = ui_config.read_bytes() if ui_config.exists() else None
    processes, logs, stops = [], [], []
    ready = []
    try:
        for index in range(2):
            selected_capture = args.capture if index == 0 else (args.other_capture or (
                args.capture if not args.deep_analysis else None))
            use_capture = bool(selected_capture)
            capture = Path(selected_capture).resolve() if use_capture else output / (str(index) + ".png")
            if not use_capture:
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
            deadline = time.monotonic() + args.startup_timeout
            while not ready_path.exists():
                for error_path in (Path(str(ready_path) + ".error"), Path(str(control) + ".error")):
                    if error_path.exists():
                        raise RuntimeError(error_path.read_text("utf-8"))
                if processes[-1].poll() is not None or time.monotonic() > deadline:
                    phase_path = Path(str(ready_path) + ".phase.json")
                    phase = phase_path.read_text("utf-8") if phase_path.exists() else "before Python script"
                    raise RuntimeError("RenderDoc test host did not start (%s); inspect %s" % (phase, output))
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
            assert {tool.name for tool in tools} == {
                "list_instances", "browse_tools", "get_tool_schema", "search_tools", "call_tool"}
            report["advertised_schema_bytes"] = len(json.dumps(
                [tool.model_dump(by_alias=True) for tool in tools]).encode("utf-8"))
            instances = data(await first.call_tool("list_instances", {}))["instances"]
            assert {item["instance_id"] for item in instances} == {item["instance_id"] for item in ready}
            report["checks"].append("real stdio handshake, 5 discovery tools, discover 2 RenderDoc instances")
            report["tool_directory"] = data(await first.call_tool("browse_tools", {}))
            page = data(await first.call_tool("search_tools", {"pattern": ".*", "limit": 3}))
            assert page["total_matches"] > page["returned"] and page["truncated"]
            schema = data(await first.call_tool("get_tool_schema", {"name": "get_post_vs_data"}))
            assert "event_id" in schema["inputSchema"]["required"]
            await asyncio.sleep(2)
            report["idle_cpu"].append(await idle_sample("one MCP client", mcp_processes(), args.idle_seconds))
            second = await stack.enter_async_context(client())
            await second.list_tools()
            await asyncio.sleep(2)
            report["idle_cpu"].append(await idle_sample("two MCP clients", mcp_processes(), args.idle_seconds))

            async def call(which, operation, **params):
                selected = ready[which]
                value = data(await (first if which == 0 else second).call_tool("call_tool", {
                    "name": operation, "arguments": {"instance_id": selected["instance_id"], **params}
                }))
                return json.loads(Path(value["path"]).read_text("utf-8")) if value.get("stored") else value

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
                report.setdefault("pipeline_comparisons", []).append(difference)
                assert difference["changes"] == [], difference
                assert difference["equal"] if difference["complete"] else difference["equal"] is None
                if args.deep_analysis and which == 0:
                    if args.event_id is None:
                        draws = await call(which, "get_draw_calls", only_actions=True,
                                           flags_filter=["Drawcall"], limit=200)
                        draw = next(item for item in draws["actions"] if item["num_indices"] > 0)
                        event = draw["event_id"]
                    else:
                        event = args.event_id
                    details = await call(which, "get_draw_call_details", event_id=event)
                    pipeline = await call(which, "get_pipeline_state", event_id=event)
                    native = await call(which, "get_native_pipeline_state", event_id=event)
                    assert native["status"] == "ok", native
                    post = await call(which, "get_post_vs_data", event_id=event, limit=3, include_bytes=True)
                    assert post["vertices"] and all(v.get("finite") for v in post["vertices"]), post
                    events = await call(which, "get_api_events", event_id_min=max(0, event - 100),
                                        event_id_max=event, include_parameters=True, limit=200)
                    assert events.get("events"), events
                    deep = {"draw": details, "native": native, "post_vs": post, "api_events": events,
                            "constant_buffers": []}
                    report["deep_analysis"] = deep  # Preserve successful evidence if a later check fails.
                    for shader_stage, stage_value in pipeline["shaders"].items():
                        stage = shader_stage.rsplit(".", 1)[-1].lower()
                        for block_index, block in enumerate(stage_value["constant_buffers"]):
                            assert "error" not in block, block
                            constant = await call(which, "get_constant_buffer", event_id=event,
                                                  stage=stage, block_index=block_index)
                            assert constant["complete"] and "declarations" in constant, constant
                            if constant.get("raw_read"):
                                raw = constant["raw_read"]
                                exported = await call(which, raw["tool"], **raw["arguments"])
                                exported["sha256"] = hashlib.sha256(Path(exported["path"]).read_bytes()).hexdigest()
                                constant["raw_export"] = exported
                            deep["constant_buffers"].append(constant)
                    index = details["vertex_offset"]
                    if "Indexed" in details["flags"]:
                        ib = pipeline["input_assembly"]["index_buffer"]
                        raw = await call(which, "get_buffer_contents", event_id=event,
                            resource_id=ib["resource_id"], offset=ib["offset"] + details["index_offset"] * ib["stride"],
                            length=ib["stride"], output="inline")
                        index = int.from_bytes(base64.b64decode(raw["content_base64"]), "little") + details["base_vertex"]
                    debug = await call(which, "debug_shader", event_id=event, stage="vertex",
                                       invocation={"vertex": 0, "index": index}, max_steps=50)
                    if debug.get("stored"):
                        debug = json.loads(Path(debug["path"]).read_text("utf-8"))
                    assert debug.get("inputs") and debug.get("states") and debug["trace_released"], debug
                    deep["debug_vertex"] = debug
                    if details["outputs"]:
                        rid = details["outputs"][0]["resource_id"]
                        usage = await call(which, "get_resource_usage", resource_id=rid)
                        assert usage["usages"], usage
                        info = await call(which, "get_texture_info", resource_id=rid)
                        x, y = args.pixel or (info["width"] // 2, info["height"] // 2)
                        history = await call(which, "get_pixel_history", event_id=event, resource_id=rid,
                                             x=x, y=y, limit=200 if args.pixel else 50)
                        assert history["status"] in ("ok", "empty", "partial"), history
                        deep.update(resource_usage=usage, pixel_history=history)
                        if args.pixel_primitive is not None:
                            assert any(item["eventId"] == event and item["primitiveID"] == args.pixel_primitive
                                       for item in history["history"]), history
                        if args.pixel:
                            invocation = {"x": x, "y": y}
                            if args.pixel_primitive is not None:
                                invocation["primitive"] = args.pixel_primitive
                            pixel_debug = await call(which, "debug_shader", event_id=event, stage="pixel",
                                                     invocation=invocation, max_steps=50)
                            assert pixel_debug.get("inputs") and pixel_debug.get("states") and pixel_debug["trace_released"], pixel_debug
                            deep["debug_pixel"] = pixel_debug
                    dispatches = await call(which, "get_draw_calls", only_actions=True,
                                            flags_filter=["Dispatch"], limit=1)
                    if dispatches["actions"]:
                        compute = await call(which, "debug_shader", event_id=dispatches["actions"][0]["event_id"],
                            stage="compute", invocation={"group": [0, 0, 0], "thread": [0, 0, 0]}, max_steps=50)
                        assert compute.get("states") and compute["trace_released"], compute
                        deep["debug_compute"] = compute
                    assert (await call(which, "get_capture_status"))["event_id"] == instance["event_id"]
                    report["deep_analysis"] = deep
                    report["checks"].append("real native state, Post-VS, API parameters, constants/bytes, shader debug, usage, pixel history; selected event restored")
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
            comparison = data(await first.call_tool("call_tool", {"name": "compare_capture_states", "arguments": {
                "instance_id": ready[0]["instance_id"], "event_id": args.event_id if args.event_id is not None else ready[0]["event_id"],
                "other_instance_id": ready[1]["instance_id"], "other_event_id": args.other_event_id if args.other_event_id is not None else ready[1]["event_id"],
                "fields": ["api", "render_targets"], "include_native": bool(args.other_capture)}}))
            if comparison.get("stored"):
                comparison = json.loads(Path(comparison["path"]).read_text("utf-8"))
            assert comparison["before"]["instance_id"] == ready[0]["instance_id"]
            assert comparison["after"]["instance_id"] == ready[1]["instance_id"]
            report["cross_capture_comparison"] = comparison
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
        config_after = ui_config.read_bytes() if ui_config.exists() else None
        report["user_ui_config_unchanged"] = config_after == config_before
        report["ui_config_sha256"] = (hashlib.sha256(config_after).hexdigest()
                                      if config_after is not None else None)
        if config_after != config_before:
            report["success"] = False
            report["error"] = "User UI.config changed during the test; inspect host persistence or concurrent UI edits"
        (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print("Report: " + str(output / "report.json"), flush=True)
        if config_after != config_before:
            raise RuntimeError(report["error"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--renderdoc", default=r"C:\Program Files\RenderDoc\qrenderdoc.exe")
    parser.add_argument("--capture")
    parser.add_argument("--other-capture", help="Optional second capture for explicit cross-capture comparison")
    parser.add_argument("--event-id", type=int, help="Select a specific draw instead of the first")
    parser.add_argument("--other-event-id", type=int, help="Explicitly paired event in the second capture")
    parser.add_argument("--pixel", type=int, nargs=2, metavar=("X", "Y"), help="Also debug this top-left pixel")
    parser.add_argument("--pixel-primitive", type=int, help="Select a specific fragment for pixel debugging")
    parser.add_argument("--deep-analysis", action="store_true", help="Exercise deep queries on the first real draw")
    parser.add_argument("--idle-seconds", type=float, default=10)
    parser.add_argument("--startup-timeout", type=float, default=120,
                        help="Per-host startup/replay budget in seconds")
    args = parser.parse_args()
    if args.idle_seconds <= 0:
        parser.error("--idle-seconds must be positive")
    if args.startup_timeout <= 0:
        parser.error("--startup-timeout must be positive")
    if args.deep_analysis and not args.capture:
        parser.error("--deep-analysis requires --capture with real draws")
    if any(value is not None and value < 0 for value in (args.event_id, args.other_event_id, args.pixel_primitive)) or (
            args.pixel and any(value < 0 for value in args.pixel)):
        parser.error("Event and pixel coordinates must be nonnegative")
    if (args.event_id is not None or args.pixel or args.pixel_primitive is not None) and not args.deep_analysis:
        parser.error("Draw/pixel selection requires --deep-analysis")
    asyncio.run(run(args))
