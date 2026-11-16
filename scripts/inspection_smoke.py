"""Real-GPU check of buffer element reads/diffs, vertex inputs, mesh export, batch debugging,
debug messages and state scans.

Run: uv run python scripts/inspection_smoke.py --capture path/to/file.rdc [--pixel-event N]
Uses one dedicated hidden RenderDoc process (scripts/renderdoc_test_host.py) and a private
IPC directory; the installed extension and user configuration are not involved.
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

from fastmcp import Client
from fastmcp.client.transports import StdioTransport

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smoke_test import ROOT, data, hidden_process  # noqa: E402


def close(a, b, tolerance=1e-4):
    return all(isinstance(x, (int, float)) and isinstance(y, (int, float))
               and abs(x - y) <= tolerance * max(1.0, abs(x), abs(y)) for x, y in zip(a, b))


async def run(args):
    output = ROOT / ".test-artifacts" / ("inspection-" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True)
    report = {"directory": str(output), "capture": args.capture, "checks": [], "timings": {}}
    control, ready_path, stop = output / "control.json", output / "ready.json", output / "stop"
    control.write_text(json.dumps({"repository": str(ROOT), "ipc_dir": str(output / "ipc"),
                                   "ready": str(ready_path), "stop": str(stop),
                                   "api_validation": args.api_validation,
                                   "capture": str(Path(args.capture).resolve())}), encoding="utf-8")
    log = (output / "renderdoc.log").open("w", encoding="utf-8")
    process = hidden_process([args.renderdoc, "--python", str(ROOT / "scripts" / "renderdoc_test_host.py")],
                             {**os.environ, "RENDERDOC_MCP_TEST_CONTROL": str(control)}, log)
    try:
        deadline = time.monotonic() + args.startup_timeout
        while not ready_path.exists():
            for error in (Path(str(ready_path) + ".error"), Path(str(control) + ".error")):
                if error.exists():
                    raise RuntimeError(error.read_text("utf-8"))
            if process.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("RenderDoc test host did not start; inspect " + str(output))
            await asyncio.sleep(0.2)
        while True:
            try:
                ready = json.loads(ready_path.read_text("utf-8"))
                break
            except ValueError:
                await asyncio.sleep(0.2)
        env = {"RENDERDOC_MCP_IPC_DIR": str(output / "ipc"),
               "RENDERDOC_MCP_ARTIFACT_DIR": str(output / "artifacts")}
        transport = StdioTransport(sys.executable, ["-m", "mcp_server.server"], env=env,
                                   cwd=str(ROOT), keep_alive=False)
        async with Client(transport) as client:
            async def call(name, **arguments):
                started = time.monotonic()
                value = data(await client.call_tool("call_tool", {"name": name, "arguments": {
                    "instance_id": ready["instance_id"], **arguments}}))
                report["timings"].setdefault(name, []).append(round(time.monotonic() - started, 2))
                return json.loads(Path(value["path"]).read_text("utf-8")) if value.get("stored") else value

            records, offset = [], 0
            while True:
                page = await call("get_draw_calls", only_actions=True, offset=offset, limit=200)
                records.extend(page["actions"])
                if page["next_offset"] is None:
                    break
                offset = page["next_offset"]
            draws = [a for a in records if "Indexed" in a["flags"] and a["num_indices"] >= 300]
            dispatches = [a for a in records if "Dispatch" in a["flags"]]
            draw = args.draw_event
            if draw is None:
                # A draw whose first post-VS vertex is finite and in front of the camera.
                middle = len(draws) // 2
                for candidate in sorted(draws, key=lambda a: abs(draws.index(a) - middle))[:40]:
                    rows = (await call("get_post_vs_data", event_id=candidate["event_id"], limit=3))["vertices"]
                    if rows and all(r.get("finite") and r["position"][3] > 0 for r in rows):
                        draw = candidate["event_id"]
                        break
            assert draw, "No indexed draw with usable post-VS output"
            report["draw_event"] = draw

            # --- Vertex inputs, cross-checked against a direct typed read of each buffer.
            inputs = await call("get_vertex_inputs", event_id=draw, limit=8)
            layout = inputs["layout"]
            assert inputs["rows"] and layout["indexed"], inputs
            row = inputs["rows"][3]
            ib = layout["index_buffer"]
            index_read = await call("read_buffer_elements", event_id=draw, resource_id=ib["resource_id"],
                                    byte_offset=ib["byte_offset"], start=3, count=1)
            assert index_read["layout_source"] == "index buffer", index_read
            assert index_read["elements"][0]["values"]["index"] == row["index"], (index_read, row)
            position = next(a for a in layout["attributes"] if a["used"] and a["decodable"] and not a["per_instance"])
            vb = next(b for b in layout["vertex_buffers"] if b["slot"] == position["slot"])
            probe = await call("read_buffer_elements", event_id=draw, resource_id=vb["resource_id"],
                               element_stride=4, count=1)
            slot = next(i for i, b in enumerate(probe["bindings"])
                        if b["source"] == "vertex buffer slot %d" % position["slot"])
            vertex_read = await call("read_buffer_elements", event_id=draw, resource_id=vb["resource_id"],
                                     start=row["vertex"], count=1, binding=slot)
            assert vertex_read["layout_source"].startswith("vertex buffer"), vertex_read
            assert vertex_read["elements"][0]["values"][position["name"]] == row["values"][position["name"]], (
                vertex_read["elements"][0], row)
            report["vertex_inputs"] = {"topology": layout["topology"], "attributes": [
                (a["name"], a["format"], a["slot"]) for a in layout["attributes"]], "row3": row}
            report["checks"].append("vertex inputs agree with direct index/vertex buffer element reads")

            # --- Mesh export from VS input and VS output, cross-checked with get_post_vs_data.
            mesh = await call("export_mesh", event_id=draw, stage="vsin", format="obj")
            text = Path(mesh["path"]).read_text("utf-8")
            faces = sum(1 for line in text.splitlines() if line.startswith("f "))
            verts = sum(1 for line in text.splitlines() if line.startswith("v "))
            assert faces == mesh["primitives"] and verts == mesh["vertices"] > 0, mesh
            if layout["topology"] == "TriangleList":
                assert faces == layout["num_indices"] // 3, (faces, layout["num_indices"])
            first = next(line for line in text.splitlines() if line.startswith("v ")).split()[1:]
            # The first OBJ vertex is the first vertex referenced by the index stream.
            assert close([float(v) for v in first],
                         inputs["rows"][0]["values"][mesh["attributes"]["position"]][:3]), first
            report["mesh_vsin"] = {k: mesh[k] for k in ("vertices", "primitives", "attributes",
                                                        "degenerate_primitives", "nonfinite_vertices_zeroed")}
            post = await call("export_mesh", event_id=draw, stage="vsout", format="ply")
            ply = Path(post["path"]).read_text("utf-8").split("end_header\n")
            assert f"element vertex {post['vertices']}" in ply[0] and f"element face {post['primitives']}" in ply[0]
            pvs = await call("get_post_vs_data", event_id=draw, limit=50)
            order = []  # PLY vertices follow first use in the index stream.
            for record in pvs["vertices"]:
                if record.get("vertex_index") is not None and record["vertex_index"] not in order:
                    order.append(record["vertex_index"])
            sample = next(r for r in pvs["vertices"] if r.get("finite") and r["position"][3] > 0)
            clip = sample["position"]
            expected = [clip[0] / clip[3], clip[1] / clip[3], clip[2] / clip[3]]
            line = ply[1].splitlines()[order.index(sample["vertex_index"])]
            assert close([float(v) for v in line.split()], expected, 1e-5), (line, clip)
            report["mesh_vsout_first_rows"] = [r["position"] for r in pvs["vertices"][:3]]
            report["mesh_vsout"] = {k: post[k] for k in ("vertices", "primitives", "space",
                                                         "vertices_behind_camera", "nonfinite_vertices_zeroed")}
            report["checks"].append("OBJ (VS in) and PLY (VS out, NDC) exports match vertex/post-VS reads")

            # --- Compute: dispatch info, auto-layout UAV read, before/after diff, batch debug.
            target = None
            for action in dispatches[:80]:
                state = await call("get_pipeline_state", event_id=action["event_id"], fields=["shaders"])
                compute = state["shaders"].get("ShaderStage.Compute", {})
                uav = next((u for u in compute.get("uavs", []) if u.get("type") == "buffer"), None)
                if uav:
                    target = (action["event_id"], uav["resource_id"])
                    break
            assert target, "No dispatch with a buffer UAV"
            dispatch, uav = target
            details = await call("get_draw_call_details", event_id=dispatch)
            assert details["dispatch_groups"] and details["thread_group_size"], details
            elements = await call("read_buffer_elements", event_id=dispatch, resource_id=uav, count=4)
            assert elements["elements"] and elements["layout"], elements
            diff = await call("compare_buffer_contents", event_id=dispatch, resource_id=uav,
                              element_stride=elements["element_stride"], offset=elements["byte_offset"],
                              length=min(elements["total_elements"] * elements["element_stride"], 1 << 22))
            assert diff["before_sha256"] and diff["equal"] == (diff["changed_elements"] == 0), diff
            if not diff["equal"]:
                assert diff["before_sha256"] != diff["after_sha256"] and diff["changes"]
            report["compute"] = {"event": dispatch, "uav": uav, "details": {
                k: details[k] for k in ("dispatch_groups", "thread_group_size", "dispatch_threads")},
                "layout": elements["layout"], "layout_source": elements["layout_source"],
                "first_element": elements["elements"][0], "diff": {
                    k: diff[k] for k in ("compared_bytes", "changed_elements", "changed_ranges", "equal")}}
            size = details["thread_group_size"]
            batch = await call("debug_shader_batch", event_id=dispatch, stage="compute", invocations=[
                {"global": [0, 0, 0]}, {"group": [0, 0, 0], "thread": [min(1, size[0] - 1), 0, 0]}])
            if batch.get("status") != "unsupported":
                assert all(r["status"] in ("ok", "partial") for r in batch["invocations"]), batch
                assert batch["invocations"][0]["final_values"], batch
            report["compute"]["batch_debug"] = [{k: r.get(k) for k in ("status", "steps", "finished")}
                                                for r in batch["invocations"]]
            report["checks"].append("dispatch info, reflected UAV layout, before/after diff, compute batch debug")

            if args.pixel_event:
                single = await call("debug_shader", event_id=args.pixel_event, stage="pixel",
                                    invocation={"x": 64, "y": 64}, max_steps=10000)
                pixel = await call("debug_shader_batch", event_id=args.pixel_event, stage="pixel",
                                   invocations=[{"x": 64, "y": 64}, {"x": 200, "y": 120}],
                                   variables=["o0", "SV_Target"])
                assert pixel["invocations"][0]["status"] == "ok", pixel
                assert pixel["invocations"][0]["final_values"], pixel  # DXBC o0 or DXIL _OUT.SV_Target
                # The last o0 change in the stepped trace equals the batch's final o0.
                last = None
                for step in single["states"]:
                    for change in step["changes"]:
                        if change["after"]["name"] in [v["name"] for v in pixel["invocations"][0]["final_values"]]:
                            last = change["after"]
                if last is not None and single.get("finished"):
                    final = next(v for v in pixel["invocations"][0]["final_values"] if v["name"] == last["name"])
                    assert final.get("value") == last.get("value"), (final, last)
                report["pixel_batch"] = [{k: r.get(k) for k in ("status", "steps", "final_values")}
                                         for r in pixel["invocations"]]
                report["checks"].append("pixel batch debug final outputs agree with a stepped trace")

            messages = await call("get_debug_messages", limit=5)
            report["debug_messages"] = {k: messages[k] for k in ("total", "total_unfiltered", "by_severity")}
            report["debug_messages"]["first"] = messages["messages"][:3]
            report["checks"].append("debug messages queried")

            low = max(1, draw - 400)
            scan = await call("scan_state_changes", event_id_min=low, event_id_max=draw, max_events=60)
            assert scan["baseline"] and scan["scanned"] > 1, scan
            filtered = await call("scan_state_changes", event_id_min=low, event_id_max=draw, max_events=60,
                                  ignore_paths=["/constant_buffers/", "/input_assembly/"])
            assert sum(e["change_count"] for e in filtered["events"]) <= sum(e["change_count"] for e in scan["events"])
            report["scan"] = {"range": [low, draw], "scanned": scan["scanned"], "events_with_changes":
                              len(scan["events"]), "unchanged": scan["unchanged_events"],
                              "truncated": scan["truncated"], "next": scan["next_event_id_min"],
                              "sections": [e["changed_sections"] for e in scan["events"][:10]]}
            report["checks"].append("state change scan with and without ignored paths")

            status = await call("get_capture_status")
            assert status["event_id"] == ready["event_id"], status
            report["checks"].append("UI-selected event restored after all operations")
        report["success"] = True
    except Exception as exc:
        report["success"] = False
        report["error"] = repr(exc)
        raise
    finally:
        stop.touch()
        try:
            await asyncio.to_thread(process.wait, 30)
        except subprocess.TimeoutExpired:
            process.terminate()
        log.close()
        (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print("Report: " + str(output / "report.json"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--renderdoc", default=r"C:\Program Files\RenderDoc\qrenderdoc.exe")
    parser.add_argument("--capture", required=True)
    parser.add_argument("--draw-event", type=int, help="An indexed draw (default: middle indexed draw)")
    parser.add_argument("--pixel-event", type=int, help="A draw with a pixel shader covering (64,64)")
    parser.add_argument("--api-validation", action="store_true", help="Replay with API validation")
    parser.add_argument("--startup-timeout", type=float, default=300)
    asyncio.run(run(parser.parse_args()))
