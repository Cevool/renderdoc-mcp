"""Real-GPU check of replay edits, texture evidence and resource bundles.

Run: uv run python scripts/workbench_smoke.py --capture path/to/file.rdc --event-id N
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

MAGENTA = """float4 main() : SV_Target0 { return float4(1.0, 0.0, 1.0, 1.0); }\n"""


async def run(args):
    output = ROOT / ".test-artifacts" / ("workbench-" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True)
    report = {"directory": str(output), "checks": []}
    control, ready_path, stop = output / "control.json", output / "ready.json", output / "stop"
    control.write_text(json.dumps({"repository": str(ROOT), "ipc_dir": str(output / "ipc"),
                                   "ready": str(ready_path), "stop": str(stop),
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
        while True:  # The host may still be writing a large texture list.
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
                if name not in ("compare_texture_evidence", "validate_resource_bundle"):
                    arguments = {"instance_id": ready["instance_id"], **arguments}
                value = data(await client.call_tool("call_tool", {"name": name, "arguments": arguments}))
                return json.loads(Path(value["path"]).read_text("utf-8")) if value.get("stored") else value

            event = args.event_id
            details = await call("get_draw_call_details", event_id=event)
            target = details["outputs"][0]["resource_id"]
            info = await call("get_texture_info", resource_id=target)
            report.update(event=event, target=target, texture=info)

            before = await call("capture_texture_evidence", event_id=event, resource_id=target,
                                grid=8, histogram_range=[0.0, 1.0])
            assert before["complete"] and len(before["samples"]) == 64, before
            assert before["capture"]["edit_generation"] == 0 and len(before["histogram"]["buckets"]) > 0
            report["checks"].append("baseline evidence: 64 samples, min/max, histogram, raw hash")

            failed = await call("replace_shader", event_id=event, stage="pixel",
                                source="float4 main() : SV_Target0 { return undefined_symbol; }",
                                entry_point="main")
            assert failed["status"] == "compile_failed" and not failed["applied"], failed
            assert failed["compiler_output"], failed
            report["checks"].append("compile failure reports diagnostics and applies nothing")

            replaced = await call("replace_shader", event_id=event, stage="pixel", source=MAGENTA,
                                  entry_point="main")
            assert replaced["status"] == "ok" and replaced["applied"], replaced
            report["replace"] = replaced
            listed = await call("list_shader_replacements")
            assert [r["original_shader"] for r in listed["replacements"]] == [replaced["original_shader"]]

            after = await call("capture_texture_evidence", event_id=event, resource_id=target,
                               grid=8, histogram_range=[0.0, 1.0])
            assert after["capture"]["edit_generation"] == 1 and after["capture"]["shader_replacements"]
            diff = await call("compare_texture_evidence", before_artifact_id=before["artifact_id"],
                              after_artifact_id=after["artifact_id"])
            assert diff["changed_samples"] > 0 and diff["raw_equal"] in (False, None), diff
            magenta = [s for s in after["samples"] if s["value"][:3] == [1.0, 0.0, 1.0]]
            report["magenta_samples"] = len(magenta)
            assert magenta, after["samples"][:4]
            report["checks"].append("replacement changes the event output; evidence records generation 1")

            outputs = await call("fingerprint_event_outputs", event_id=event, grid=4)
            assert outputs["outputs"], outputs
            image = await call("export_texture", event_id=event, resource_id=target,
                               range_min=0.0, range_max=2.0, channel=0)
            assert Path(image["path"]).read_bytes().startswith(b"\x89PNG")
            report["checks"].append("output fingerprint and ranged single-channel PNG while edited")

            swapped = await call("replace_shader", event_id=event, stage="pixel", entry_point="main",
                                 source=MAGENTA.replace("1.0, 0.0, 1.0", "0.0, 1.0, 0.0"))
            assert swapped["replaced_previous_mcp_edit"] and swapped["generation"] == 2, swapped
            reverted = await call("revert_shader")
            assert [r["original_shader"] for r in reverted["reverted"]] == [replaced["original_shader"]]
            assert not reverted["remaining"]
            restored = await call("capture_texture_evidence", event_id=event, resource_id=target,
                                  grid=8, histogram_range=[0.0, 1.0])
            back = await call("compare_texture_evidence", before_artifact_id=before["artifact_id"],
                              after_artifact_id=restored["artifact_id"])
            assert back["sample_equal"] and back["raw_equal"] in (True, None), back
            report["restored_comparison"] = {k: back[k] for k in ("sample_equal", "raw_equal", "raw_comparison_available")}
            report["checks"].append("swap candidate, revert all, output returns to baseline")

            light = await call("export_resource_bundle", event_id=event, resources="none")
            assert light["validation"]["valid"] and light["file_counts"].get("shader_binary"), light
            assert light["file_counts"].get("shader_disassembly") and light["file_counts"].get("constants_decoded")
            full = await call("export_resource_bundle", event_id=event, byte_limit=512 * 1024 * 1024)
            assert full["validation"]["valid"] and full["file_counts"].get("texture"), full
            again = await call("validate_resource_bundle", bundle=full["bundle"])
            assert again["valid"]
            report["bundles"] = {"shaders_only": {k: light[k] for k in ("file_counts", "data_bytes", "omission_count")},
                                 "full": {k: full[k] for k in ("file_counts", "data_bytes", "omission_count", "complete")}}
            report["checks"].append("shader/constant-only and full bundles validate")
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
    parser.add_argument("--event-id", type=int, required=True, help="A draw with a pixel shader and color output")
    parser.add_argument("--startup-timeout", type=float, default=300)
    asyncio.run(run(parser.parse_args()))
