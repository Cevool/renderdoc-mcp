"""Compare connected-but-idle MCP subprocesses, including original lockfile baseline."""

import argparse
import asyncio
import json
import sys
import time
from contextlib import AsyncExitStack
from pathlib import Path

from fastmcp import Client
from fastmcp.client.transports import StdioTransport

from smoke_test import ROOT, idle_sample, mcp_processes


async def measure(label, python, project, seconds, output):
    results = []
    async with AsyncExitStack() as stack:
        for count in (1, 2):
            transport = StdioTransport(str(python), ["-m", "mcp_server.server"], cwd=str(project),
                                       keep_alive=False, log_file=output / (label + str(count) + ".log"))
            client = await stack.enter_async_context(Client(transport))
            tools = await client.list_tools()
            await asyncio.sleep(2)
            result = await idle_sample(label + " / " + str(count) + " clients", mcp_processes(), seconds)
            result.update({"tool_count": len(tools), "catalog_bytes": len(json.dumps(
                [tool.model_dump(by_alias=True) for tool in tools]).encode("utf-8"))})
            results.append(result)
    return results


async def run(args):
    output = ROOT / ".test-artifacts" / ("idle-" + time.strftime("%Y%m%d-%H%M%S"))
    output.mkdir(parents=True)
    report = {"python": sys.version, "samples": []}
    if args.baseline:
        baseline = Path(args.baseline).resolve()
        report["samples"].extend(await measure("FastMCP-2.14.2", baseline / ".venv/Scripts/python.exe",
                                               baseline, args.seconds, output))
    report["samples"].extend(await measure("FastMCP-4.1.0", sys.executable, ROOT, args.seconds, output))
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Report: " + str(output / "report.json"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline")
    parser.add_argument("--seconds", type=float, default=10)
    asyncio.run(run(parser.parse_args()))
