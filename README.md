# RenderDoc MCP Server

Give AI clients structured access to running RenderDoc instances. Version 2.0 uses FastMCP 4.1, on-demand tool discovery, and isolated request queues to support multiple RenderDoc instances and MCP clients.

## Setup and migration

The server requires Python 3.10+. The recommended setup uses [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage an isolated project environment. Development and integration tests use Windows, Python 3.14.4, uv 0.11.12, and RenderDoc 1.46 with its embedded Python 3.8. The extension retains Python 3.6 syntax compatibility and uses only the standard library and RenderDoc's bundled modules; FastMCP is installed in the external server environment.

Run from the repository root. These commands create the project virtual environment and install its locked dependencies:

```powershell
uv sync --locked
uv run --locked python scripts/install_extension.py
```

The installer copies the extension to `%APPDATA%\qrenderdoc\extensions\renderdoc_mcp_bridge`. Enable **RenderDoc MCP Bridge** under **Tools > Manage Extensions** in RenderDoc.

**Upgrading from 1.x requires updating both the server and the extension.** Reload the extension or restart RenderDoc, then restart the MCP connection. Save captures you need before restarting existing windows. The new protocol does not consume the old shared queue.

The repository's `.mcp.json` starts the current project through uv and expects the client to use the repository as its working directory. For other clients, configure an equivalent command with an absolute repository path:

```json
{
  "mcpServers": {
    "renderdoc": {
      "command": "uv",
      "args": [
        "run", "--directory", "C:/path/to/renderdoc-mcp", "--locked", "renderdoc-mcp"
      ]
    }
  }
}
```

Replace `C:/path/to/renderdoc-mcp` with your checkout path. The `uv` executable must be available to the MCP client's process; use its full executable path if necessary. If you keep using a globally installed `renderdoc-mcp` command, update that installation separately; updating the project environment does not replace it.

Alternatively, install the server into an existing Python virtual environment with `python -m pip install .`, run `python scripts/install_extension.py` using that environment, and configure the client to launch that environment's `renderdoc-mcp` executable.

## Three advertised tools

| Tool | Purpose |
| --- | --- |
| `list_instances` | Discover live instances, including instance ID, PID, version, and capture filename |
| `search_tools` | Find operations by regex; return full parameter schemas for up to three matches |
| `call_tool` | Execute an operation by name with its arguments |

Example workflow:

```text
list_instances()
search_tools(pattern="pipeline|compare")
call_tool(name="get_pipeline_state", arguments={"instance_id":"<discovered ID>","event_id":123})
```

Search matches names, descriptions, and parameters. Useful patterns include `shader`, `frame_summary`, `texture`, and `get_pipeline_state\b`. Every RenderDoc operation requires an explicit `instance_id`; there is no global selection for clients to overwrite. Closing an instance or reloading its extension invalidates its ID, so discover it again.

The original capture management, action queries, shader/resource searches, GPU timings, pipeline queries, and texture/buffer operations remain available through discovery. Additional operations include:

- `get_pipeline_states`: inspect up to 32 events in a serialized batch.
- `compare_pipeline_states`: return changed values using JSON Pointer paths. Incomplete queries return `complete=false` and `equal=null` rather than claiming equality.
- `export_texture`: convert a texture mip/slice to PNG using RenderDoc.
- `view_image_artifact`: return an exported PNG as an MCP image.
- `read_artifact`: read bounded chunks of a saved JSON result.

## Result size and replay state

- Actions, search results, timings, and capture lists default to 50 records per page, with a maximum of 200. Continue using `offset` and `next_offset`.
- Action records form a flat list with `parent_event_id`. Marker, event-range, and flag filters apply before pagination. The optional `unity_game_rendering` preset excludes common editor UI markers; it is never applied automatically.
- Action and pipeline queries support field selection. JSON results larger than 24,000 UTF-8 bytes are saved in full as artifacts, with an ID, path, and retrieval hint returned to the client.
- Buffer reads export 4096 bytes by default. Set `length=0` to export the remaining bytes from the requested offset. Explicit `output="inline"` reads return at most 4096 bytes of Base64 data.
- Raw textures are exported directly to binary files, avoiding large Base64 payloads in IPC and model context. `format` describes the texture format; `artifact_format` describes the output file type.
- Buffer reads, raw texture exports, and PNG exports require an `event_id`. Replay queries restore the UI-selected event on success and failure.
- Artifacts are local files retained for inspection. Archive or remove them when appropriate. PNGs can be viewed through MCP without direct client access to the server's filesystem.

## IPC and idle behavior

```text
MCP client -> stdio server -> instances/<instance_id>/requests/<request_id>.json
                                           |
                                  Qt file notification
                                           |
                                    target RenderDoc
                                           |
                               responses/<request_id>.json
```

Each instance owns its manifest, requests, and responses. Files are published atomically with retries for transient Windows sharing violations. Both sides validate protocol, instance, and request IDs. Requests execute serially within an instance; separate instances operate independently.

The extension uses `QFileSystemWatcher`, with five-second maintenance for discovery metadata, missed notifications, and expired queue files. The MCP process waits for responses only during active calls. It does not start an idle polling loop or a background task worker. FastMCP's optional tasks extension is neither installed nor enabled.

Expired queued requests are discarded, and late responses cannot satisfy later requests. **An operation that has already started cannot be forcibly cancelled.** After an `open_capture` timeout, check capture status before deciding whether to retry.

Optional environment variables:

| Variable | Default | Configure in |
| --- | --- | --- |
| `RENDERDOC_MCP_IPC_DIR` | `renderdoc_mcp` under the system temporary directory | Both RenderDoc and the MCP server; values must match |
| `RENDERDOC_MCP_TIMEOUT` | `30` seconds | MCP server |
| `RENDERDOC_MCP_ARTIFACT_DIR` | `artifacts` under the IPC root | MCP server |

The obsolete `RENDERDOC_MCP_HOST` and `RENDERDOC_MCP_PORT` settings have been removed. File IPC does not listen on a TCP port. **Tools > MCP Bridge > Status** displays the instance ID and actual IPC directory.

## Testing

```powershell
uv sync --locked --group dev
uv run --locked pytest -q
uv run --locked ruff check mcp_server renderdoc_extension scripts tests
uv run --locked python scripts/smoke_test.py
```

Automated tests cover concurrent isolation, response correlation, timeouts, late responses, instance restart/unload, PID reuse, Windows file contention, malformed-request idle behavior, tool discovery and validation, pagination, artifacts, event restoration, resource exports, and pipeline differences.

`smoke_test.py` starts two dedicated hidden RenderDoc processes and two real stdio clients, then closes the test processes. By default, it creates red/blue image captures and exercises real RenderDoc texture replay. Pass `--capture D:/captures/frame.rdc` to test a captured frame. Reports, logs, and exports are saved under `.test-artifacts/`. The script does not install the extension or replace captures in existing user windows.

`scripts/benchmark_idle.py` measures connected idle CPU time and the initial tool catalog size for one and two clients. Its optional `--baseline` argument accepts a project containing the original source and lockfile. CPU percentages use one logical core as 100%; a zero in a short sample means no growth was measured at OS timer precision.

Live image-capture tests do not establish coverage for complex game frames, D3D12/Vulkan/OpenGL workloads, or hardware-specific GPU counters.

## License

MIT
