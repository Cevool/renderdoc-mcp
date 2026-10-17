# RenderDoc MCP Server

Give AI clients structured access to running RenderDoc instances. Version 2.1 adds a category directory, explicit search completeness, and deep GPU evidence queries while retaining isolated replay sessions and bounded results.

## Setup and migration

The server requires Python 3.10+. The recommended setup uses [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage an isolated project environment. Development and integration tests use Windows, Python 3.14.4, uv 0.11.12, and RenderDoc 1.46 with its embedded Python 3.8. The extension retains Python 3.6 syntax compatibility and uses only the standard library and RenderDoc's bundled modules; FastMCP is installed in the external server environment.

Run from the repository root. These commands create the project virtual environment and install its locked dependencies:

```powershell
uv sync --locked
uv run --locked python scripts/install_extension.py
```

The installer copies the extension to `%APPDATA%\qrenderdoc\extensions\renderdoc_mcp_bridge`. Enable **RenderDoc MCP Bridge** under **Tools > Manage Extensions** in RenderDoc.

**Upgrading from 1.x requires updating both the server and the extension.** Reload the extension or restart RenderDoc, then restart the MCP connection. Save captures you need before restarting existing windows. The new protocol does not consume the old shared queue.

For **2.0 to 2.1**, run the extension installer again and reload the extension, then reconnect the MCP client. The IPC protocol remains version 2; new analysis operations require extension 2.1. Search now returns a paginated object instead of an unlabelled array; its default records are summaries. Use `get_tool_schema` or `detail="schema"` for complete definitions.

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

## Category-first tool discovery

| Tool | Purpose |
| --- | --- |
| `list_instances` | Discover live instances, including instance ID, PID, version, and capture filename |
| `browse_tools` | Without arguments, list every category and count; with a category, page through operation names and purposes |
| `get_tool_schema` | Return one full definition by exact operation name |
| `search_tools` | Optional regex shortcut with totals, pagination and exact-name priority |
| `call_tool` | Execute an operation by name with its arguments |

Example workflow:

```text
list_instances()
browse_tools()
browse_tools(category="pipeline")
get_tool_schema(name="get_native_pipeline_state")
call_tool(name="get_native_pipeline_state", arguments={"instance_id":"<discovered ID>","event_id":123})
```

The directory groups operations into `capture`, `actions`, `pipeline`, `resources`, `analysis`, and `artifacts`. It is built from the callable catalog, respects visibility/auth filtering, and omits parameter schemas until requested. This supports capability exploration without guessing operation names or interpreting a short search as a complete inventory. Five small entry points are advertised; the 28 operation schemas remain hidden.

Search remains useful when the target is known:

```text
search_tools(pattern="texture", limit=3)
# {tools:[...], returned:3, total_matches:4, truncated:true, next_offset:3, ...}
search_tools(pattern="texture", offset=3, limit=3)
search_tools(pattern="^get_pipeline_state$", detail="schema")
call_tool(name="get_pipeline_state", arguments={"instance_id":"<discovered ID>","event_id":123})
```

Search matches names, descriptions, and parameter names with a case-insensitive, bounded Rust regex engine. Invalid/unsupported expressions return an error. Default pages contain 10 summaries, up to 50; `detail="schema"` returns up to three full definitions per page with the same completeness metadata. `get_tool_schema` always selects exactly one name. Every RenderDoc operation requires an explicit instance ID; there is no global selection for clients to overwrite. Closing an instance or reloading its extension invalidates its ID, so discover it again.

The original capture management, action queries, shader/resource searches, GPU timings, pipeline queries, and texture/buffer operations remain available through discovery. Additional operations include:

- `get_pipeline_states`: inspect up to 32 events in a serialized batch.
- `compare_pipeline_states`: return changed values using JSON Pointer paths. Incomplete queries return `complete=false` and `equal=null` rather than claiming equality.
- `export_texture`: convert a texture mip/slice to PNG using RenderDoc.
- `view_image_artifact`: return an exported PNG as an MCP image.
- `read_artifact`: read bounded chunks of a saved JSON result.

## GPU evidence queries

| Operation | Evidence and limits |
| --- | --- |
| `get_native_pipeline_state` | Native rasterizer, depth/stencil, blend, viewport/scissor and attachment state. Vulkan also includes `currentPass`. Native member names and enum names/numbers are preserved; no cross-API normalization is implied. |
| `get_constant_buffer` | Reflected declarations, actual resource/offset/range, decoded values and explicit unbound state. Indexes the reflection block list, not a register. `raw_read` provides a complete byte-export call with the same instance ID. Push constants are decoded without requiring a backing buffer. |
| `get_post_vs_data` | VS or final geometry stage output, layout, resolved indices, raw records on request, floating point positions, finite and zero-W checks. Pages index rows, honors base vertex/restart, and makes no visibility/degeneracy claim beyond the returned sample. |
| `get_pixel_history` | Modifications through an explicit event, pre/shader/post values and native rejection flags including backface, depth, stencil and discard. Coordinates always start at the top left. Unsupported is distinct from empty. |
| `get_api_events` | API names, chunk indexes, action flags and optional structured parameters over at most 4096 event IDs. Clear values, pass load/store fields and resource references are exposed where recorded; nothing is inferred from first use. |
| `get_resource_usage` | Recorded read/write/clear usages with event IDs and view IDs, for initialization/history analysis. |
| `debug_shader` | One vertex, pixel or compute invocation; inputs/constants and instruction/variable changes. At most 10,000 saved states, 64 native batches, and a 10-second budget checked between batches. A native call itself cannot be interrupted. The trace is always freed. |
| `compare_capture_states` | Two explicit instance/event pairs and capture filenames, structural differences with JSON paths, and completeness. No automatic event/resource correspondence. |

Deep queries return `status` (`ok`, `empty`, `partial`, `unsupported`, or `error`) and `complete` where applicable. Pagination, serializer limits and stopped traces are explicit (`truncated`, totals, and/or `next_offset`). A shader trace prefix is not a finished execution; read saved JSON through `read_artifact`, or repeat with a larger bound. No native debugger or background worker is left alive between requests.

Native serializers preserve available reflected fields and report missing fields, capped arrays and unreadable values. Structured chunks are [RenderDoc's version-dependent serialized data](https://github.com/baldurk/renderdoc/blob/v1.x/docs/python_api/in_depth/structured_data.rst); they are evidence to interpret in the captured API's context. A pass snapshot alone does not prove its original clear/load/store operations. Pixel history and debugging depend on API, shader and replay-driver support. Read-only resource queries still require a valid captured resource ID.

Resource IDs are resolved from the replay resource catalog because RenderDoc's native IDs are opaque; setting a Python `.id` attribute cannot construct one. Output-attachment reverse lookups use the current common pipeline API. Sampler filters are serialized by value so identical-event comparisons do not mistake different SWIG object addresses for GPU state changes. Missing pipeline sections remain explicit even when selecting fields.

For reconstruction validation, pair these GPU queries with a freshly exported target image: verify resource usage/initialization order, actual constant bytes, sampled geometry, native fixed state and pixel history before comparing the final image. Use explicit event mappings across captures, fix the subresource/color conversion conventions, and inspect the character region separately from the background. An offline shader compile or finite Post-VS sample alone does not establish a correct image.

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

Bridge errors carry protocol/bridge/RenderDoc versions where available, instance/capture identity, request ID, operation/event/stage, execution status and whether side effects remain possible. `not_submitted` means the request was not published; `failed` means a remote error was received; `unknown` means a published request's final state cannot be confirmed. A timeout never proves that the operation did not run. Transient Windows response-file locks retry the read within the original deadline without resubmitting work.

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
uv run --locked python scripts/smoke_test.py --capture path/to/frame.rdc --deep-analysis
```

Automated tests cover concurrent isolation, response correlation, timeouts, late responses, instance restart/unload, PID reuse, Windows file contention, malformed-request idle behavior, tool discovery and validation, pagination, artifacts, event restoration, resource exports, and pipeline differences.

`smoke_test.py` starts two dedicated hidden RenderDoc processes and two real stdio clients, then closes the test processes. By default, it creates red/blue image captures and exercises real texture replay. Pass `--capture D:/captures/frame.rdc --deep-analysis` to validate native state, sampled geometry, every reflected constant block and its bytes, API events, vertex/compute shader debugging, resource usage and pixel history. The second instance uses a small PNG unless `--other-capture` is supplied. This check requires a capture with supported debuggable geometry; failures are recorded instead of treated as feature coverage.

Use `--event-id` to select a known draw instead of the first. `--pixel X Y` also validates pixel shader debugging; `--pixel-primitive` selects an unambiguous fragment and requires a matching pixel-history record at that event. `--other-capture` and `--other-event-id` test an explicit cross-capture pair, including native state. IDs are capture-local and are never automatically matched. Both replay sessions must fit available memory. The per-host startup budget defaults to 120 seconds (`--startup-timeout`); phase files distinguish startup before Python, capture loading and bridge registration from an operation timeout. Reports, logs, and exports are saved under `.test-artifacts/`. The script does not install the extension or replace captures in existing user windows.

`scripts/benchmark_idle.py` measures connected idle CPU time and the initial tool catalog size for one and two clients. Its optional `--baseline` argument accepts a project containing the original source and lockfile. CPU percentages use one logical core as 100%; a zero in a short sample means no growth was measured at OS timer precision.

Dedicated test hosts disable configuration persistence with `PersistentConfig.Close()` before loading captures. Smoke reports check that the user's `UI.config` bytes remain unchanged, avoiding concurrent writes from the two hosts. Real-capture validation on RenderDoc 1.46 covers D3D11 and Vulkan, including vertex/pixel/compute debugging and an explicit Vulkan-to-D3D11 event pair. Bounded trace prefixes are distinguished from finished traces. Image-only tests do not establish deep GPU coverage, and these results do not establish coverage for D3D12, OpenGL, other drivers, or hardware-specific counters.

## License

MIT
