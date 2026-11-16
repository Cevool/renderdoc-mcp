"""Replay edits, numeric texture evidence and resource bundles (server side)."""

import base64
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Annotated, Literal

from fastmcp.exceptions import ToolError
from pydantic import Field

from .comparison import incomplete

Event = Annotated[int, Field(ge=0)]
Offset = Annotated[int, Field(ge=0)]
Stage = Literal["vertex", "hull", "domain", "geometry", "pixel", "compute"]
Cast = Literal["Typeless", "Float", "UNorm", "UNormSRGB", "SNorm", "UInt", "SInt", "Depth"]
EvidenceCast = Literal["Auto", "Typeless", "Float", "UNorm", "UNormSRGB", "SNorm", "UInt", "SInt", "Depth"]
Encoding = Literal["HLSL", "GLSL", "Slang", "DXBC", "DXIL", "SPIRV", "SPIRVAsm",
                   "OpenGLSPIRV", "OpenGLSPIRVAsm"]
TEXT_ENCODINGS = {"HLSL", "GLSL", "Slang", "SPIRVAsm", "OpenGLSPIRVAsm"}
EVIDENCE_SCHEMA = "renderdoc-mcp-texture-evidence/1"
BUNDLE_SCHEMA = "renderdoc-mcp-resource-bundle/1"
MAX_SHADER_BYTES = 16 * 1024 * 1024
EDIT_TIMEOUT = 180.0
EXPORT_TIMEOUT = 900.0


def compare_evidence(before, after, absolute_tolerance=0.0, relative_tolerance=0.0):
    if not all(math.isfinite(x) and x >= 0 for x in (absolute_tolerance, relative_tolerance)):
        raise ValueError("Tolerances must be finite and nonnegative")
    for field in ("width", "height", "components", "format", "numeric_type", "type_cast", "value_space"):
        if before.get(field) != after.get(field):
            raise ValueError("Incompatible evidence: " + field)
    left, right = before["samples"], after["samples"]
    if not left or [(p["x"], p["y"]) for p in left] != [(p["x"], p["y"]) for p in right]:
        raise ValueError("Evidence must contain identical, nonempty sampling coordinates")
    errors, changed, nonfinite = [], [], 0
    for a, b in zip(left, right):
        if len(a["value"]) != len(b["value"]) or len(a["value"]) != before["components"]:
            raise ValueError("Malformed sample component count")
        differs = False
        for x, y in zip(a["value"], b["value"]):
            if (type(x) not in (float, int) or type(y) not in (float, int)
                    or not math.isfinite(x) or not math.isfinite(y)):
                nonfinite += 1
                differs |= x != y
                continue
            delta = abs(x - y)
            errors.append(delta)
            differs |= delta > absolute_tolerance + relative_tolerance * max(abs(x), abs(y))
        if differs:
            changed.append({"x": a["x"], "y": a["y"], "before": a["value"], "after": b["value"]})

    def raw_ready(value):
        size, check, sha = value.get("raw_bytes"), value.get("raw_readback", {}), value.get("raw_sha256")
        return (type(size) is int and size > 0 and isinstance(sha, str)
                and re.fullmatch(r"[0-9a-f]{64}", sha) is not None
                and check.get("size_verified") is True
                and check.get("actual_bytes") == check.get("expected_bytes") == size)

    raw_available = raw_ready(before) and raw_ready(after)
    raw_equal = (before["raw_sha256"] == after["raw_sha256"]) if raw_available else None
    complete = not (incomplete(before) or incomplete(after))
    keys = ("capture", "event_id", "resource_id", "mip", "slice", "sample")
    return {"status": "ok" if complete else "partial", "complete": complete,
            "sample_equal": not changed, "raw_equal": raw_equal,
            "raw_comparison_available": raw_available,
            "sample_count": len(left), "changed_samples": len(changed),
            "nonfinite_components": nonfinite,
            "max_absolute_error": max(errors) if errors else None,
            "mean_absolute_error": sum(errors) / len(errors) if errors else None,
            "rms_error": math.sqrt(sum(e * e for e in errors) / len(errors)) if errors else None,
            "absolute_tolerance": absolute_tolerance, "relative_tolerance": relative_tolerance,
            "changes": changed[:256], "changes_truncated": len(changed) > 256,
            "before": {k: before.get(k) for k in keys}, "after": {k: after.get(k) for k in keys},
            "scope": ("Numeric metrics cover sampled pixels only. raw_equal compares whole-subresource "
                      "hashes when both readbacks have verified sizes. Check capture.edit_generation "
                      "and shader_replacements to confirm which replay state each record saw.")}


def validate_bundle(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text("utf-8"))
    if manifest.get("schema") != BUNDLE_SCHEMA:
        raise ValueError("Unknown resource bundle schema")
    files = manifest.get("files", [])
    if not isinstance(files, list) or len(files) > 100000:
        raise ValueError("Invalid manifest file list")
    checked, errors, seen = 0, [], set()
    for record in files:
        name = record["file"]
        path = (directory / name).resolve()
        if path.parent != directory or name in seen:
            raise ValueError("Bundle path escapes its directory or repeats a file")
        seen.add(name)
        if not path.is_file():
            errors.append({"file": name, "error": "Missing file"})
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if path.stat().st_size != record["bytes"] or digest.hexdigest() != record["sha256"]:
            errors.append({"file": name, "error": "Size or hash mismatch"})
        checked += 1
    return {"valid": not errors, "checked_files": checked, "errors": errors,
            "export_complete": bool(manifest.get("complete")),
            "omission_count": len(manifest.get("omissions", [])),
            "scope": "File integrity only; omissions are listed in manifest.json"}


def read_shader_source(source, source_path, encoding):
    if (source is None) == (source_path is None):
        raise ToolError("Provide exactly one of source or source_path")
    if source is not None:
        if encoding not in TEXT_ENCODINGS:
            raise ToolError(encoding + " is binary; pass the compiled file through source_path")
        return source.encode("utf-8")
    path = Path(source_path)
    if not path.is_absolute() or not path.is_file():
        raise ToolError("source_path must be an existing absolute file")
    if path.stat().st_size > MAX_SHADER_BYTES:
        raise ToolError("Shader file exceeds 16 MiB")
    return path.read_bytes()


def register_workbench_tools(mcp, invoke, artifacts):
    def saved(value):
        path = artifacts.allocate(".json")
        path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
        return artifacts.bound({**value, **artifacts.describe(path)})

    def evidence(artifact_id):
        if not re.fullmatch(r"[0-9a-f]{32}\.json", artifact_id):
            raise ToolError("Expected an artifact_id returned by capture_texture_evidence")
        try:
            value = json.loads((artifacts.directory / artifact_id).read_text("utf-8"))
        except OSError as exc:
            raise ToolError(str(exc)) from exc
        if value.get("schema") != EVIDENCE_SCHEMA:
            raise ToolError("Artifact is not texture evidence")
        return value

    @mcp.tool
    def replace_shader(instance_id: str, event_id: Event, stage: Stage,
                       source: Annotated[str, Field(min_length=1, max_length=4194304)] | None = None,
                       source_path: str | None = None, encoding: Encoding = "HLSL",
                       entry_point: str | None = None,
                       compile_flags: dict[str, str] | None = None) -> dict:
        """Compile a shader and replace the one bound at event/stage for every use in this replay.

        Give exactly one of source (text) or source_path (absolute file; required for binary DXBC/
        DXIL/SPIR-V). entry_point and compile_flags default to the captured shader's values;
        compile_flags entries override by name. A failed compile changes nothing and returns
        compiler_output. Calling again on the same shader swaps candidates. Undo with revert_shader.
        Shaders already edited in RenderDoc's own Shader Viewer are refused.
        """
        data = read_shader_source(source, source_path, encoding)
        return artifacts.bound(invoke(instance_id, "replace_shader", _timeout=EDIT_TIMEOUT,
            event_id=event_id, stage=stage, encoding=encoding,
            source_base64=base64.b64encode(data).decode("ascii"),
            entry_point=entry_point, compile_flags=compile_flags))

    @mcp.tool
    def revert_shader(instance_id: str, shader_id: str | None = None) -> dict:
        """Remove MCP shader replacements: one original shader_id, or all when omitted."""
        return artifacts.bound(invoke(instance_id, "revert_shader", shader_id=shader_id))

    @mcp.tool
    def list_shader_replacements(instance_id: str) -> dict:
        """List shader replacements made through MCP in the current capture session."""
        return artifacts.bound(invoke(instance_id, "list_shader_replacements"))

    @mcp.tool
    def capture_texture_evidence(instance_id: str, event_id: Event, resource_id: str,
            mip: Offset = 0, slice: Offset = 0, sample: Offset = 0,
            grid: Annotated[int, Field(ge=1, le=32)] = 8, type_cast: EvidenceCast = "Auto",
            points: Annotated[list[tuple[Offset, Offset]], Field(min_length=1, max_length=1024)] | None = None,
            histogram_range: tuple[float, float] | None = None,
            hash_byte_limit: Annotated[int, Field(ge=0, le=268435456)] = 67108864) -> dict:
        """Save decoded pixel values, per-channel min/max, optional histogram and raw hash at one event.

        Samples a grid x grid lattice unless explicit [x, y] points (top-left texels) are given.
        type_cast Auto uses the texture format, or for typeless textures the view bound at the event.
        Values are full precision (HDR/negative preserved). Records the capture session and MCP
        edit generation; pass two saved artifact_ids to compare_texture_evidence for A/B.
        """
        value = invoke(instance_id, "texture_evidence", event_id=event_id, resource_id=resource_id,
                       mip=mip, slice=slice, sample=sample, grid=grid, type_cast=type_cast,
                       hash_byte_limit=hash_byte_limit, histogram_range=histogram_range,
                       points=[list(p) for p in points] if points else None)
        return saved({"schema": EVIDENCE_SCHEMA, "instance_id": instance_id, **value})

    @mcp.tool
    def compare_texture_evidence(before_artifact_id: str, after_artifact_id: str,
                                 absolute_tolerance: float = 0.0, relative_tolerance: float = 0.0) -> dict:
        """Compare two saved texture evidence records (e.g. before/after replace_shader)."""
        try:
            return saved(compare_evidence(evidence(before_artifact_id), evidence(after_artifact_id),
                                          absolute_tolerance, relative_tolerance))
        except ValueError as exc:
            raise ToolError(str(exc)) from exc

    @mcp.tool
    def fingerprint_event_outputs(instance_id: str, event_id: Event,
            grid: Annotated[int, Field(ge=1, le=32)] = 8,
            hash_byte_limit: Annotated[int, Field(ge=0, le=268435456)] = 67108864) -> dict:
        """Evidence for every bound color/depth output at an event (first mip/layer, sample 0; no UAVs)."""
        return saved(invoke(instance_id, "fingerprint_outputs", event_id=event_id, grid=grid,
                            hash_byte_limit=hash_byte_limit))

    @mcp.tool
    def export_resource_bundle(instance_id: str, event_id: Event,
            resources: Literal["all", "none"] | Annotated[list[str], Field(min_length=1, max_length=256)] = "all",
            byte_limit: Annotated[int, Field(ge=1024, le=2147483648)] = 268435456,
            output_dir: str | None = None) -> dict:
        """Export an event to a directory: shader binaries, disassembly, reflection, embedded source,
        decoded + raw constant buffers, and bound buffers/textures with hashes.

        resources="none" exports only shaders and constants; a list limits exports to those bound
        resource IDs. Contents are the replayed state after the event. Budget overflows and
        unsupported readbacks are listed as omissions. output_dir must be a new absolute path.
        """
        if output_dir is not None:
            directory = Path(output_dir)
            if not directory.is_absolute() or directory.exists():
                raise ToolError("output_dir must be a new absolute directory")
        else:
            directory = artifacts.allocate(".bundle")
        value = invoke(instance_id, "export_resource_bundle", _timeout=EXPORT_TIMEOUT,
                       event_id=event_id, output_dir=str(directory.resolve()),
                       byte_limit=byte_limit, resources=resources)
        summary = {key: value.get(key) for key in ("event_id", "capture", "complete", "data_bytes",
                                                     "byte_limit", "resources", "scope")}
        kinds = {}
        for record in value.get("files", []):
            kinds[record["kind"]] = kinds.get(record["kind"], 0) + 1
        return artifacts.bound({**summary, "bundle": str(directory.resolve()),
                                "manifest": str((directory / "manifest.json").resolve()),
                                "file_counts": kinds, "omissions": value.get("omissions", [])[:50],
                                "omission_count": len(value.get("omissions", [])),
                                "validation": validate_bundle(directory)})

    @mcp.tool
    def validate_resource_bundle(bundle: str) -> dict:
        """Recheck sizes/hashes of a bundle directory returned by export_resource_bundle."""
        directory = Path(bundle)
        if not directory.is_absolute():
            directory = artifacts.directory / bundle
        try:
            return artifacts.bound(validate_bundle(directory))
        except (OSError, ValueError, KeyError) as exc:
            raise ToolError(str(exc)) from exc
