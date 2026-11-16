"""Shader replacement through RenderDoc's public replay API (Python 3.6).

BuildTargetShader + ReplaceResource replace every use of the original shader in this replay,
not one event. The RDC file is never modified. Only replacements created here are reverted
here; a shader already edited in RenderDoc's own Shader Viewer is left alone.
"""

import base64

import renderdoc as rd

from ..utils import Parsers
from .replay_common import digest, event_query

ENCODINGS = ("HLSL", "GLSL", "Slang", "DXBC", "DXIL", "SPIRV", "SPIRVAsm",
             "OpenGLSPIRV", "OpenGLSPIRVAsm")
SCOPE = ("Every use of the original shader in this capture's replay, until reverted or the "
         "capture is closed. The RDC file is unchanged.")


def _key(resource_id):
    return str(Parsers.extract_numeric_id(str(resource_id)))


class EditService(object):
    def __init__(self, ctx, invoke_fn, state):
        self.ctx = ctx
        self._invoke = invoke_fn
        self.state = state

    @staticmethod
    def _public(record):
        return {key: record[key] for key in ("original_shader", "replacement_shader", "event_id",
                                             "stage", "encoding", "entry_point", "source_sha256",
                                             "compile_flags", "generation")}

    @staticmethod
    def _flags(base, overrides):
        """Start from the captured shader's flags (profile etc.); apply explicit overrides."""
        pairs = [(str(flag.name), str(flag.value)) for flag in (base.flags if base else [])]
        if not overrides:
            return base if base is not None else rd.ShaderCompileFlags(), [
                {"name": name, "value": value} for name, value in pairs]
        merged = dict(pairs)
        order = [name for name, _ in pairs]
        for name, value in sorted(overrides.items()):
            if name not in merged:
                order.append(name)
            merged[name] = str(value)
        flags = rd.ShaderCompileFlags()
        items = []
        for name in order:
            flag = rd.ShaderCompileFlag()
            flag.name, flag.value = name, merged[name]
            items.append(flag)
        flags.flags = items
        return flags, [{"name": name, "value": merged[name]} for name in order]

    def _ui_replacement(self, original):
        try:
            if self.ctx.IsResourceReplaced(original):
                return str(self.ctx.GetResourceReplacement(original))
        except Exception:
            pass
        return None

    def _ui_notify(self, register, original, target=None):
        # Keeps RenderDoc's resource lists/viewers consistent with the replay; best effort.
        try:
            if register:
                self.ctx.RegisterReplacement(original, target)
            else:
                self.ctx.UnregisterReplacement(original)
        except Exception as exc:
            print("[MCP Bridge] UI replacement notification failed: %s" % exc)

    def replace_shader(self, event_id, stage, encoding, source_base64,
                       entry_point=None, compile_flags=None):
        if encoding not in ENCODINGS or getattr(rd.ShaderEncoding, encoding, None) is None:
            raise ValueError("Encoding %s is unavailable; expected one of %s" % (encoding, ", ".join(ENCODINGS)))
        if compile_flags is not None and not (
                isinstance(compile_flags, dict) and all(isinstance(k, str) and isinstance(v, str)
                                                       for k, v in compile_flags.items())):
            raise ValueError("compile_flags must map flag names to string values")
        try:
            source = base64.b64decode(source_base64.encode("ascii"), validate=True)
        except Exception:
            raise ValueError("source_base64 is not valid Base64")
        if not source:
            raise ValueError("Empty shader source")
        stage_enum = Parsers.parse_stage(stage)
        applied, bound = {}, {}

        def resolve(controller):
            original = controller.GetPipelineState().GetShader(stage_enum)
            if original == rd.ResourceId.Null():
                raise ValueError("No %s shader bound at event %s" % (stage, event_id))
            bound["original"] = original
            return {}

        # CaptureContext replacement queries are marshalled onto the UI thread. Calling them
        # inside BlockInvoke (UI thread blocked on the replay thread) deadlocks, so check here.
        event_query(self.ctx, self._invoke, self.state, event_id, resolve)
        ui_target = self._ui_replacement(bound["original"])
        if _key(bound["original"]) not in self.state.replacements and ui_target is not None:
            raise ValueError("Shader %s is already replaced in the RenderDoc UI (by %s); "
                             "revert it there first" % (bound["original"], ui_target))

        def read(controller):
            pipe = controller.GetPipelineState()
            original = pipe.GetShader(stage_enum)
            if original != bound["original"]:
                raise ValueError("The shader bound at event %s changed; retry" % event_id)
            key = _key(original)
            previous = self.state.replacements.get(key)
            # The binding returns plain integers here, not enum members.
            values = set(int(item) for item in controller.GetTargetShaderEncodings())
            supported = [name for name in ENCODINGS
                         if getattr(rd.ShaderEncoding, name, None) is not None
                         and int(getattr(rd.ShaderEncoding, name)) in values]
            if encoding not in supported:
                raise ValueError("This replay accepts %s, not %s" % (", ".join(supported), encoding))
            reflection = pipe.GetShaderReflection(stage_enum)
            entry = entry_point or (reflection.entryPoint if reflection is not None
                                    and reflection.entryPoint else "main")
            base = reflection.debugInfo.compileFlags if reflection is not None else None
            flags, used = self._flags(base, compile_flags)
            target, messages = controller.BuildTargetShader(entry, getattr(rd.ShaderEncoding, encoding),
                                                            source, flags, stage_enum)
            result = {"original_shader": str(original), "stage": stage, "encoding": encoding,
                      "entry_point": entry, "compile_flags": used, "source_sha256": digest(source),
                      "compiler_output": str(messages or "")}
            if target == rd.ResourceId.Null():
                result.update(status="compile_failed", applied=False,
                              hint="The previous state of this shader is unchanged")
                return result
            if previous is not None:
                controller.RemoveReplacement(original)
                controller.FreeTargetResource(previous["target"])
            controller.ReplaceResource(original, target)
            self.state.generation += 1
            record = dict(result, original=original, target=target, event_id=event_id,
                          replacement_shader=str(target), generation=self.state.generation)
            self.state.replacements[key] = record
            applied.update(original=original, target=target, previous=previous)
            result.update(status="ok", applied=True, replacement_shader=str(target),
                          replaced_previous_mcp_edit=previous is not None,
                          generation=self.state.generation)
            return result

        result = event_query(self.ctx, self._invoke, self.state, event_id, read)
        if applied:
            if applied["previous"] is not None:
                self._ui_notify(False, applied["original"])
            self._ui_notify(True, applied["original"], applied["target"])
            result["scope"] = SCOPE
        return result

    def revert_shader(self, shader_id=None):
        if not self.ctx.IsCaptureLoaded():
            raise ValueError("No capture loaded")
        if shader_id is None:
            keys = sorted(self.state.replacements)
        else:
            keys = [str(Parsers.extract_numeric_id(shader_id))]
            if keys[0] not in self.state.replacements:
                raise ValueError("No MCP replacement for shader %s; see list_shader_replacements" % shader_id)
        reverted, result = [], {}

        def callback(controller):
            for key in keys:
                record = self.state.replacements.pop(key)
                controller.RemoveReplacement(record["original"])
                controller.FreeTargetResource(record["target"])
                reverted.append(record)
            if reverted:
                self.state.generation += 1
            result["capture"] = self.state.identity(controller)

        self._invoke(callback)
        for record in reverted:
            self._ui_notify(False, record["original"])
        return dict(result, status="ok", reverted=[self._public(r) for r in reverted],
                    remaining=[self._public(r) for _, r in sorted(self.state.replacements.items())])

    def list_shader_replacements(self):
        loaded = self.ctx.IsCaptureLoaded()
        return {"capture_loaded": loaded, "session_id": self.state.session,
                "edit_generation": self.state.generation,
                "replacements": [self._public(r) for _, r in sorted(self.state.replacements.items())],
                "scope": "Replacements created through MCP in this capture session only; Shader Viewer "
                         "edits made in the RenderDoc UI are not listed"}

    def shutdown(self):
        """Extension unload: revert our replacements so none are left without an owner."""
        if self.state.replacements and self.ctx.IsCaptureLoaded():
            try:
                self.revert_shader()
            except Exception as exc:
                print("[MCP Bridge] Could not revert MCP shader replacements: %s" % exc)
        self.state.reset()
