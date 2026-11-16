"""Typed decoding of buffer elements and vertex formats (Python 3.6, no renderdoc import)."""

import math
import re
import struct

# name: (struct code, bytes per component, interpretation)
SCALARS = {
    "float": ("f", 4, "float"), "half": ("e", 2, "float"), "double": ("d", 8, "float"),
    "int": ("i", 4, "int"), "uint": ("I", 4, "int"), "bool": ("I", 4, "int"),
    "int8": ("b", 1, "int"), "uint8": ("B", 1, "int"), "int16": ("h", 2, "int"),
    "uint16": ("H", 2, "int"), "int64": ("q", 8, "int"), "uint64": ("Q", 8, "int"),
    "unorm8": ("B", 1, "unorm"), "snorm8": ("b", 1, "snorm"),
    "unorm16": ("H", 2, "unorm"), "snorm16": ("h", 2, "snorm"),
}
PACKED = {"r10g10b10a2_unorm": 4, "r10g10b10a2_uint": 4, "r11g11b10_float": 4}
_HLSL = re.compile(r"^(float|half|double|int|uint|bool)([1-4])?(?:x([1-4]))?$")
_SIZED = re.compile(r"^(int8|uint8|int16|uint16|int64|uint64|unorm8|snorm8|unorm16|snorm16)(?:x([1-4]))?$")
TYPE_PATTERN = (r"^((float|half|double|int|uint|bool)([1-4](x[1-4])?)?"
                r"|(int8|uint8|int16|uint16|int64|uint64|unorm8|snorm8|unorm16|snorm16)(x[1-4])?"
                r"|r10g10b10a2_unorm|r10g10b10a2_uint|r11g11b10_float)$")


def parse_type(text):
    """'float3', 'float4x4', 'uint16x2', 'unorm8x4', 'r11g11b10_float' -> spec dict."""
    text = text.strip().lower()
    if text in PACKED:
        return {"type": text, "packed": text, "count": 4 if "a2" in text else 3, "bytes": 4}
    match = _HLSL.match(text)
    if match:
        base, rows, cols = match.group(1), int(match.group(2) or 1), int(match.group(3) or 1)
        count = rows * cols
    else:
        match = _SIZED.match(text)
        if not match:
            raise ValueError("Unknown element type: %s" % text)
        base, count = match.group(1), int(match.group(2) or 1)
    code, width, kind = SCALARS[base]
    return {"type": text, "code": code, "width": width, "kind": kind, "count": count,
            "bytes": width * count}


def _json_number(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _small_float(bits, mantissa_bits):
    exponent, mantissa = bits >> mantissa_bits, bits & ((1 << mantissa_bits) - 1)
    if exponent == 0:
        return mantissa / float(1 << mantissa_bits) * 2.0 ** -14
    if exponent == 31:
        return float("nan") if mantissa else float("inf")
    return 2.0 ** (exponent - 15) * (1.0 + mantissa / float(1 << mantissa_bits))


def decode(spec, raw, offset=0):
    """Decode one element; None if raw is too short. Non-finite floats become strings."""
    if offset < 0 or offset + spec["bytes"] > len(raw):
        return None
    packed = spec.get("packed")
    if packed:
        word = struct.unpack_from("<I", raw, offset)[0]
        if packed == "r11g11b10_float":
            values = [_small_float(word & 0x7FF, 6), _small_float((word >> 11) & 0x7FF, 6),
                      _small_float(word >> 22, 5)]
        else:
            values = [(word >> shift) & 0x3FF for shift in (0, 10, 20)] + [word >> 30]
            if packed == "r10g10b10a2_unorm":
                values = [v / 1023.0 for v in values[:3]] + [values[3] / 3.0]
    else:
        values = list(struct.unpack_from("<" + spec["code"] * spec["count"], raw, offset))
        if spec["kind"] == "unorm":
            scale = float((1 << (8 * spec["width"])) - 1)
            values = [v / scale for v in values]
        elif spec["kind"] == "snorm":
            scale = float((1 << (8 * spec["width"] - 1)) - 1)
            values = [max(v / scale, -1.0) for v in values]
    if spec.get("bgra") and len(values) >= 3:
        values[0], values[2] = values[2], values[0]
    return [_json_number(v) for v in values]


def format_spec(type_name, comp_name, count, width, bgra=False):
    """Spec for a RenderDoc ResourceFormat given enum names, or None when unsupported."""
    spec = None
    if type_name == "R10G10B10A2" and comp_name in ("UNorm", "UInt"):
        spec = parse_type("r10g10b10a2_" + comp_name.lower())
    elif type_name == "R11G11B10":
        spec = parse_type("r11g11b10_float")
    elif type_name == "Regular" and 1 <= count <= 4:
        names = {
            "Float": {2: "half", 4: "float", 8: "double"},
            "UNorm": {1: "unorm8", 2: "unorm16"}, "UNormSRGB": {1: "unorm8"},
            "SNorm": {1: "snorm8", 2: "snorm16"},
            "UInt": {1: "uint8", 2: "uint16", 4: "uint", 8: "uint64"},
            "UScaled": {1: "uint8", 2: "uint16", 4: "uint"},
            "SInt": {1: "int8", 2: "int16", 4: "int", 8: "int64"},
            "SScaled": {1: "int8", 2: "int16", 4: "int"},
            "Typeless": {1: "uint8", 2: "uint16", 4: "uint", 8: "uint64"},
        }
        base = names.get(comp_name, {}).get(width)
        if base:
            spec = parse_type(base + (("x%d" % count) if "8" in base or "16" in base or "64" in base
                                      else ("%d" % count if count > 1 else "")))
    if spec is not None and bgra:
        spec["bgra"] = True
    return spec


def decode_fields(fields, raw, offset=0):
    """fields: [{"name", "offset", "spec"}] -> {name: values}; scalars are unwrapped."""
    element = {}
    for field in fields:
        values = decode(field["spec"], raw, offset + field["offset"])
        element[field["name"]] = values[0] if values is not None and len(values) == 1 else values
    return element
