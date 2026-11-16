"""Element decoding, topology expansion and mesh writing, without a GPU."""

import importlib.util
import re
import struct
from pathlib import Path

import pytest

from mcp_server.inspection import TYPE_PATTERN

UTILS = Path(__file__).resolve().parents[1] / "renderdoc_extension" / "utils"


def standalone(name):
    # Loaded by path: the utils package imports renderdoc, which other tests fake differently.
    spec = importlib.util.spec_from_file_location("standalone_" + name, UTILS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


elements = standalone("elements")
geometry = standalone("geometry")
decode, format_spec, parse_type = elements.decode, elements.format_spec, elements.parse_type
primitives, write_mesh = geometry.primitives, geometry.write_mesh


def test_server_type_pattern_matches_extension_parser():
    assert TYPE_PATTERN == elements.TYPE_PATTERN
    for text in ("float", "float3", "float4x4", "uint2", "half4", "bool", "uint16x2", "unorm8x4",
                 "snorm16", "int64", "r10g10b10a2_unorm", "r11g11b10_float"):
        assert re.match(TYPE_PATTERN, text) and parse_type(text)
    for text in ("float5", "uint8x5", "vec3", "unorm32"):
        assert not re.match(TYPE_PATTERN, text)
        with pytest.raises(ValueError):
            parse_type(text)


def test_decodes_normalized_packed_and_bgra_formats():
    assert decode(parse_type("float4x4"), struct.pack("<16f", *range(16))) == [float(v) for v in range(16)]
    assert decode(parse_type("half2"), struct.pack("<2e", 1.5, float("inf"))) == [1.5, "inf"]
    assert decode(parse_type("snorm8x4"), bytes([127, 0x81, 0x80, 0])) == [1.0, -1.0, -1.0, 0.0]
    assert decode(parse_type("unorm16"), struct.pack("<H", 65535)) == [1.0]
    word = 1023 | (0 << 10) | (512 << 20) | (3 << 30)
    assert decode(parse_type("r10g10b10a2_uint"), struct.pack("<I", word)) == [1023, 0, 512, 3]
    # r11g11b10: 1.0 in each channel (exponent 15, mantissa 0).
    word = (15 << 6) | ((15 << 6) << 11) | ((15 << 5) << 22)
    assert decode(parse_type("r11g11b10_float"), struct.pack("<I", word)) == [1.0, 1.0, 1.0]
    bgra = format_spec("Regular", "UNorm", 4, 1, True)
    assert decode(bgra, bytes([255, 0, 0, 255])) == [0.0, 0.0, 1.0, 1.0]  # B8G8R8A8: blue first.
    assert format_spec("Regular", "Float", 3, 4)["type"] == "float3"
    assert format_spec("Regular", "UInt", 4, 1)["type"] == "uint8x4"
    assert format_spec("Regular", "Depth", 1, 4) is None
    assert decode(parse_type("float3"), b"\0" * 8) is None  # Short reads are not padded.


def test_topologies_expand_with_winding_and_restart():
    assert primitives("TriangleStrip", [0, 1, 2, 3, None, 4, 5, 6]) == (
        "triangles", [(0, 1, 2), (2, 1, 3), (4, 5, 6)])
    assert primitives("TriangleFan", [0, 1, 2, 3]) == ("triangles", [(0, 1, 2), (0, 2, 3)])
    assert primitives("TriangleList_Adj", list(range(6))) == ("triangles", [(0, 2, 4)])
    assert primitives("TriangleList", [0, 1, 2, 3]) == ("triangles", [(0, 1, 2)])
    assert primitives("LineStrip", [0, 1, 2]) == ("lines", [(0, 1), (1, 2)])
    assert primitives("PatchList_3CPs", [0, 1, 2]) == ("points", [])


def test_obj_and_ply_share_vertices_and_zero_nonfinite(tmp_path):
    vertices = [{"position": [0, 0, 0], "uv": [0, 0]}, {"position": [1, 0, 0], "uv": [1, 0]},
                {"position": ["nan", 1, 0], "uv": [0, 1]}]
    summary = write_mesh(str(tmp_path / "m.obj"), "obj", vertices, "triangles", [(0, 1, 2)], ["test"])
    text = (tmp_path / "m.obj").read_text()
    assert "v 0 0 0\n" in text and "f 1/1 2/2 3/3\n" in text and "vn" not in text
    assert summary["nonfinite_vertices_zeroed"] == 1 and summary["has_uvs"]
    write_mesh(str(tmp_path / "m.ply"), "ply", vertices, "triangles", [(0, 1, 2)], ["test"])
    ply = (tmp_path / "m.ply").read_text().split("end_header\n")
    assert "element face 1" in ply[0] and ply[1].splitlines()[-1] == "3 0 1 2"
    with pytest.raises(FileExistsError):
        write_mesh(str(tmp_path / "m.obj"), "obj", vertices, "triangles", [], [])
