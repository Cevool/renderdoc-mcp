"""Index streams to primitives, and OBJ/PLY writers (Python 3.6, no renderdoc import)."""

import math


def _segments(indices, restart):
    """Split a strip/fan/loop index stream at primitive restart markers (None entries)."""
    segment = []
    for index in indices:
        if index is None:
            if segment:
                yield segment
            segment = []
        else:
            segment.append(index)
    if segment:
        yield segment


def primitives(topology, indices):
    """Return (kind, primitives) for an ordered index stream; restarts are None entries.

    kind is "triangles", "lines" or "points". Lists ignore restarts (they cannot occur).
    """
    flat = [i for i in indices if i is not None]
    if topology == "TriangleList":
        return "triangles", [tuple(flat[i:i + 3]) for i in range(0, len(flat) - 2, 3)]
    if topology == "TriangleList_Adj":
        return "triangles", [(flat[i], flat[i + 2], flat[i + 4]) for i in range(0, len(flat) - 5, 6)]
    if topology in ("TriangleStrip", "TriangleStrip_Adj", "TriangleFan"):
        result = []
        for segment in _segments(indices, True):
            if topology == "TriangleStrip_Adj":
                segment = segment[0::2]
            for i in range(len(segment) - 2):
                if topology == "TriangleFan":
                    result.append((segment[0], segment[i + 1], segment[i + 2]))
                elif i % 2 == 0:
                    result.append((segment[i], segment[i + 1], segment[i + 2]))
                else:
                    result.append((segment[i + 1], segment[i], segment[i + 2]))
        return "triangles", result
    if topology == "LineList":
        return "lines", [tuple(flat[i:i + 2]) for i in range(0, len(flat) - 1, 2)]
    if topology == "LineList_Adj":
        return "lines", [(flat[i + 1], flat[i + 2]) for i in range(0, len(flat) - 3, 4)]
    if topology in ("LineStrip", "LineStrip_Adj", "LineLoop"):
        result = []
        for segment in _segments(indices, True):
            if topology == "LineStrip_Adj":
                segment = segment[1:-1]
            result.extend((segment[i], segment[i + 1]) for i in range(len(segment) - 1))
            if topology == "LineLoop" and len(segment) > 2:
                result.append((segment[-1], segment[0]))
        return "lines", result
    # PointList, patch lists (tessellation control points) and unknown topologies.
    return "points", []


def _finite(values, size):
    values = list(values[:size]) + [0.0] * max(0, size - len(values))
    ok = all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)
    return ([float(v) for v in values] if ok else [0.0] * size), ok


def write_mesh(path, fmt, vertices, kind, prims, header):
    """vertices: [{"position": [...], "normal"?: [...], "uv"?: [...]}] in output order.

    prims index into vertices (0-based). Non-finite attributes are written as 0 and counted.
    Creates a new file; never overwrites.
    """
    has_normal = bool(vertices) and all("normal" in v for v in vertices)
    has_uv = bool(vertices) and all("uv" in v for v in vertices)
    nonfinite = 0
    rows = []
    for vertex in vertices:
        position, ok = _finite(vertex["position"], 3)
        normal, ok_n = _finite(vertex.get("normal", []), 3)
        uv, ok_uv = _finite(vertex.get("uv", []), 2)
        nonfinite += not (ok and (ok_n or not has_normal) and (ok_uv or not has_uv))
        rows.append((position, normal, uv))
    with open(path, "x", encoding="utf-8", newline="\n") as stream:
        if fmt == "obj":
            for line in header:
                stream.write("# %s\n" % line)
            for position, normal, uv in rows:
                stream.write("v %.9g %.9g %.9g\n" % tuple(position))
            if has_uv:
                for _, _, uv in rows:
                    stream.write("vt %.9g %.9g\n" % tuple(uv))
            if has_normal:
                for _, normal, _ in rows:
                    stream.write("vn %.9g %.9g %.9g\n" % tuple(normal))
            if kind == "triangles":
                def ref(i):
                    i += 1
                    if has_uv and has_normal:
                        return "%d/%d/%d" % (i, i, i)
                    if has_normal:
                        return "%d//%d" % (i, i)
                    return "%d/%d" % (i, i) if has_uv else "%d" % i
                for prim in prims:
                    stream.write("f %s\n" % " ".join(ref(i) for i in prim))
            elif kind == "lines":
                for a, b in prims:
                    stream.write("l %d %d\n" % (a + 1, b + 1))
        else:
            stream.write("ply\nformat ascii 1.0\n")
            for line in header:
                stream.write("comment %s\n" % line)
            stream.write("element vertex %d\n" % len(rows))
            stream.write("property float x\nproperty float y\nproperty float z\n")
            if has_normal:
                stream.write("property float nx\nproperty float ny\nproperty float nz\n")
            if has_uv:
                stream.write("property float s\nproperty float t\n")
            if kind == "triangles":
                stream.write("element face %d\nproperty list uchar uint vertex_indices\n" % len(prims))
            elif kind == "lines":
                stream.write("element edge %d\nproperty uint vertex1\nproperty uint vertex2\n" % len(prims))
            stream.write("end_header\n")
            for position, normal, uv in rows:
                values = position + (normal if has_normal else []) + (uv if has_uv else [])
                stream.write(" ".join("%.9g" % v for v in values) + "\n")
            for prim in prims:
                stream.write(("3 %d %d %d\n" if kind == "triangles" else "%d %d\n") % prim)
    return {"vertices": len(rows), "primitives": len(prims), "primitive_kind": kind,
            "has_normals": has_normal, "has_uvs": has_uv, "nonfinite_vertices_zeroed": nonfinite}
