"""Bounded serialization of RenderDoc's reflected native state (Python 3.6)."""

import math


def native_value(value, depth=0):
    if value is None or type(value) in (str, bool, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"non_finite": str(value)}
    if type(value).__name__ == "ResourceId":
        return str(value)
    if isinstance(value, int):
        return {"name": str(value), "value": int(value)}
    if depth >= 10:
        return {"unavailable": "Serialization depth limit", "truncated": True}
    if isinstance(value, dict):
        return {str(k): native_value(v, depth + 1) for k, v in value.items()}
    if hasattr(value, "__len__") and hasattr(value, "__getitem__"):
        count = len(value)
        items = [native_value(value[i], depth + 1) for i in range(min(count, 256))]
        if count > 256:
            return {"items": items, "total": count, "truncated": True}
        return items
    result = {}
    if type(value).__name__ == "ModificationValue":
        result["valid"] = value.IsValid()
    for key in dir(value):
        if key.startswith("_") or key in ("this", "thisown"):
            continue
        try:
            child = getattr(value, key)
            if not callable(child):
                result[key] = native_value(child, depth + 1)
        except Exception as exc:
            result.setdefault("unavailable", {})[key] = str(exc)
    if not result:
        return {"unavailable": "Unserializable native type: " + type(value).__name__}
    return result


def incomplete(value):
    if isinstance(value, dict):
        return bool(value.get("unavailable") or value.get("error") or value.get("truncated")
                    or value.get("complete") is False or value.get("status") in ("unsupported", "error")) or any(
            incomplete(child) for child in value.values())
    if isinstance(value, list):
        return any(incomplete(child) for child in value)
    return False


def result_status(result, records=None):
    result["complete"] = not incomplete(result)
    result["status"] = ("partial" if not result["complete"] else
                        "empty" if records is not None and not records else "ok")
    return result
