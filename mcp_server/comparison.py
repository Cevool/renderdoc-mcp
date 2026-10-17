"""Compare JSON evidence without importing RenderDoc's embedded runtime."""


def incomplete(value):
    if isinstance(value, dict):
        return bool(value.get("unavailable") or value.get("error") or value.get("truncated")
                    or value.get("complete") is False or value.get("status") in ("unsupported", "error")) or any(
            incomplete(child) for child in value.values())
    if isinstance(value, list):
        return any(incomplete(child) for child in value)
    return False


def differences(before, after, path=""):
    if isinstance(before, dict) and isinstance(after, dict):
        result = []
        for key in sorted(set(before) | set(after)):
            child = path + "/" + str(key).replace("~", "~0").replace("/", "~1")
            if key not in before:
                result.append({"path": child, "change": "added", "after": after[key]})
            elif key not in after:
                result.append({"path": child, "change": "removed", "before": before[key]})
            else:
                result.extend(differences(before[key], after[key], child))
        return result
    if isinstance(before, list) and isinstance(after, list) and len(before) == len(after):
        result = []
        for index, (left, right) in enumerate(zip(before, after)):
            result.extend(differences(left, right, path + "/" + str(index)))
        return result
    if before != after:
        return [{"path": path, "change": "changed", "before": before, "after": after}]
    return []
