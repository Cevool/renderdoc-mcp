"""Bounded tool results and immutable local artifacts."""

import json
import re
import uuid
from pathlib import Path

INLINE_BUDGET = 24000


class ArtifactStore:
    def __init__(self, directory: Path):
        self.directory = directory

    def allocate(self, suffix: str) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        return self.directory / (uuid.uuid4().hex + suffix)

    def describe(self, path: Path) -> dict:
        return {"artifact_id": path.name, "path": str(path.resolve()),
                "bytes": path.stat().st_size, "artifact_format": path.suffix[1:]}

    def bound(self, value):
        text = json.dumps(value, ensure_ascii=False, indent=2)
        if len(text.encode("utf-8")) <= INLINE_BUDGET:
            return value
        path = self.allocate(".json")
        path.write_text(text, encoding="utf-8")
        return {"stored": True, **self.describe(path),
                "hint": "Full result saved. Discover read_artifact for bounded text chunks."}

    def read(self, artifact_id: str, offset: int, length: int) -> dict:
        if not re.fullmatch(r"[0-9a-f]{32}\.json", artifact_id):
            raise ValueError("Only JSON artifact IDs returned by this server can be read as text")
        path = self.directory / artifact_id
        with path.open("r", encoding="utf-8") as stream:
            # Offset is in Unicode characters, not encoded bytes.
            remaining = offset
            while remaining:
                chunk = stream.read(min(remaining, 16000))
                if not chunk:
                    break
                remaining -= len(chunk)
            text = stream.read(length)
            more = bool(stream.read(1))
        return {"artifact_id": artifact_id, "offset": offset, "text": text,
                "next_offset": offset + len(text) if more else None}


def page_result(value: dict, key: str, offset: int, limit: int, fields=None) -> dict:
    items = value.get(key, [])
    selected = items[offset:offset + limit]
    if fields:
        selected = [{key: item[key] for key in fields if key in item} for item in selected]
    return {**{k: v for k, v in value.items() if k != key}, key: selected,
            "total": len(items), "offset": offset,
            "next_offset": offset + len(selected) if offset + len(selected) < len(items) else None}


def flatten_actions(actions, parent=None):
    for action in actions:
        yield {**{k: v for k, v in action.items() if k != "children"}, "parent_event_id": parent}
        yield from flatten_actions(action.get("children", []), action["event_id"])
