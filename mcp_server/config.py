"""Configuration shared by local MCP clients."""

import os
import tempfile
from pathlib import Path


class Settings:
    def __init__(self):
        self.ipc_dir = Path(os.environ.get(
            "RENDERDOC_MCP_IPC_DIR", str(Path(tempfile.gettempdir()) / "renderdoc_mcp")
        ))
        self.timeout = float(os.environ.get("RENDERDOC_MCP_TIMEOUT", "30"))
        self.artifact_dir = Path(os.environ.get(
            "RENDERDOC_MCP_ARTIFACT_DIR", str(self.ipc_dir / "artifacts")
        ))


settings = Settings()
