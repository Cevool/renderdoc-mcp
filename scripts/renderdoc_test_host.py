"""Run with qrenderdoc --python, in a dedicated hidden process (not the user's window).

RENDERDOC_MCP_TEST_CONTROL is a JSON file containing repository/root/ready/stop paths
and an optional capture. This uses real Qt notifications and the installed RenderDoc API.
"""

import json
import os
import sys
import traceback

def run():
    with open(os.environ["RENDERDOC_MCP_TEST_CONTROL"], "r", encoding="utf-8") as stream:
        config = json.load(stream)
    def phase(name):
        with open(config["ready"] + ".phase.json", "w", encoding="utf-8") as stream:
            json.dump({"phase": name, "pid": os.getpid()}, stream)
    phase("Python script started")
    sys.path.insert(0, config["repository"])
    os.environ["RENDERDOC_MCP_IPC_DIR"] = config["ipc_dir"]
    import renderdoc_extension

    try:
        import renderdoc as rd
        from PySide2.QtCore import QFileSystemWatcher, QEventLoop, QTimer
        pyrenderdoc = globals()["pyrenderdoc"]
        # Close disables persistence while retaining in-memory settings. Dedicated test hosts
        # must not race each other writing the user's UI.config on exit.
        pyrenderdoc.Config().Close()
        if config.get("capture"):
            phase("loading capture")
            pyrenderdoc.LoadCapture(config["capture"], rd.ReplayOptions(),
                                   config["capture"], False, True)
        phase("registering bridge")
        renderdoc_extension.register(rd.GetVersionString(), pyrenderdoc)
        loop = QEventLoop()
        timer = QTimer()
        def check_stop(*args):
            if os.path.exists(config["stop"]):
                loop.quit()
        timer.timeout.connect(check_stop)
        timer.start(5000)
        watcher = QFileSystemWatcher([os.path.dirname(config["stop"])])
        watcher.directoryChanged.connect(check_stop)
        replay = {"textures": []}
        if pyrenderdoc.IsCaptureLoaded():
            pyrenderdoc.Replay().BlockInvoke(lambda controller: replay.update({
                "textures": [str(texture.resourceId) for texture in controller.GetTextures()]
            }))
        with open(config["ready"], "w", encoding="utf-8") as stream:
            json.dump({"instance_id": renderdoc_extension._server.bridge.instance_id,
                       "pid": os.getpid(), "python": sys.version,
                       "capture_loaded": pyrenderdoc.IsCaptureLoaded(),
                       "event_id": pyrenderdoc.CurEvent(), **replay}, stream)
        phase("ready")
        loop.exec_()
        timer.stop()
    except Exception:
        with open(config["ready"] + ".error", "w", encoding="utf-8") as stream:
            stream.write(traceback.format_exc())
    finally:
        renderdoc_extension.unregister()


try:
    run()
except Exception:
    with open(os.environ["RENDERDOC_MCP_TEST_CONTROL"] + ".error", "w", encoding="utf-8") as stream:
        stream.write(traceback.format_exc())
sys.exit(0)
