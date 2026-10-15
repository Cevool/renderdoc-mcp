"""Event-driven Qt adapter; filename retained for extension upgrade compatibility."""

from PySide2.QtCore import QFileSystemWatcher, QObject, QTimer

from .ipc import FileBridge


class MCPBridgeServer(QObject):
    def __init__(self, handler, version="", status_provider=None, parent=None):
        super(MCPBridgeServer, self).__init__(parent)
        self.bridge = FileBridge(handler, version=version, status_provider=status_provider)
        self._watcher = QFileSystemWatcher(self)
        self._watcher.directoryChanged.connect(self._drain)
        self._timer = QTimer(self)
        self._timer.setInterval(5000)
        self._timer.timeout.connect(self._maintenance)

    def start(self):
        self.bridge.start()
        self._watcher.addPath(self.bridge.requests)
        self._timer.start()
        self._drain()
        print("[MCP Bridge] Instance %s: %s" % (self.bridge.instance_id, self.bridge.directory))
        return True

    def stop(self):
        self._timer.stop()
        paths = self._watcher.directories()
        if paths:
            self._watcher.removePaths(paths)
        self.bridge.stop()

    def is_running(self):
        return self.bridge.running

    def _drain(self, *args):
        if self.bridge.drain():
            QTimer.singleShot(0, self._drain)

    def _maintenance(self):
        self.bridge.refresh()
        self._drain()
